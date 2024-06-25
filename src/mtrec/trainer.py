import torch
import os
import copy
from gradient_surgery import PCGrad
from utils_training import *

    
def train(model, dataloader_train, dataloader_val, cfg, 
          print_flag = True, save_dir:str = "saved_models", use_wandb:bool = False, 
          hypertuning:bool = False, name_run:str = "unnamed"):
    """
    Function to train the model on the given dataset.
    
    Args:
        news_encoder (torch.nn.Module): The news encoder module.
        user_encoder (torch.nn.Module): The user encoder module.
        epochs (int): The number of training epochs.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        criterion (torch.nn.Module): The loss function.
        dataloader_train (torch.utils.data.DataLoader): The dataloader for the training dataset.
        dataloader_val (torch.utils.data.DataLoader): The dataloader for the validation dataset.
        device (torch.device): The device to be used for training.
    """
    if use_wandb: import wandb
    #initialize optimizer
    user_encoder = model.user_encoder
    news_encoder = model.news_encoder
    params = [
        {"params": [user_encoder.W,  user_encoder.q],   "lr": cfg["lr_user"]},  # lr of attention layer in user encoder
        {"params": list(news_encoder.cat_net.parameters()) + list(news_encoder.ner_net.parameters()),
                   "lr": cfg["lr_news"]},  # lr of auxiliary tasks
        {"params": news_encoder.bert.parameters(), "lr": cfg["lr_bert"]}  # Parameters of BERT
    ] 


    # Perhaps learning rate decay for bert
    # optimizer_grouped_parameters = [
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'layer.11' in name or 'layer.10' in name], 'lr': 5e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'layer.9' in name or 'layer.8' in name], 'lr': 3e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'layer.7' in name or 'layer.6' in name], 'lr': 2e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'layer.5' in name or 'layer.4' in name], 'lr': 1e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'layer.3' in name or 'layer.2' in name], 'lr': 1e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'layer.1' in name or 'layer.0' in name], 'lr': 1e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'embeddings' in name], 'lr': 1e-5},
    #     {'params': [param for name, param in news_encoder.bert.named_parameters() if 'classifier' in name], 'lr': 5e-5},
    # ]
    best_model = None

    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params)
    elif cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params)
    else:
        print("Invalid optimizer <{}>.".format(cfg["optimizer"]))
        return

    #TURNEMALLON TODO: remove this?
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.requires_grad=True

    if not cfg["skip_gs"]: 
        optimizer = PCGrad(optimizer, lamb = cfg["aux_scaler"])
    
    # Initialize to track best
    best_loss = float('inf')
    save_num = 0

    # Ensure the save directory ends with a slash
    if not save_dir.endswith('/'):
        save_dir += '/'

    # Determine a unique save directory
    while os.path.exists(os.path.join(save_dir, f'{name_run}_run{save_num}')):
        save_num += 1

    save_path = os.path.join(save_dir, f'{name_run}_run{save_num}')
    os.makedirs(save_path, exist_ok=True)

    # Debugging: Confirm directory creation
    if not os.path.exists(save_path):
        raise RuntimeError(f"Failed to create directory: {save_path}")

    if print_flag: print(f"Saving models to {save_path}")
    try: # Training can be interrupted by catching KeyboardInterrupt
        for epoch in range(cfg['epochs']):
            if print_flag: print(f"Epoch {epoch} / {cfg['epochs']}")
            
            # Training
            train_loss, train_main_loss, train_cat_loss, train_ner_loss = model.train(dataloader_train, optimizer, print_flag, cfg)

            # Validation
            val_loss, val_main_loss, val_cat_loss, val_ner_loss = model.validate(dataloader_val, print_flag, cfg)

            
            # Log the losses to wandb
            if use_wandb and not hypertuning:
                wandb.log({"Training Main Loss": train_main_loss, "Training Total Loss": train_loss,
                           "Training Cat Loss": train_cat_loss, "Training NER Loss": train_ner_loss,
                            "Validation Main Loss": val_main_loss, "Validation Total Loss": val_loss,
                            "Validation Cat Loss": val_cat_loss, "Validation NER Loss": val_ner_loss})
                        
            elif use_wandb and hypertuning:
                wandb.log({"Validation Main Loss": val_main_loss, "Validation Total Loss": val_loss, 
                           "Validation Cat Loss": val_cat_loss, "Validation NER Loss": val_ner_loss})	
            # Saving best models
            if val_loss < best_loss:   
                best_loss = val_loss
                if not hypertuning:
                    if print_flag:           
                        print(f"Saving model @{epoch}")
                    model.save_model(save_path)
                    best_model = copy.deepcopy(model)
    
    except KeyboardInterrupt:
        print(f"Training interrupted @{epoch}. Returning the best model so far.")
    
    if use_wandb and not hypertuning: wandb.finish()
    return best_model, best_loss

# # Calculate the metrics #TODO look at dimensions of scores and labels
# if print_flag:
#     print("Information for calculating metrics")
#     print(f"The shape of the scores is {total_scores.shape}")
#     print(f"The shape of the labels is {total_labels.shape}")
#     print("The input to the metric evaluator should be lists of lists. Converting the tensors to lists.")
#     print("Outside list has the length of the number of data points. Inside list should have the length of the number of inview news articles and should differ.")
# metrics = MetricEvaluator(
#     labels=total_labels.to_list(),
#     predictions=total_scores.to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# metrics.evaluate()

# # Add the ebrec/evaluation directory to sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'ebrec', 'evaluation'))
# sys.path.insert(0, parent_dir)
# from metrics._ranking import mrr_score
# from metrics._ranking import ndcg_score
# from metrics._classification import auc_score_custom
# def get_metrics(y_true, y_pred, metrics):
#     """
#     Function to calculate the metrics for the given true and predicted labels.
    
#     Args:
#         y_true (torch.Tensor): The true labels.
#         y_pred (torch.Tensor): The predicted labels.
        
#     Returns:
#         dict: A dictionary containing the calculated metrics.
#     """
#     result = {
#         'mrr_score': mrr_score(y_true, y_pred),
#         'ndcg_score': ndcg_score(y_true, y_pred),
#         'auc_score_custom': auc_score_custom(y_true, y_pred)
#     }
#     for key in metrics:
#             metrics[key] += result[key]

#     return metrics

# def test(news_encoder, user_encoder, dataloader_test,
#           scoring_function:callable = cosine_sim, device:str = "cpu"):
#     """
#     Function to test the model on the given dataset.
    
#     Args:
#         news_encoder (torch.nn.Module): The news encoder module.
#         user_encoder (torch.nn.Module): The user encoder module.
#         dataloader_test (torch.utils.data.DataLoader): The dataloader for the test dataset.
#         device (torch.device): The device to be uszed for testing.
#     """
#     user_encoder.eval()
#     news_encoder.eval()
#     metrics = {
#         'mrr_score': 0,
#         'ndcg_score': 0,
#         'auc_score_custom': 0
#     }

#     for data in dataloader_test:
#         # Get the data
#         (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = get2device(data, device)
#         inview_news_embeddings, inview_news_cat, inview_news_ner = news_encoder(news_tokens, news_mask)  
#         history_news_embeddings, history_news_cat, history_news_ner = news_encoder(user_histories, user_mask) 
#         user_embeddings = user_encoder(history_news_embeddings)                    
#         # MAIN task: Click prediction
#         scores = scoring_function(user_embeddings, inview_news_embeddings)
#         # Calculate the metrics
#         metrics = get_metrics(labels, scores, metrics)
#     for key in metrics:
#         metrics[key] /= len(dataloader_test)
#     for key, value in metrics.items():
#         print(f"{key:<5}: {value:.3f}")
        
#     return metrics