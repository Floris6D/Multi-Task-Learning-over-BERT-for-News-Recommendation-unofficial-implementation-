import torch
import torch.nn as nn
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from gradient_surgery import PCGrad

# Import the required functions from the metrics package
#from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore


import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper


class TestNet(nn.Module): #TODO: remove this
    def __init__(self, input_dim=2*768, output_dim=1, hidden_dim=128):
        super(TestNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, user_embedding, candidate_embeddings):
        # user_embedding: batch_size * embedding_dim
        # candidate_embeddings: batch_size * N * embedding_dim
        bs, N, emb_dim = candidate_embeddings.shape
        candidate_embeddings = candidate_embeddings.reshape(bs*N, emb_dim)
        x = torch.cat([user_embedding, candidate_embeddings], dim=1)
        x = self.fc(x)
        x = x.reshape(bs, N)
        return x

def cross_product(user_embedding, news_embedding):
    """
    Function to calculate the cross product of the user and news embeddings.
    
    Args:
        user_embedding (torch.Tensor): Batch_size * embedding_dimension tensor of user embeddings.
        news_embedding (torch.Tensor): Batch_size * N * embedding_dimension tensor of news embeddings.
        
    Returns:
        torch.Tensor: Batch_size * N tensor of scores.
    """
    bsu, emb_dimu = user_embedding.shape
    bsn, N, emb_dimn = news_embedding.shape
    assert bsu == bsn , "Batch sizes of user and news embeddings do not match"
    assert emb_dimu == emb_dimn, "Embedding dimensions of user and news embeddings do not match"
    assert user_embedding.requires_grad, "User embedding requires grad"
    assert news_embedding.requires_grad, "News embedding requires grad"
    scores = torch.einsum("bk,bik->bi",user_embedding, news_embedding)
    return scores


def print_optimizer_parameters(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Parameter group {i}:")
        for param in param_group['params']:
            print(f"Parameter: {param.shape}")
            print(f"Requires Grad: {param.requires_grad}")

def cosine_sim(user_embedding, news_embedding):
    """
    Function to calculate the cross product of the user and news embeddings.
    
    Args:
        user_embedding (torch.Tensor): Batch_size * embedding_dimension tensor of user embeddings.
        news_embedding (torch.Tensor): Batch_size * N * embedding_dimension tensor of news embeddings.
        
    Returns:
        torch.Tensor: Batch_size * N tensor of scores.
    """
    scores = torch.cosine_similarity(user_embedding.unsqueeze(1), news_embedding, axis = 2)
    return scores

def get2device(data, device):
    (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview), impression_id = data
    return (user_histories.to(device), user_mask.to(device), news_tokens.to(device), news_mask.to(device)), (labels.to(device), c_labels_his.to(device), c_labels_inview.to(device), ner_labels_his.to(device), ner_labels_inview.to(device)), impression_id.to(device)

def main_loss(scores, labels, normalization = True):
    assert scores.requires_grad, "Scores should require grad"
    if normalization: # normalization? TODO
        scores = scores - torch.max(scores, dim=1, keepdim=True)[0]  # subtract the maximum value for numerical stability
        scores = torch.exp(scores)  # apply exponential function
        sum_exp = torch.sum(scores, dim=1, keepdim=True)  # calculate the sum of exponential scores
        scores = scores / sum_exp  # normalize the scores to sum to 1
    sum_exp = torch.sum(torch.exp(scores), dim = 1)
    pos_scores = torch.sum(scores * labels, axis = 1)
    return -torch.log(torch.exp(pos_scores)/sum_exp).mean() 

def category_loss(p1, p2, l1, l2):
    """
    First we untangle all the category predictions and labels
    Then apply cross entropy loss
    """
    bs, N1, num_cat = p1.shape
    bs, N2, num_cat = p2.shape
    p1 = p1.reshape(bs*N1, num_cat)
    p2 = p2.reshape(bs*N2, num_cat)
    l1 = torch.argmax(l1, dim=2) # go from one-hot to index
    l2 = torch.argmax(l2, dim=2) # go from one-hot to index
    l1 = l1.reshape(bs*N1)
    l2 = l2.reshape(bs*N2)
    predictions = torch.cat([p1, p2], dim = 0)
    labels = torch.cat([l1, l2], dim = 0)
    return nn.CrossEntropyLoss()(predictions, labels)

def NER_loss(p1, p2, l1, l2, mask1, mask2): 
    """
    First we untangle all the NER predictions and labels
    Then apply cross entropy loss
    """
    # Get shapes
    bs, N1, tl1, num_ner = p1.shape
    bs, N2, tl2, num_ner = p2.shape
    # Reshape predictions
    p1 = p1.reshape(bs*N1*tl1, num_ner)
    p2 = p2.reshape(bs*N2*tl2, num_ner)
    predictions = torch.cat([p1, p2], dim = 0)
    # Reshape mask
    mask1   = mask1[:,:,:tl1].reshape(bs*N1*tl1)
    mask2   = mask2[:,:,:tl2].reshape(bs*N2*tl2)
    mask = torch.cat([mask1, mask2], dim = 0)
    # Reshape labels
    l1      = l1[:,:,:tl1].reshape(bs*N1*tl1)
    l2      = l2[:,:,:tl2].reshape(bs*N2*tl2)
    labels = torch.cat([l1, l2], dim = 0).long()
    # Apply mask
    labels = torch.masked_select(labels, mask.bool())
    predictions = predictions[mask.bool()]
    # Calculate loss
    return nn.CrossEntropyLoss()(predictions, labels)

def plot_loss(loss_train, loss_val, title:str = "Loss", save_dir:str = "default_savedir", xlabel:str = "Epoch", ylabel:str = "Loss"):
    X = np.arange(1, len(loss_train)+1)
    plt.plot(X, loss_train, label = "Training Loss")
    X = np.arange(1, len(loss_val)+1)
    plt.plot(X, loss_val, label = "Validation Loss")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{title}.png") 
    
def train(model, dataloader_train, dataloader_val, cfg, 
          print_flag = True, save_dir:str = "saved_models"):
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

    #optimizer = PCGrad(optimizer) #TODO: PCGrad
    
    #initialize to track best
    best_loss = float('inf')
    save_num = 0
    # Ensure the save directory ends with a slash
    if not save_dir.endswith('/'):
        save_dir += '/'

    # Determine a unique save directory
    while os.path.exists(os.path.join(save_dir, f'run{save_num}')):
        save_num += 1

    save_path = os.path.join(save_dir, f'run{save_num}')
    os.makedirs(save_path, exist_ok=True)

    # Debugging: Confirm directory creation
    if not os.path.exists(save_path):
        raise RuntimeError(f"Failed to create directory: {save_path}")
    
    training_losses, validation_losses = [], []
    if print_flag: print(f"Saving models to {save_path}")
    try: #training can be interrupted by catching KeyboardInterrupt
        #training
        for epoch in range(cfg['epochs']):
            if print_flag: print(f"Epoch {epoch} / {cfg['epochs']}")
            
            # Training
            total_loss, total_main_loss = model.train(dataloader_train, optimizer, print_flag)
            training_losses.append(total_main_loss)

            #validation
            total_loss_val, total_main_loss_val = model.validate(dataloader_val, print_flag)
            validation_losses.append(total_main_loss_val)
            #saving best models
            if total_loss_val < best_loss:   
                best_loss = total_loss_val
                if print_flag:
                    print(f"total loss val: {total_loss_val}")
                    print(f"best loss: {best_loss}")              
                    print(f"Saving model @{epoch}")
                model.save_model(save_path)
                best_model = copy.deepcopy(model)
    except KeyboardInterrupt:
        print(f"Training interrupted @{epoch}. Returning the best model so far.")
    plot_loss(training_losses, validation_losses, save_dir = save_path)
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