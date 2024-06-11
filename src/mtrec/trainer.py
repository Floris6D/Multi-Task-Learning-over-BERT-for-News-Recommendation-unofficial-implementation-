import torch
import torch.nn as nn
import os
import sys
import copy

# Add the ebrec/evaluation directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'ebrec', 'evaluation'))
sys.path.insert(0, parent_dir)

# Import the required functions from the metrics package
from metrics._ranking import mrr_score
from metrics._ranking import ndcg_score
from metrics._classification import auc_score_custom

def cosine_sim(user_embedding, news_embedding):
    """
    Function to calculate the cross product of the user and news embeddings.
    
    Args:
        user_embedding (torch.Tensor): Batch_size * embedding_dimension tensor of user embeddings.
        news_embedding (torch.Tensor): Batch_size * N * embedding_dimension tensor of news embeddings.
        
    Returns:
        torch.Tensor: Batch_size * N tensor of scores.
    """
    # scores = user_embedding.unsqueeze(1)* news_embedding
    scores = torch.cosine_similarity(user_embedding.unsqueeze(1), news_embedding, axis = 2)
    return scores

def train(news_encoder, user_encoder, dataloader_train, dataloader_val, cfg, scoring_function:callable = cosine_sim,
          criterion: nn.Module = nn.CrossEntropyLoss(),  device:str = "cpu", save_dir:str = "saved_models"):
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
    news_encoder.train()
    user_encoder.train()
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam([
            {'params': news_encoder.parameters(), 'lr': cfg["lr_news"]},
            {'params': user_encoder.parameters(), 'lr': cfg["lr_user"]}
        ])
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([
            {'params': news_encoder.parameters(), 'lr': cfg["lr_news"]},
            {'params': user_encoder.parameters(), 'lr': cfg["lr_user"]}
            ])
    else:
        print("Invalid optimizer <{}>.".format(optimizer))
        return
    
    total_loss = 0
    best_loss = float('inf')
    best_user_encoder, best_news_encoder = None, None
    save_num = 0
    while os.path.exists(save_dir+f'/run{save_num}'):
        save_num += 1
    save_path = save_dir+f'/run{save_num}'
    os.makedirs(save_path)
    print(f"Saving models to {save_path}")
    try: #training can be interrupted by catching KeyboardInterrupt
        #training
        for epoch in range(cfg['epochs']):
            for data in dataloader_train:
                # Get the data
                (user_histories, user_mask, news_tokens, news_mask) , labels = data
                user_histories = user_histories.to(device)
                user_mask = user_mask.to(device)
                news_tokens = news_tokens.to(device)
                news_mask = news_mask.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # Get the embeddings
                user_embeddings = user_encoder(user_histories, user_mask)
                news_embeddings = news_encoder(news_tokens, news_mask)
                scores = scoring_function(user_embeddings, news_embeddings)
                loss = criterion(scores, labels) ##TODO make criterion correct for ranking
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("Epoch {} Loss: {}".format(epoch, total_loss))
            user_encoder.eval()
            news_encoder.eval()
            total_loss_val = 0
            #validation
            for data in dataloader_val:
                (user_histories, user_mask, news_tokens, news_mask) , labels = data
                user_histories = user_histories.to(device)
                user_mask = user_mask.to(device)
                news_tokens = news_tokens.to(device)
                news_mask = news_mask.to(device)
                labels = labels.to(device)
                user_embeddings = user_encoder(user_histories, user_mask)
                news_embeddings = news_encoder(news_tokens, news_mask)
                scores = scoring_function(user_embeddings, news_embeddings)
                loss = criterion(scores, labels)
                total_loss_val += loss.item()
            #saving best models
            if total_loss_val < best_loss:                 
                print("Saving model @{epoch}")
                best_loss = total_loss_val
                torch.save(user_encoder.state_dict(), save_path + '/user_encoder.pth')
                torch.save(news_encoder.state_dict(), save_path + '/news_encoder.pth')
                best_user_encoder = copy.deepcopy(user_encoder)
                best_news_encoder = copy.deepcopy(news_encoder)
            print("Validation Loss: {}".format(total_loss_val))
    except KeyboardInterrupt:
        print(f"Training interrupted @{epoch}. Returning the best models so far.")
    
    return best_user_encoder, best_news_encoder

    
def get_metrics(y_true, y_pred):
    """
    Function to calculate the metrics for the given true and predicted labels.
    
    Args:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels.
        
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metrics = {
        'mrr_score': mrr_score(y_true, y_pred),
        'ndcg_score': ndcg_score(y_true, y_pred),
        'auc_score_custom': auc_score_custom(y_true, y_pred)
    }
    return metrics

def test(news_encoder, user_encoder, dataloader_test,
          scoring_function:callable = cosine_sim, device:str = "cpu"):
    """
    Function to test the model on the given dataset.
    
    Args:
        news_encoder (torch.nn.Module): The news encoder module.
        user_encoder (torch.nn.Module): The user encoder module.
        dataloader_test (torch.utils.data.DataLoader): The dataloader for the test dataset.
        device (torch.device): The device to be used for testing.
    """
    user_encoder.eval()
    news_encoder.eval()
    metrics_total = {
        'mrr_score': 0,
        'ndcg_score': 0,
        'auc_score_custom': 0
    }
    i = 0
    for data in dataloader_test:
        # Get the data
        (user_histories, user_mask, news_tokens, news_mask) , labels = data
        user_histories = user_histories.to(device)
        user_mask = user_mask.to(device)
        news_tokens = news_tokens.to(device)
        news_mask = news_mask.to(device)
        labels = labels.to(device)
        # Get the embeddings
        user_embeddings = user_encoder(user_histories)
        news_embeddings = news_encoder(news_tokens, news_mask)
        # Get the scores
        scores = scoring_function(user_embeddings, news_embeddings)
        # Calculate the metrics
        metrics = get_metrics(labels, scores)
        for key in metrics_total:
            metrics_total[key] += metrics[key]
    
    for key in metrics_total:
        metrics_total[key] /= i
    
    for key, value in metrics.items():
        print(f"{key:<5}: {value:.3f}")
        
    return metrics