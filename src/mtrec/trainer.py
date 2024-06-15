import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import os
import sys
import copy

from gradient_surgery import PCGrad

# Add the ebrec/evaluation directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'ebrec', 'evaluation'))
sys.path.insert(0, parent_dir)

# Import the required functions from the metrics package
from metrics._ranking import mrr_score
from metrics._ranking import ndcg_score
from metrics._classification import auc_score_custom

def cross_product(user_embedding, news_embedding):
    """
    Function to calculate the cross product of the user and news embeddings.
    
    Args:
        user_embedding (torch.Tensor): Batch_size * embedding_dimension tensor of user embeddings.
        news_embedding (torch.Tensor): Batch_size * N * embedding_dimension tensor of news embeddings.
        
    Returns:
        torch.Tensor: Batch_size * N tensor of scores.
    """
    scores = user_embedding.unsqueeze(1)* news_embedding
    return scores

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
    (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = data
    if isinstance(ner_labels_his, list):
        return (user_histories.to(device), user_mask.to(device), news_tokens.to(device), news_mask.to(device)), (labels.to(device), c_labels_his.to(device), c_labels_inview.to(device), ner_labels_his, ner_labels_inview)
    return (user_histories.to(device), user_mask.to(device), news_tokens.to(device), news_mask.to(device)), (labels.to(device), c_labels_his.to(device), c_labels_inview.to(device), ner_labels_his.to(device), ner_labels_inview.to(device))

def main_loss(scores, labels):
    # normalization? TODO
    # scores = scores - torch.max(scores, dim=1, keepdim=True)[0]  # subtract the maximum value for numerical stability
    # scores = torch.exp(scores)  # apply exponential function
    # sum_exp = torch.sum(scores, dim=1, keepdim=True)  # calculate the sum of exponential scores
    # scores = scores / sum_exp  # normalize the scores to sum to 1
    sum_exp = torch.sum(torch.exp(scores), dim = 1)
    #pos_scores =  torch.einsum("bs,bs->b", scores, labels*1.0)
    pos_scores = torch.sum(scores * labels, axis = 1)
    return -torch.log(pos_scores/sum_exp).mean() #no need for sum since only one positive label

# def category_loss(predicted_probs, labels):
#     r, c = predicted_probs.shape
#     return -torch.mean(torch.sum(labels.reshape(r, c) * torch.log(predicted_probs), dim=1))

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
    bs, N1, tl1, num_ner = p1.shape
    bs, N2, tl2, num_ner = p2.shape
       
    p1 = p1.reshape(bs*N1*tl1, num_ner)
    p2 = p2.reshape(bs*N2*tl2, num_ner)
    l1 = l1.reshape(bs*N1*tl1)
    l2 = l2.reshape(bs*N2*tl2)
       
    mask1 = mask1[:,:,:tl1].reshape(bs*N1*tl1)
    mask2 = mask2[:,:,:tl2].reshape(bs*N2*tl2)
    mask = torch.cat([mask1, mask2], dim = 0)
    predictions = torch.cat([p1, p2], dim = 0)
    predictions = predictions[mask.bool()]
    labels = torch.cat([l1, l2], dim = 0).long()
    labels = torch.masked_select(labels, mask.bool())
    return nn.CrossEntropyLoss()(predictions, labels) 
    
def train(user_encoder, news_encoder, dataloader_train, dataloader_val, cfg, scoring_function:callable = cosine_sim,
          criterion:callable = main_loss,  device:str = "cpu", save_dir:str = "saved_models"):
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
    params = [
        {"params": [user_encoder.W,  user_encoder.q],   "lr": cfg["lr_user"]},  # lr of attention layer in user encoder
        {"params": list(news_encoder.cat_net.parameters()) + list(news_encoder.ner_net.parameters()),
                   "lr": cfg["lr_news"]},  # lr of auxiliary tasks
        {"params": news_encoder.bert.parameters(), "lr": cfg["lr_bert"]}  # Parameters of BERT
    ] 

    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params)
    else:
        print("Invalid optimizer <{}>.".format(optimizer))
        return
    
    optimizer = PCGrad(optimizer)
    
    #initialize to track best
    total_loss = 0
    best_loss = 0
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
            print(f"Epoch {epoch} / {cfg['epochs']}")
            news_encoder.train()
            user_encoder.train()
            for data in dataloader_train:
                # Get the data
                (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = get2device(data, device)
                
                optimizer.zero_grad()
                # Get the embeddings
                inview_news_embeddings, inview_news_cat, inview_news_ner = news_encoder(news_tokens, news_mask)  
                history_news_embeddings, history_news_cat, history_news_ner = news_encoder(user_histories, user_mask) 
                user_embeddings = user_encoder(history_news_embeddings)
                # AUX task: Category prediction            
                cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
                print(f"cat loss: {cat_loss}")
                # AUX task: NER 
                # bs, N1, tl1, num_ner= inview_news_ner.shape
                # ner_labels_inview  = 2*torch.ones((bs, N1, tl1)).to(device)
                # bs, N2, tl2, num_ner= history_news_ner.shape
                # ner_labels_his  = 2*torch.ones((bs, N2, tl2)).to(device)
                ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his, news_mask, user_mask)
                print(f"ner loss: {ner_loss}")
                scores = scoring_function(user_embeddings, inview_news_embeddings)
                main_loss = criterion(scores, labels)           
                optimizer.pc_backward([main_loss, cat_loss, ner_loss])
                print(f"main loss: {main_loss}")
                optimizer.step()
                total_loss += main_loss.item() + cat_loss.item() + ner_loss.item()
                break #TODO remove
            print("training loss: ", total_loss)
            user_encoder.eval()
            news_encoder.eval()
            total_loss_val = 0
            #validation
            for data in dataloader_val:
                # Get the data
                (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = get2device(data, device)
                optimizer.zero_grad()
                # Get the embeddings
                inview_news_embeddings, inview_news_cat, inview_news_ner = news_encoder(news_tokens, news_mask)  
                history_news_embeddings, history_news_cat, history_news_ner = news_encoder(user_histories, user_mask) 
                user_embeddings = user_encoder(history_news_embeddings)                    
                # # AUX task: Category prediction            
                # cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
                # print(f"cat loss: {cat_loss}")
                # # AUX task: NER 
                # ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his)
                # print(f"ner loss: {ner_loss}")
                # MAIN task: Click prediction
                scores = scoring_function(user_embeddings, inview_news_embeddings)
                main_loss = criterion(scores, labels)
                print(f"main loss: {main_loss}")
                total_loss_val += main_loss.item() + cat_loss.item() + ner_loss.item()
                break #TODO: remove
            #saving best models
            if total_loss_val < best_loss:   
                print(f"total loss val: {total_loss_val}")
                print(f"best loss: {best_loss}")              
                print("Saving model @{epoch}")
                best_loss = total_loss_val
                
                print(f"total loss val: {total_loss_val}")
                print(f"best loss: {best_loss}")
                torch.save(user_encoder.state_dict(), save_path + '/user_encoder.pth')
                torch.save(news_encoder.state_dict(), save_path + '/news_encoder.pth')
                best_user_encoder = copy.deepcopy(user_encoder)
                best_news_encoder = copy.deepcopy(news_encoder)
            print(f"Validation Loss: {total_loss_val}")
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
        device (torch.device): The device to be uszed for testing.
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