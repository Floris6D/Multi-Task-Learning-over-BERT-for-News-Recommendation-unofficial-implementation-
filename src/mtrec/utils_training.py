import torch
from torch import nn
from utils import timer

def cross_product(user_embedding, news_embedding):
    """
    Function to calculate the cross product of the user and news embeddings.
    
    Args:
        user_embedding (torch.Tensor): Batch_size * embedding_dimension tensor of user embeddings.
        news_embedding (torch.Tensor): Batch_size * N * embedding_dimension tensor of news embeddings.
        
    Returns:
        torch.Tensor: Batch_size * N tensor of scores.
    """
    device = user_embedding.device
    bsu, emb_dimu = user_embedding.shape
    bsn, N, emb_dimn = news_embedding.shape
    assert bsu == bsn , "Batch sizes of user and news embeddings do not match"
    assert emb_dimu == emb_dimn, "Embedding dimensions of user and news embeddings do not match"
    scores = torch.einsum("bk,bik->bi",user_embedding, news_embedding).to(device)
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
    device = user_embedding.device
    scores = torch.cosine_similarity(user_embedding.unsqueeze(1), news_embedding, axis = 2).to(device)
    return scores


def get2device(data, device):
    '''
    Move the input data tensors to the specified device.

    Args:
        data (tuple): A tuple containing input data tensors.
        device (torch.device): The target device to move the tensors to.

    Returns:
        tuple: A tuple containing the input data tensors moved to the specified device.
    '''
    (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview), impression_id = data
    return (user_histories.to(device), user_mask.to(device), news_tokens.to(device), news_mask.to(device)), (labels.to(device), c_labels_his.to(device), c_labels_inview.to(device), ner_labels_his.to(device), ner_labels_inview.to(device)), impression_id.to(device)


def main_loss(scores, labels, normalization=True):
    """
    Calculate the main loss for a given set of scores and labels.

    Args:
        scores (torch.Tensor): (batch_size * num_inview) The predicted scores.
        labels (torch.Tensor): (batch_size * num_inview) The ground truth labels.
        normalization (bool, optional): Whether to normalize the scores. Defaults to True.

    Returns:
        torch.Tensor: The calculated main loss.
    """
    if normalization: 
        scores = scores - torch.max(scores, dim=1, keepdim=True)[0]  #TODO why  [0]? 
        scores = torch.exp(scores)  # apply exponential function
        sum_exp = torch.sum(scores, dim=1, keepdim=True)  # calculate the sum of exponential scores
        scores = scores / sum_exp  # normalize the scores to sum to 1
    sum_exp = torch.sum(torch.exp(scores), dim=1)
    pos_scores = torch.sum(scores * labels, axis=1)
    output = -torch.log(torch.exp(pos_scores)/sum_exp).mean() 
    return output


def category_loss(p1, p2, l1, l2):
    """
    First we untangle all the category predictions and labels
    Then apply cross entropy loss
    p1 is prediction 1 related to the inview articles
    p2 is prediction 2 related to the history articles
    l1 is label 1 related to the inview articles
    l2 is label 2 related to the history articles (this contains nans for padding)
    """
    device =  p1.device
    bs, N1, num_cat = p1.shape
    bs, N2, num_cat = p2.shape
    p1 = p1.reshape(bs*N1, num_cat)
    p2 = p2.reshape(bs*N2, num_cat)
    l1 = l1.reshape(bs*N1)
    l2 = l2.reshape(bs*N2)
    predictions = torch.cat([p1, p2], dim = 0)
    labels = torch.cat([l1, l2], dim = 0)
    
    # Filter out the rows which only contain nans in the predictions
    mask = ~torch.isnan(predictions).any(dim=1).to(device)
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Return cross entropy loss
    return nn.CrossEntropyLoss()(predictions, labels)



def NER_loss(p1, p2, l1, l2, mask1, mask2): 
    """
    First we untangle all the NER predictions and labels
    Then apply cross entropy loss
    """
    device = p1.device
    # Get shapes
    bs, N1, tl1, num_ner = p1.shape
    bs, N2, tl2, num_ner = p2.shape
    # Reshape predictions
    p1 = p1.reshape(bs*N1*tl1, num_ner)
    p2 = p2.reshape(bs*N2*tl2, num_ner)
    predictions = torch.cat([p1, p2], dim = 0)
    # Reshape mask
    mask1   = mask1.reshape(bs*N1*tl1)
    mask2   = mask2.reshape(bs*N2*tl2)
    mask = torch.cat([mask1, mask2], dim = 0)
    # Reshape labels (insert a -1 for the cls token and remove last row of demension 2)
    l1 = torch.cat([torch.zeros(bs, N1, 1).long().to(device), l1], dim = 2)
    l1 = l1[:,:,:tl1].reshape(bs*N1*tl1)
    l2 = torch.cat([torch.zeros(bs, N2, 1).long().to(device), l2], dim = 2)
    l2 = l2[:,:,:tl2].reshape(bs*N2*tl2)
    labels = torch.cat([l1, l2], dim = 0).long()
    # Apply mask
    labels = labels[mask.bool()]
    predictions = predictions[mask.bool()]
    # Laslty also remove the labels which are -1 #TODO: JE: This is caused by different tokenization by us and BERT
    mask = labels != -1
    labels = labels[mask]
    predictions = predictions[mask]
    # Calculate loss
    return nn.CrossEntropyLoss()(predictions, labels)