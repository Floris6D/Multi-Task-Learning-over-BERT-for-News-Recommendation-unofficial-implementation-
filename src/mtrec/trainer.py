import torch
import torch.nn as nn
import os
import copy

def cross_product(user_embedding, news_embedding):
    """
    Function to calculate the cross product of the user and news embeddings.
    
    Args:
        user_embedding (torch.Tensor): Batch_size * embedding_dimension tensor of user embeddings.
        news_embedding (torch.Tensor): Batch_size * N * embedding_dimension tensor of news embeddings.
        
    Returns:
        torch.Tensor: Batch_size * N tensor of scores.
    """
    # scores = user_embedding.unsqueeze(1)* news_embedding
    scores = torch.sum(user_embedding.unsqueeze(1)* news_embedding, axis = 2)
    return scores


def test_cross_product():
    # Define input tensors
    user_embedding = torch.tensor([[1, 2, 3], [4, 5, 6]])
    news_embedding = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    
    # Calculate expected output
    expected_scores = torch.tensor([[14, 32], [122, 167]])
    
    # Calculate actual output
    actual_scores = cross_product(user_embedding, news_embedding)
    # Check if the actual output matches the expected output
    assert torch.all(actual_scores == expected_scores)


def train(news_encoder, user_encoder, dataloader_train, dataloader_val, config, scoring_function:callable = cross_product,
          criterion: nn.Module = nn.CrossEntropyLoss(),  device:str = "cpu"):
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
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam([
            {'params': news_encoder.parameters(), 'lr': 0.001},
            {'params': user_encoder.parameters(), 'lr': 0.0001}
        ])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([{'params': news_encoder.parameters(), 'lr': config["lr_news"]},
            {'params': user_encoder.parameters(), 'lr': config["lr_user"]}])
    else:
        print("Invalid optimizer <{}>.".format(optimizer))
        return
    
    total_loss = 0
    best_loss = float('inf')
    best_user_encoder, best_news_encoder = None, None
    create_save_dir = True
    try:
        for epoch in range(config['epochs']):
            for data in dataloader_train:
                user_histories, news, labels = data #TODO make compatible
                user_histories = user_histories.to(device)
                news = news.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                user_embeddings = user_encoder(user_histories)
                news_embeddings = news_encoder(news)
                scores = scoring_function(user_embeddings, news_embeddings)
                loss = criterion(scores, labels) ##TODO make criterion correct for ranking
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("Epoch {} Loss: {}".format(epoch, total_loss))
            user_encoder.eval()
            news_encoder.eval()
            total_loss_val = 0
            for data in dataloader_val:
                user_histories, news, labels = data #TODO make compatible
                user_histories = user_histories.to(device)
                news = news.to(device)
                labels = labels.to(device)
                user_embeddings = user_encoder(user_histories)
                news_embeddings = news_encoder(news)
                scores = scoring_function(user_embeddings, news_embeddings)
                loss = criterion(scores, labels)
                total_loss_val += loss.item()
            if total_loss_val < best_loss:
                if create_save_dir: #only run once
                    if not os.path.exists('saved_models'):
                        os.makedirs('saved_models')
                    save_num = 0
                    while os.path.exists(f'saved_models/run{save_num}'):
                        save_num += 1
                    save_path = f'saved_models/run{save_num}'
                    os.makedirs(save_path)
                    create_save_dir = False
                    print(f"Saving models to {save_path}")

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

    

def test(news_encoder, user_encoder, dataloader_test,
          scoring_function:callable = cross_product, device:str = "cpu"):
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
    metrics = {}
    for data in dataloader_test:
        user_histories, news, labels = data #TODO make compatible
        user_histories = user_histories.to(device)
        news = news.to(device)
        labels = labels.to(device)
        user_embeddings = user_encoder(user_histories)
        news_embeddings = news_encoder(news)
        scores = scoring_function(user_embeddings, news_embeddings)
        #TODO: calc metrics
    
    for key, value in metrics.items():
        print(f"{key:<5}: {value:.3f}")
        
    return metrics