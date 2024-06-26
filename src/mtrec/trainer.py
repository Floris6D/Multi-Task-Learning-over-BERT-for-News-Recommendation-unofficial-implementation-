import torch
import os
import copy
from gradient_surgery import PCGrad
from utils_training import *
import time
import wandb

    
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
    if use_wandb: 
        start_time = time.time()
        import wandb
        end_time = time.time()
        import_time = end_time - start_time
        print(f"Importing wandb took {import_time} seconds.")
    # Initialize model
    user_encoder = model.user_encoder
    news_encoder = model.news_encoder
    params = [
        {"params": [user_encoder.W,  user_encoder.q],   "lr": cfg["lr_user"]},  # lr of attention layer in user encoder
        {"params": list(news_encoder.cat_net.parameters()) + list(news_encoder.ner_net.parameters()),
                   "lr": cfg["lr_news"]},  # lr of auxiliary tasks
        {"params": news_encoder.bert.parameters(), "lr": cfg["lr_bert"]}  # Parameters of BERT
    ] 

    best_model = None
    # Initialize optimizer
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params)
    elif cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params)
    else:
        print("Invalid optimizer <{}>.".format(cfg["optimizer"]))
        return
    

    # Make sure we tune the entire BERT model
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.requires_grad=True

    # Add gradient surgery
    if not cfg["skip_gs"]: 
        optimizer = PCGrad(optimizer, aux_scaler = cfg["aux_scaler"])

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
    if print_flag: print(f"Saving models to {save_path}")
    print(f"time before training: {time.time()-start_time}")
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
            # Saving best model
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