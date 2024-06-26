import argparse
from trainer import train
import optuna
from functools import partial
from utils import load_configuration, get_dataloaders
import copy
from mtrec_class import Mtrec
import torch
import yaml
from utils import timer
import os

@timer
def test_config(trial, cfg_base, device):
    """
    Perform hyperparameter tuning for the configuration.

    Args:
        trial (optuna.Trial): The optuna trial object for hyperparameter optimization.
        cfg_base (dict): The base configuration.
        device (str): The device to run the training on.

    Returns:
        float: The best validation loss from this hyperparameter instance.
    """
    cfg = copy.deepcopy(cfg_base)
    # Get the hyperparameter options
    hcf = cfg["hypertuning"]
    hidden_sizes = hcf["hidden_size"]
    nl_min, nl_max = hcf["num_layers"]["min"],hcf["num_layers"]["max"]
    lr_min, lr_max = hcf["lr"]["min"], hcf["lr"]["max"]
    batch_sizes = hcf["batch_size"]
    optimizers = hcf["optimizer"]
    # Categorical net
    cfg["news_encoder"]["cfg_cat"]["hidden_size"] = trial.suggest_categorical("cat_hidden_size", hidden_sizes)
    cfg["news_encoder"]["cfg_cat"]["num_layers"] = trial.suggest_int("cat_num_layers", nl_min, nl_max)
    # NER net
    cfg["news_encoder"]["cfg_ner"]["hidden_size"] = trial.suggest_categorical("ner_hidden_size", hidden_sizes)
    cfg["news_encoder"]["cfg_ner"]["num_layers"] = trial.suggest_int("ner_num_layers", nl_min, nl_max)
    # User encoder
    cfg["user_encoder"]["hidden_size"] = trial.suggest_categorical("user_hidden_size", hidden_sizes)
    # Training
    cfg["trainer"]["batch_size"] = trial.suggest_categorical("batch_size", batch_sizes)
    
    cfg["trainer"]["lr_user"] = trial.suggest_float("lr_user", lr_min, lr_max, log=True)
    cfg["trainer"]["lr_news"] = trial.suggest_float("lr_news", lr_min, lr_max, log=True)
    cfg["trainer"]["lr_bert"] = trial.suggest_float("lr_bert", lr_min, lr_max, log=True)
    
    cfg["trainer"]["optimizer"] = trial.suggest_categorical("optimizer", optimizers)
    # Initialize model
    Mtrec_model = Mtrec(cfg, device=device)
    print(f"Configuration: {cfg}")
    #Train the model
    try:    
        (dataloader_train, dataloader_val) = get_dataloaders(cfg)
        best_validation_loss =       train(model      = Mtrec_model, 
                                            dataloader_train = dataloader_train, 
                                            dataloader_val   = dataloader_val, 
                                            cfg              = cfg["trainer"],  
                                            use_wandb        = cfg["wandb"],
                                            hypertuning      = True, 
                                            name_run         = cfg.get("name_run", "unnamed"))
    
    except KeyboardInterrupt:
        best_validation_loss = 1000
    finally: 
        #Report best result
        return best_validation_loss


def merge(best_params, cfg):
    """
    Merge the best_params dictionary into the cfg dictionary to prepare for export.

    Parameters:
    - best_params (dict): A dictionary containing the best hyperparameters.
    - cfg (dict): A dictionary containing the configuration.

    Returns:
    - cfg (dict): The updated configuration with the best hyperparameters.
    """

    # Categorical net
    cfg["news_encoder"]["cfg_cat"]["hidden_size"] = best_params["cat_hidden_size"]
    cfg["news_encoder"]["cfg_cat"]["num_layers"] = best_params["cat_num_layers"]
    
    # NER net
    cfg["news_encoder"]["cfg_ner"]["hidden_size"] = best_params["ner_hidden_size"]
    cfg["news_encoder"]["cfg_ner"]["num_layers"] = best_params["ner_num_layers"]
    
    # User encoder
    cfg["user_encoder"]["hidden_size"] = best_params["user_hidden_size"]
    
    # Training
    cfg["trainer"]["batch_size"] = best_params["batch_size"]
    
    cfg["trainer"]["lr_user"] = best_params["lr_user"]
    cfg["trainer"]["lr_news"] = best_params["lr_news"]
    cfg["trainer"]["lr_bert"] = best_params["lr_bert"]
    
    cfg["trainer"]["optimizer"] = best_params["optimizer"]
    return cfg

def main():
    # Parse the arguments and load config
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='hypertune', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
    
    # Initialise wandb if needed
    if cfg["wandb"]: 
        import wandb
        wandb.init(project="mtrec", name="hypertuning", config=cfg )
    
    # Get the device
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define the config for non-tuned parameters
    target_func = partial(test_config, cfg_base = cfg, device=device)
    
    # Perform the hyperparameter tuning
    study = optuna.create_study(direction = "minimize")
    study.optimize(target_func, n_trials=cfg["hypertuning"]["n_trials"], show_progress_bar=True)
    
    # Finish the wandb run
    print("best parameters:\n", study.best_params)
    wandb.log({"best params": study.best_params})
    wandb.finish()

    # Export the best configuration
    best_cfg = merge(study.best_params, cfg)
    os.makedirs("hypertuning", exist_ok=True)
    with open('hypertuning/best_config.yml', 'w') as file:
        yaml.dump(best_cfg, file, default_flow_style=False)

if __name__ == "__main__":
    main()

