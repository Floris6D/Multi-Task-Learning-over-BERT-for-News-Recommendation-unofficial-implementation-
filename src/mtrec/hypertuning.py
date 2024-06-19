import argparse
from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from trainer import train
import optuna
from functools import partial
from utils import load_configuration, get_dataloaders
import copy
from mtrec_class import Mtrec
import torch


def test_config(trial, cfg, device):
    cfg = copy.deepcopy(cfg)
    hcf = cfg["hypertuning"]
    hidden_sizes = hcf["hidden_size"]
    nl_min, nl_max = hcf["num_layers"]["min"],hcf["num_layers"]["max"]
    lr_min, lr_max = hcf["lr"]["min"], hcf["lr"]["max"]
    batch_sizes = hcf["batch_size"]
    # Categorical net
    cfg["news_encoder"]["cfg_cat"]["hidden_size"] = trial.suggest_categorical("cat_hidden_size", hidden_sizes)
    cfg["news_encoder"]["cfg_cat"]["num_layers"] = trial.suggest_int("cat_num_layers", nl_min, nl_max)
    # NER net
    cfg["news_encoder"]["cfg_ner"]["hidden_size"] = trial.suggest_categorical("ner_hidden_size", hidden_sizes)
    cfg["news_encoder"]["cfg_ner"]["num_layers"] = trial.suggest_int("ner_num_layers", nl_min, nl_max)
    # User encoder
    cfg["user_encoder"]["hidden_size"] = trial.suggest_categorical("user_hidden_size", hidden_sizes)
    #Training
    cfg["trainer"]["batch_size"] = trial.suggest_categorical("batch_size", batch_sizes)
    
    cfg["trainer"]["lr_user"] = trial.suggest_float("lr_user", lr_min, lr_max, log=True)
    cfg["trainer"]["lr_news"] = trial.suggest_float("lr_news", lr_min, lr_max, log=True)
    cfg["trainer"]["lr_bert"] = trial.suggest_float("lr_bert", lr_min, lr_max, log=True)
    
    Mtrec_model = Mtrec(cfg, device=device)
    
    dataloaders = get_dataloaders(cfg)
    (dataloader_train, dataloader_val) = (dataloaders[0], dataloaders[1])
    model, best_validation_loss =       train(model     = Mtrec_model, 
                                       dataloader_train = dataloader_train, 
                                       dataloader_val   = dataloader_val, 
                                       cfg              = cfg["trainer"], 
                                       device           = device)
    
    return best_validation_loss

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test_hypertune', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
   
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    target_func = partial(test_config, cfg = cfg, device=device)

    study = optuna.create_study()
    study.optimize(target_func, n_trials=100)

    print("best parameters:\n", study.best_params)

if __name__ == "__main__":
    main()

