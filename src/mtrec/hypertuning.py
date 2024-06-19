import yaml
import argparse

#get data
from NeRD_data import EB_NeRDDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from trainer import train
from peft import LoraConfig, get_peft_model

import optuna

from functools import partial
from utils import load_configuration, get_dataloaders
import copy

def test_config(trial, cfg, bert):
    cfg = copy.deepcopy(cfg)
    hcf = cfg["hypertuning"]
    hidden_sizes = [int(x) for x in hcf["hidden_size"]]
    nl_min, nl_max = int(hcf["num_layers"]["min"]), int(hcf["num_layers"]["max"])
    lr_min, lr_max = hcf["lr"]["min"], hcf["lr"]["max"]
    batch_sizes = int(hcf["batch_size"])
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
    
    cfg["trainer"]["lr_user"] = trial.suggest_loguniform("lr_user", lr_min, lr_max)
    cfg["trainer"]["lr_news"] = trial.suggest_loguniform("lr_news", lr_min, lr_max)
    cfg["trainer"]["lr_bert"] = trial.suggest_loguniform("lr_bert", lr_min, lr_max)
    
    embedding_dim = bert.config.hidden_size
    user_encoder = UserEncoder(**cfg['user_encoder'], embedding_dim=embedding_dim)
    news_encoder = NewsEncoder(**cfg['news_encoder'], bert=bert, embedding_dim=embedding_dim)

    # (dataloader_train, dataloader_val, dataloader_test) = get_dataloaders(cfg)
    (dataloader_train, dataloader_val) = get_dataloaders(cfg)

    user_encoder, news_encoder, 
    best_validation_loss =       train(user_encoder     = user_encoder, 
                                       news_encoder     = news_encoder, 
                                       dataloader_train = dataloader_train, 
                                       dataloader_val   = dataloader_val, 
                                       cfg              = cfg["trainer"])
    
    return best_validation_loss

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test_hypertune', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
   
    bert = BertModel.from_pretrained(cfg['model']['pretrained_model_name'])
    # Get the embedding dimension
    bert = get_peft_model(bert, LoraConfig(cfg["lora_config"]))
    
    target_func = partial(test_config, cfg = cfg, bert = bert)
    study = optuna.create_study()
    study.optimize(target_func, n_trials=100)

    print("best parameters:\n", study.best_params)

if __name__ == "__main__":
    main()

