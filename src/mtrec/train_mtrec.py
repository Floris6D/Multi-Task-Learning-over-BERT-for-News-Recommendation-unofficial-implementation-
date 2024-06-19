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
from utils import load_configuration, get_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
    

    bert = BertModel.from_pretrained(cfg['model']['pretrained_model_name'])
    # Get the embedding dimension
    embedding_dim = bert.config.hidden_size
    bert = get_peft_model(bert, LoraConfig(cfg["lora_config"]))
    
    user_encoder = UserEncoder(**cfg['user_encoder'], embedding_dim=embedding_dim)
    news_encoder = NewsEncoder(**cfg['news_encoder'], bert=bert, embedding_dim=embedding_dim)

    
    
    # (dataloader_train, dataloader_val, dataloader_test) = get_dataloaders(cfg)
    (dataloader_train, dataloader_val) = get_dataloaders(cfg)

    user_encoder, news_encoder, best_validation_loss =       train(user_encoder     = user_encoder, 
                                       news_encoder     = news_encoder, 
                                       dataloader_train = dataloader_train, 
                                       dataloader_val   = dataloader_val, 
                                       cfg              = cfg["trainer"])
    # results = test(news_encoder,
    #                user_encoder, 
    #                dataloader_test)
    
    #TODO: JE: Make submission file

if __name__ == "__main__":
    main()

