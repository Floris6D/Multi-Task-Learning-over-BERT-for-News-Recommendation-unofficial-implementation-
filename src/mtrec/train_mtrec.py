import yaml
import argparse

#get data
from NeRD_data import EB_NeRDDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from trainer import train
#from peft import LoraConfig, get_peft_model

def load_configuration(config):
    file_path = f'src/mtrec/configs/{config}.yml'
    with open(file_path, 'r') as file:
        configuration = yaml.safe_load(file)
    return configuration

def get_dataloaders(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])
    return (DataLoader(
                    EB_NeRDDataset(tokenizer, **cfg['dataset'], split=split, override_eval2false= True),
                    batch_size=cfg["trainer"]["batch_size"], shuffle=True, num_workers=16, drop_last=True) 
                    for split in ['train', 'validation'] #TODO: add test, need to download
                    )	

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
    

    bert = BertModel.from_pretrained(cfg['model']['pretrained_model_name'])
    #bert = get_peft_model(bert, LoraConfig(cfg["lora_config"]))
    
    user_encoder = UserEncoder(**cfg['user_encoder'])
    news_encoder = NewsEncoder(**cfg['news_encoder'], bert=bert)

    
    
    # (dataloader_train, dataloader_val, dataloader_test) = get_dataloaders(cfg)
    (dataloader_train, dataloader_val) = get_dataloaders(cfg)

    user_encoder, news_encoder = train(user_encoder     = user_encoder, 
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

