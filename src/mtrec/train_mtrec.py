import yaml
import argparse
import yaml

#get data
from NeRD_data import EB_NeRDDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from trainer import train, test

def load_configuration(config):
    file_path = f'configs/{config}.yml'
    with open(file_path, 'r') as file:
        configuration = yaml.safe_load(file)
    return configuration

def get_dataloaders(cfg):

    tokenizer = BertTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])
    return (DataLoader(
                    EB_NeRDDataset(tokenizer, **cfg['dataset'], split=split),
                    batch_size=cfg["trainer"]["batch_size"], shuffle=True) 
                    for split in ['train', 'val', 'test']
            )	

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
    
    # uec = cfg['user_encoder']
    # user_encoder = UserEncoder(uec['input_size'], uec['embedding_dim'])
    # nec = cfg['news_encoder']
    # news_encoder = NewsEncoder(nec['tok_size'], nec['embedding_dim'], 
    #                            nec['num_classes'], nec['num_ner'])

    user_encoder = UserEncoder(**cfg['user_encoder'])
    news_encoder = NewsEncoder(**cfg['news_encoder'])
    
    (dataloader_train, dataloader_val, dataloader_test) = get_dataloaders(cfg)

    user_encoder, news_encoder = train(user_encoder, 
                                       news_encoder, 
                                       dataloader_train, 
                                       dataloader_val, 
                                       cfg["trainer"])
    results = test(news_encoder,
                   user_encoder, 
                   dataloader_test)

if __name__ == "__main__":
    main()

