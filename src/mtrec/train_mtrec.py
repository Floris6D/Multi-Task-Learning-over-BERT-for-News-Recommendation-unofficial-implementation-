import yaml
import argparse
import yaml

from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from dataloaders import get_dataloaders #TODO make compatible
from trainer import train, test

def load_configuration(config):
    file_path = f'configs/{config}.yml'
    with open(file_path, 'r') as file:
        configuration = yaml.safe_load(file)
    return configuration

def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test', help='Path to the configuration file')
    args = parser.parse_args()
    config = load_configuration(args.file)
    
    uec = config['user_encoder']
    user_encoder = UserEncoder(uec['input_size'], uec['embedding_dim'])
    nec = config['news_encoder']
    news_encoder = NewsEncoder(nec['tok_size'], nec['embedding_dim'], 
                               nec['num_classes'], nec['num_ner'])
    dataloader_train, dataloader_val, dataloader_test = get_dataloaders()

    user_encoder, news_encoder = train(user_encoder, 
                                       news_encoder, 
                                       dataloader_train, 
                                       dataloader_val, 
                                       config["trainer"])
    results = test(news_encoder,
                   user_encoder, 
                   dataloader_test)

if __name__ == "__main__":
    main()

