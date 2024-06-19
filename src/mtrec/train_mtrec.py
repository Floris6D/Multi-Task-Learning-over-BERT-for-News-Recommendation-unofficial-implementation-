import argparse
from mtrec_class import Mtrec
from trainer import train
import torch
from utils import load_configuration, get_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--file', default='test', help='Path to the configuration file')
    args = parser.parse_args()
    cfg = load_configuration(args.file)
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    Mtrec_model = Mtrec(cfg, device=device)
    
    # (dataloader_train, dataloader_val, dataloader_test) = get_dataloaders(cfg)
    (dataloader_train, dataloader_val) = get_dataloaders(cfg)

    model, best_validation_loss =       train(model     = Mtrec_model, 
                                       dataloader_train = dataloader_train, 
                                       dataloader_val   = dataloader_val, 
                                       cfg              = cfg["trainer"])
    # results = test(news_encoder,
    #                user_encoder, 
    #                dataloader_test)
    
    #TODO: JE: Make submission file

if __name__ == "__main__":
    main()

