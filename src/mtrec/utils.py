from transformers import BertTokenizer
import yaml
from NeRD_data import EB_NeRDDataset
from torch.utils.data import DataLoader

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