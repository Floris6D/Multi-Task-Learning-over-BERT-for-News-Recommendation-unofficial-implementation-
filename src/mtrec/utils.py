from transformers import BertTokenizer
import yaml
from NeRD_data import EB_NeRDDataset
from torch.utils.data import DataLoader

def load_configuration(config_file):
    file_path = f'src/mtrec/configs/{config_file}.yml'
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    # Convert string values to float in trainer
    cfg['trainer']['lr_user'] = float(cfg['trainer']['lr_user'])
    cfg['trainer']['lr_news'] = float(cfg['trainer']['lr_news'])
    cfg['trainer']['lr_bert'] = float(cfg['trainer']['lr_bert'])
    if "hypertuning" in cfg:
        cfg['hypertuning']['lr']['min'] = float(cfg['hypertuning']['lr']['min'])
        cfg['hypertuning']['lr']['max'] = float(cfg['hypertuning']['lr']['max'])
    return cfg

def get_dataloaders(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg['model']['pretrained_model_name'])
    return (DataLoader(
                    EB_NeRDDataset(tokenizer, **cfg['dataset'], split=split, override_eval2false= True),
                    batch_size=cfg["trainer"]["batch_size"], shuffle=True, num_workers=16, drop_last=True) 
                    for split in ['train', 'validation'] #TODO: add test, need to download
                    )	