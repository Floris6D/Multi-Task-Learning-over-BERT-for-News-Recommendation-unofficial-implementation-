import os
import random
import numpy as np
from pathlib import Path
import polars as pl

from torch.utils.data import Dataset

class EB_NeRDDataset(Dataset):
    def __init__(self, tokenizer, split='train',**kwargs):
        '''
            kwargs: max_length, hist_dim, cand_dim
        '''
        self.tokenizer = tokenizer
        self.split = split
        # Contains path (see config.yaml) to the json file
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Now load the data
        self.load_data()
    
    def __len__(self):
        return len(self.log)

    def __getitem__(self, index):
        user = self.log[index]

        history = user['history']
        unlike = user['unlike']

        hist_video = self.get_video(history)
        # hist_video = self.normal_sample(hist_video, self.hist_dim)

        unlike_video = self.get_video(unlike)
        # unlike_video = self.normal_sample(unlike_video, self.cand_dim)

        hist_video, cand_video = hist_video[1:], [hist_video[0]] + unlike_video
        

        hist_ids, hist_mask = self.encode(hist_video)
        hist_ids = self.normal_sample(hist_ids, self.hist_dim)
        hist_mask = self.normal_sample(hist_mask, self.hist_dim)
        
        cand_ids, cand_mask = self.encode(cand_video)
        cand_ids = self.normal_sample(cand_ids, self.cand_dim)
        cand_mask = self.normal_sample(cand_mask, self.cand_dim)

        idx = list(range(self.cand_dim))
        random.shuffle(idx)
        cand_ids = [cand_ids[i] for i in idx]
        cand_mask = [cand_mask[i] for i in idx]
        label = [np.argmin(idx)]

        return hist_ids, hist_mask, cand_ids, cand_mask, label
    
    def load_data(self):
        FULL_PATH = os.path.join(self.data_dir, self.split)
        
        # Load the data
        df_behaviors = pl.scan_parquet(os.path.join(FULL_PATH, 'behaviors.parquet'))
        df_history = pl.scan_parquet(os.path.join(FULL_PATH, 'history.parquet'))
    
    
    # def encode(self, batch):
    #     batch = [tokenize(sent) for sent in batch]
    #     token_item = self.tokenizer(batch, padding="max_length", truncation=True, max_length=self.max_length, add_special_tokens=True)
    #     return token_item['input_ids'], token_item['attention_mask']

    # def format_srt(self, str_input):
    #     return str_input.replace('"', '')

    # def normal_sample(self, input, length):
    #     n_padding = len(input[-1])
    #     n_extending = length - len(input)        
    #     tokens = input + ([[0] * n_padding] * n_extending)
    #     return tokens[:length]