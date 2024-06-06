import os
import random
import numpy as np
from pathlib import Path
import polars as pl

from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)

from torch.utils.data import Dataset

class EB_NeRDDataset(Dataset):
    def __init__(self, tokenizer, split='train',**kwargs):
        '''
            kwargs: data_dir, history_size
        '''
        self.tokenizer = tokenizer
        self.split = split
        # Contains path (see config.yaml) to the json file
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Now load the data
        self.load_data()
    
    def __len__(self):
        return len(self.df_behaviors)

    def __getitem__(self, index):
        # Get the title and label
        title = self.full_behaviors['title'].iloc[index]
        label = ...
  
        return title, label
    
    def load_data(self):
        FULL_PATH = os.path.join(self.data_dir, self.split)
        
        # Load the data
        df_behaviors = pl.scan_parquet(os.path.join(FULL_PATH, 'behaviors.parquet'))
        df_history = pl.scan_parquet(os.path.join(FULL_PATH, 'history.parquet'))
        df_history = df_history.pipe(
                        truncate_history,
                        column='article_id_fixed',
                        history_size=self.history_size,
                        padding_value=0,
                        enable_warning=False,
                    )
        
        self.full_behaviors = slice_join_dataframes(
                df1=df_behaviors.collect(),
                df2=df_history.collect(),
                on='user_id',
                how="left",
            )
        
        # Load the article data
        self.df_articles = pl.scan_parquet(os.path.join(self.data_dir, 'articles.parquet'))
        
    
    
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