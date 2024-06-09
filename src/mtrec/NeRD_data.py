import os
import random
import numpy as np
from pathlib import Path
import polars as pl

from ebrec.utils._polars import slice_join_dataframes, concat_str_columns
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)
from ebrec.utils._articles import create_article_id_to_value_mapping, convert_text2encoding_with_transformers

from torch.utils.data import Dataset

class EB_NeRDDataset(Dataset):
    def __init__(self, tokenizer, neg_sampling=False, split='train',**kwargs):
        '''
            kwargs: data_dir, history_size, batch_size
        '''
        self.tokenizer = tokenizer
        self.split = split
        self.neg_sampling = neg_sampling
        # Contains path (see config.yaml) to the json file
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Now load the data
        self.load_data()
        
        # Now tokenize the data and create the lookup tables
        self.tokenize_data()
        
        # Now create the X and y
        self.X = self.df_behaviors.drop('labels').with_columns(
            pl.col('article_ids_inview').list.len().alias('n_samples')
        )
        self.y = self.df_behaviors['labels']
        
    
    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title), batch_y
    
    def load_data(self):
        FULL_PATH = os.path.join(self.data_dir, self.split)
        
        # Load the data
        df_history = (
            pl.scan_parquet(os.path.join(FULL_PATH, 'history.parquet'))
            .pipe(
                truncate_history,
                column='article_id_fixed',
                history_size=self.history_size,
                padding_value=0,
                enable_warning=False,
            )
        )
        
        # Combine the behaviors and history data
        df_behaviors = (
            pl.scan_parquet(os.path.join(FULL_PATH, 'behaviors.parquet'))
            .collect()
            .pipe(
                slice_join_dataframes,
                df2=df_history.collect(),
                on='user_id',
                how='left',
            )
        )
        COLUMNS = ['user_id', 'article_id_fixed', 'article_ids_inview', 'article_ids_clicked', 'impression_id']
        
        # Now transform the data for negative sampling and add labels based on train, val, test
        if self.neg_sampling and self.split == 'train':
            df_behaviors.select(COLUMNS).pipe(
                sampling_strategy_wu2019,
                npratio=self.npratio,
                shuffle=True,
                with_replacement=True,
                seed=123,
            ).pipe(create_binary_labels_column).sample(fraction=self.dataset_fraction)
        else:
            df_behaviors.select(COLUMNS).pipe(create_binary_labels_column).sample(fraction=self.dataset_fraction)
        
        # Store the behaviors in the class
        self.df_behaviors = df_behaviors
        
        # Load the article data
        self.df_articles = pl.read_parquet(os.path.join(self.data_dir, 'articles.parquet'))
        
    def tokenize_data(self):
        # This concatenates the title with the subtitle in the DF, the cat_cal is the column name
        #TODO: JE: Maybe also add subtitle for prediction
        #df_articles, cat_cal = concat_str_columns(df = self.df_articles, columns=['subtitle', 'title'])
        
        # This add the bert encoding to the df
        self.df_articles, token_col_title = convert_text2encoding_with_transformers(self.df_articles, self.tokenizer, column='title', max_length=self.max_title_length)
        # Now create lookup tables
        article_mapping = create_article_id_to_value_mapping(df=self.df_articles, value_col=token_col_title)
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            article_mapping, unknown_representation='zeros'
        )
        self.unknown_index = [0]
        
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column='article_id_fixed',
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column='article_ids_inview',
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )
        
        
    
    
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