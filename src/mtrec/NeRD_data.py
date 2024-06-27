import os
import numpy as np
import polars as pl
import json
import re

from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import create_lookup_objects
from ebrec.utils._articles import create_article_id_to_value_mapping

from torch.utils.data import Dataset

from transformers import AutoTokenizer

class EB_NeRDDataset(Dataset):
    def __init__(self, tokenizer, wu_sampling=True, split='train',**kwargs):
        '''
            kwargs: data_dir, history_size, batch_size
            
            return (his_input_title, pred_input_title), y which is
            the tokenized history and the tokenized articles in the inview list together with the labels which is the click or not click based on the inview list
        '''
        #TODO: JE: Also make sure the dataloader works with negative sampling is False (doesn't work now because labels different sizes, see eval for better)
        self.tokenizer = tokenizer
        self.split = split
        self.wu_sampling = wu_sampling
        self.eval_mode = False if split == 'train' else True
        
        #self.eval_mode = False # Temporarily set to False for always wu sampling
        # Contains path (see config.yaml) to the json file
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.dataset_fraction = self.dataset_fraction if split != 'test' else self.testset_fraction
            
        # Now load the data (article_id_fixed is the history, generated using truncate history)
        COLUMNS = ['user_id', 'article_id_fixed', 'article_ids_inview', 'article_ids_clicked', 'impression_id']
        self.load_behaviors(COLUMNS)

        # Now load and tokenize the data of the articles and create the lookup tables
        self.df_articles = self.load_tokenize_articles()
        
        # Create category labels and add them to the data
        self.create_category_labels() # Now correct

        # Our extension for extended NER
        if self.extended_NER is False:
            self.generate_ner_tag()
        elif self.extended_NER is True:
            self.generate_extended_ner_tag()
        
        # Lastly transform the data to get tokens and the right format for the model using the lookup tables
        (self.his_input_title, self.mask_his_input_title, self.pred_input_title, self.mask_pred_input_title), self.y, self.id = self.transform()
        
    def __len__(self):
        return int(len(self.y))

    def __getitem__(self, idx):
        """
        his_input_title:        (samples, history_size, document_dimension)
        mask_his_input_title:   (samples, history_size, document_dimension)
        pred_input_title:       (samples, npratio, document_dimension)
        mask_pred_input_title:  (samples, npratio, document_dimension)
        batch_y:                (samples, npratio)
        """
        x = (self.his_input_title[idx], self.mask_his_input_title[idx], self.pred_input_title[idx], self.mask_pred_input_title[idx])
        y = (self.y[idx], self.c_y_his[idx], self.c_y_inview[idx], self.ner_y_his[idx], self.ner_y_inview[idx])
        impression_id = self.id[idx]
        return x, y, impression_id

    
    def load_behaviors(self, COLUMNS):
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
        
        # Now transform the data for negative sampling and add labels based on train, val, test
        if self.wu_sampling and self.eval_mode is False:
            df_behaviors = df_behaviors.select(COLUMNS).pipe(
                sampling_strategy_wu2019,
                npratio=self.npratio,
                shuffle=True,
                with_replacement=True,
                seed=123,
            ).pipe(create_binary_labels_column).sample(fraction=self.dataset_fraction)
        else:
            max_length = (df_behaviors['article_ids_inview'].list.lengths().max())
            df_behaviors = df_behaviors.with_columns([
                pl.col('article_ids_inview').apply(lambda x: pad_list(x, max_length)).alias('article_ids_inview')
            ])
            df_behaviors = df_behaviors.select(COLUMNS).pipe(create_binary_labels_column, shuffle=False).sample(fraction=self.dataset_fraction)           

        # Store the behaviors in the class
        self.df_behaviors = df_behaviors
        
        # Now create the X and y
        self.X = self.df_behaviors.drop('labels').with_columns(
            pl.col('article_ids_inview').list.len().alias('n_samples')
        ) #Drop labels and add n_samples (which is the number of articles in the inview list)
        self.y = self.df_behaviors['labels']  
        
    def load_tokenize_articles(self):
        # Load the article data
        df_articles = pl.read_parquet(os.path.join(self.data_dir, 'articles.parquet'))
        
        # This add the bert tokenization to the df
        df_articles, col_name_token_title, col_name_mask = convert_text2encoding_with_transformers_tokenizers(df_articles, self.tokenizer, column='title', max_length=self.max_title_length)
    
        # Now create lookup tables
        article_mapping_token = create_article_id_to_value_mapping(df=df_articles, value_col=col_name_token_title, article_col='article_id')
        article_mapping_mask = create_article_id_to_value_mapping(df=df_articles, value_col=col_name_mask, article_col='article_id')
                
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            article_mapping_token, unknown_representation='zeros'
        )
        self.lookup_article_index_mask, self.lookup_article_matrix_mask = create_lookup_objects(
            article_mapping_mask, unknown_representation='zeros'
        )
        
        self.unknown_index = [0]
        
        return df_articles
        
    def transform(self):
            # Map the article ids to the lookup table (not sure what this value should represent, I think it's the tokenized title)
            self.data = self.X.pipe(
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
            
                                     
            self.y = np.array(self.y.to_list())
                
            his_input_title = self.lookup_article_matrix[
                self.data['article_id_fixed'].to_list()
            ]
            mask_his_input_title = self.lookup_article_matrix_mask[
                self.data['article_id_fixed'].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                self.data['article_ids_inview'].to_list()
            ]
            mask_pred_input_title = self.lookup_article_matrix_mask[
                self.data['article_ids_inview'].to_list()
            ]
            
            pred_input_title = np.squeeze(pred_input_title, axis=2)
            mask_pred_input_title = np.squeeze(mask_pred_input_title, axis=2)
            
            his_input_title = np.squeeze(his_input_title, axis=2)
            mask_his_input_title = np.squeeze(mask_his_input_title, axis=2)
            
            # Also return the impression_id
            impression_ids = self.data['impression_id'].to_list()
            
            return (his_input_title, mask_his_input_title, pred_input_title, mask_pred_input_title), (self.y), impression_ids
    
    def create_category_labels(self): 
        '''
        This function loops over all the behaviors and is therefore not efficient but for simplicity I kept this
        '''      

        # Load the category mapping (to prevent ordering changes)
        with open(os.path.join('src/mtrec/configs', 'category_mapping.json'), 'r') as f:
            self.category_mapping = json.load(f)
        
        # Map categories to idx directly in the DataFrame
        self.df_articles = self.df_articles.with_columns(
            pl.col('category_str').apply(self.map_category_to_vector).alias('category_idx')
        )
        
        # Also create a lookup dictionary for an article id to the category index
        article_to_category_idx = dict(zip(self.df_articles['article_id'], self.df_articles['category_idx']))

        # Generate labels for all the inview articles
        labels = []
        for elem in self.df_behaviors['article_ids_inview']:
            cat_idx = []
            for id in elem:
                if id == 0: # This means padding was introduced
                    cat_idx.append(-1)
                else:
                    # Now return the category index of the article that belong to the id in the article_id column
                    cat_idx.append(article_to_category_idx[id])
            labels.append(cat_idx)
        
        self.c_y_inview = np.array(labels) 
        # Generate labels for all the history articles
        labels = [] 
        for elem in self.df_behaviors['article_id_fixed']:
            cat_idx = []            
            for id in elem:
                if id == 0: # This means padding was introduced
                    cat_idx.append(-1)
                else:
                    cat_idx.append(article_to_category_idx[id])
            labels.append(cat_idx)    
        
        self.c_y_his = np.array(labels)
        
    def map_category_to_vector(self, category_str):
            return self.category_mapping[category_str]
        
        
    def generate_ner_tag(self):        
        # Load the entity mapping (too make sure ordering doesn't change when loading the data again)
        with open(os.path.join('src/mtrec/configs', 'entity_mapping.json'), 'r') as f:
            self.entity_mapping = json.load(f)
        
        # Now for each article and the create a list with the NER tags
        NER_labels = []
        for i in range(len(self.df_articles)): # Loop over all the articles
            article_title = self.df_articles['title'][i]
            tokenized_title = self.tokenizer(article_title, add_special_tokens=True)
            row_entity = [self.entity_mapping['O']] * len(tokenized_title) # Create a list with the same length as the title (0 is None)
            for ner_cluster, entity_group in zip(self.df_articles['ner_clusters'][i], self.df_articles['entity_groups'][i]): # Loop over all the NER clusters
                # Tokenize the ner_cluster
                tokenized_ner_cluster = self.tokenizer(ner_cluster)
                
                # Now find the position of the ner_cluster in the title
                idx = find_named_entity_position(tokenized_title, tokenized_ner_cluster)
                if idx != -1:
                    for j in range(len(tokenized_ner_cluster)):
                        if j == 0:
                            row_entity[idx + j] = self.entity_mapping['B-P']
                        else:
                            row_entity[idx + j] = self.entity_mapping['I-P']
            NER_labels.append(row_entity)
            
        # Create a new column with the NER_labels in the articles DataFrame
        self.df_articles = self.df_articles.with_columns(pl.Series('ner_labels', NER_labels))
        
        # Create a lookup dictionary for an article id to the NER labels
        article_to_NER = dict(zip(self.df_articles['article_id'], self.df_articles['ner_labels']))
        
        
        # Generate labels for all the inview articles
        labels = []
        for elem in self.df_behaviors['article_ids_inview']:
            ner_inview_idx = []
            for id in elem:
                ner_idx = []
                if id == 0: # This means padding was introduced
                    ner_idx.append([-1] * self.max_title_length)
                else:
                    # Now return the category index of the article that belong to the id in the article_id column
                    ner_label = article_to_NER[id].to_list()
                    
                    # Ner label should be the max length
                    if len(ner_label) < self.max_title_length:
                        ner_label.extend([-1] * (self.max_title_length - len(ner_label)))
                    elif len(ner_label) > self.max_title_length:
                        ner_label = ner_label[:self.max_title_length]
                    
                    ner_idx.append(ner_label)
                ner_inview_idx.append(ner_idx)
            labels.append(ner_inview_idx)
        
        self.ner_y_inview = np.array(labels)
        self.ner_y_inview = self.ner_y_inview.squeeze(axis = 2)
        # Generate labels for all the history articles
        labels = [] 
        for elem in self.df_behaviors['article_id_fixed']:
            ner_inview_idx = []
            for id in elem:
                ner_idx = []
                if id == 0: # This means padding was introduced
                    ner_idx.append([-1] * self.max_title_length)
                else:
                    # Now return the category index of the article that belong to the id in the article_id column
                    ner_label = article_to_NER[id].to_list()
                    
                    # Ner label should be the max length
                    if len(ner_label) < self.max_title_length:
                        ner_label.extend([-1] * (self.max_title_length - len(ner_label)))
                    elif len(ner_label) > self.max_title_length:
                        ner_label = ner_label[:self.max_title_length]
                    
                    ner_idx.append(ner_label)
                ner_inview_idx.append(ner_idx)
            labels.append(ner_inview_idx)
        
        self.ner_y_his = np.array(labels)
        self.ner_y_his = self.ner_y_his.squeeze(axis = 2)
        
        
        
    def generate_extended_ner_tag(self):         
        # Open the entity mapping file to prevent ordering changes
        with open(os.path.join('src/mtrec/configs', 'extended_entity_mapping.json'), 'r') as f:
            self.entity_mapping = json.load(f)
        
        # Now for each article and the create a list with the NER tags
        NER_labels = []
        for i in range(len(self.df_articles)): # Loop over all the articles
            article_title = self.df_articles['title'][i]
            tokenized_article_title = self.tokenizer.tokenize(article_title, add_special_tokens=True)
            row_entity = [0] * len(tokenized_article_title) # Create a list with the same length as the title (0 is None)
            for ner_cluster, entity_group in zip(self.df_articles['ner_clusters'][i], self.df_articles['entity_groups'][i]): # Loop over all the NER clusters
                # Tokenize the ner_cluster
                tokenized_ner_cluster = self.tokenizer.tokenize(ner_cluster)
                
                # Now find the position of the ner_cluster in the title
                idx = find_named_entity_position(tokenized_article_title, tokenized_ner_cluster)
                if idx != -1:
                    for j in range(len(tokenized_ner_cluster)):
                        row_entity[idx + j] = self.entity_mapping[entity_group]
            NER_labels.append(row_entity)
            
        # Create a new column with the NER_labels in the articles DataFrame
        self.df_articles = self.df_articles.with_columns(pl.Series('ner_labels', NER_labels))
        
        # Create a lookup dictionary for an article id to the NER labels
        article_to_NER = dict(zip(self.df_articles['article_id'], self.df_articles['ner_labels']))
        
        
        # Generate labels for all the inview articles
        labels = []
        for elem in self.df_behaviors['article_ids_inview']:
            ner_inview_idx = []
            for id in elem:
                ner_idx = []
                if id == 0: # This means padding was introduced
                    ner_idx.append([-1] * self.max_title_length)
                else:
                    # Now return the category index of the article that belong to the id in the article_id column
                    ner_label = article_to_NER[id].to_list()
                    
                    # Ner label should be the max length
                    if len(ner_label) < self.max_title_length:
                        ner_label.extend([-1] * (self.max_title_length - len(ner_label)))
                    elif len(ner_label) > self.max_title_length:
                        ner_label = ner_label[:self.max_title_length]
                    
                    ner_idx.append(ner_label)
                ner_inview_idx.append(ner_idx)
            labels.append(ner_inview_idx)
        
        self.ner_y_inview = np.array(labels)
        self.ner_y_inview = self.ner_y_inview.squeeze(axis = 2)
        # Generate labels for all the history articles
        labels = [] 
        for elem in self.df_behaviors['article_id_fixed']:
            ner_inview_idx = []
            for id in elem:
                ner_idx = []
                if id == 0: # This means padding was introduced
                    ner_idx.append([-1] * self.max_title_length)
                else:
                    # Now return the category index of the article that belong to the id in the article_id column
                    ner_label = article_to_NER[id].to_list()
                    
                    # Ner label should be the max length
                    if len(ner_label) < self.max_title_length:
                        ner_label.extend([-1] * (self.max_title_length - len(ner_label)))
                    elif len(ner_label) > self.max_title_length:
                        ner_label = ner_label[:self.max_title_length]
                    
                    ner_idx.append(ner_label)
                ner_inview_idx.append(ner_idx)
            labels.append(ner_inview_idx)
        
        self.ner_y_his = np.array(labels)
        self.ner_y_his = self.ner_y_his.squeeze(axis = 2)
          

def pad_list(lst, length):
    lst = lst.to_list()
    lst.extend([0] * (length - len(lst)))
  
    return lst      


#This function was taken from the ebrec.utils._articles.py file, but modified for our needs
def convert_text2encoding_with_transformers_tokenizers( df: 
                    pl.DataFrame, tokenizer: AutoTokenizer, column: str, max_length: int = None):
    text = df[column].to_list()
    # set columns
    new_column_id = f"{column}_encode_{tokenizer.name_or_path}"
    new_column_mask = f"{column}_mask_{tokenizer.name_or_path}"
    # If 'max_length' is provided then set it, else encode each string its original length
    padding = "max_length" if max_length else False
    encoded_tokens = tokenizer(
        text,
        add_special_tokens=True, ### Now the tokenizer will add the special tokens
        padding=padding,
        max_length=max_length,
        truncation=True,
    )
    input_ids_series = encoded_tokens['input_ids']
    token_masks = encoded_tokens['attention_mask']
    df = df.with_columns(pl.Series(new_column_id, input_ids_series))
    df = df.with_columns(pl.Series(new_column_mask, token_masks))
    return df, new_column_id, new_column_mask
    
   
    
def find_named_entity_position(title_list, named_entity):
    len_title = len(title_list)
    len_entity = len(named_entity)
    
    # Loop through the title list with a window of size len_entity
    for i in range(len_title - len_entity + 1):
        # Check if the current slice matches the named entity
        if title_list[i:i + len_entity] == named_entity:
            return i
    
    return -1  # Return -1 if the named entity is not found