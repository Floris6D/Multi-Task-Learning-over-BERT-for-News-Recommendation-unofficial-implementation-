import os
import random
import numpy as np
from pathlib import Path
import polars as pl

from ebrec.utils._polars import slice_join_dataframes, concat_str_columns, filter_maximum_lengths_from_list
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
        # Contains path (see config.yaml) to the json file
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Now load the data (article_id_fixed is the history, generated using truncate history)
        COLUMNS = ['user_id', 'article_id_fixed', 'article_ids_inview', 'article_ids_clicked', 'impression_id']
        self.load_data(COLUMNS)
        
        # Now tokenize the data and create the lookup tables
        self.tokenize_data()
        
        # Now create the X and y
        self.X = self.df_behaviors.drop('labels').with_columns(
            pl.col('article_ids_inview').list.len().alias('n_samples')
        ) #Drop labels and add n_samples (which is the number of articles in the inview list)
        self.y = self.df_behaviors['labels']       
        
        self.create_category_labels()
        
        self.generate_ner_tag()
        # Lastly transform the data to get tokens and the right format for the model using the lookup tables
        (self.his_input_title, self.mask_his_input_title, self.pred_input_title, self.mask_pred_input_title), (self.y, self.c_y) = self.transform()
        
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
        y = (self.y[idx], self.c_y[idx])
        return x, y

    
    def load_data(self, COLUMNS):
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
        if self.wu_sampling and self.split == 'train':
            df_behaviors = df_behaviors.select(COLUMNS).pipe(
                sampling_strategy_wu2019,
                npratio=self.npratio,
                shuffle=True,
                with_replacement=True,
                seed=123,
            ).pipe(create_binary_labels_column).sample(fraction=self.dataset_fraction)
        elif self.split == 'validation':
            max_length = (df_behaviors['article_ids_inview'].list.lengths().max())
            df_behaviors = df_behaviors.with_columns([
                pl.col('article_ids_inview').apply(lambda x: pad_list(x, max_length)).alias('article_ids_inview')
            ])
            df_behaviors = df_behaviors.select(COLUMNS).pipe(create_binary_labels_column, shuffle=False).sample(fraction=self.dataset_fraction)           

        else:
            df_behaviors = df_behaviors.select(COLUMNS).pipe(create_binary_labels_column).sample(fraction=self.dataset_fraction)
        
        # Store the behaviors in the class
        self.df_behaviors = df_behaviors
        # Load the article data
        self.df_articles = pl.read_parquet(os.path.join(self.data_dir, 'articles.parquet'))
        
    def tokenize_data(self):
        # This concatenates the title with the subtitle in the DF, the cat_cal is the column name
        #TODO: JE: Maybe also add subtitle for prediction
        #df_articles, cat_cal = concat_str_columns(df = self.df_articles, columns=['subtitle', 'title'])
        
        # This add the bert tokenization to the df
        self.df_articles, col_name_token_title, col_name_mask = convert_text2encoding_with_transformers_tokenizers(self.df_articles, self.tokenizer, column='title', max_length=self.max_title_length)
    
        # Now create lookup tables
        article_mapping_token = create_article_id_to_value_mapping(df=self.df_articles, value_col=col_name_token_title, article_col='article_id')
        article_mapping_mask = create_article_id_to_value_mapping(df=self.df_articles, value_col=col_name_mask, article_col='article_id')
                
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            article_mapping_token, unknown_representation='zeros'
        )
        self.lookup_article_index_mask, self.lookup_article_matrix_mask = create_lookup_objects(
            article_mapping_mask, unknown_representation='zeros'
        )
        
        self.unknown_index = [0]
        
    def transform(self):
            # Map the article ids to the lookup table (not sure what this value should represent, I think it's the tokenized title)
            self.X = self.X.pipe(
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
            
            if not self.wu_sampling:
                repeats = np.array(self.X["n_samples"])
                # =>
                self.y = np.array(self.y.explode().to_list()).reshape(-1, 1)
                # =>
                his_input_title = repeat_by_list_values_from_matrix(
                    self.X['article_id_fixed'].to_list(),
                    matrix=self.lookup_article_matrix,
                    repeats=repeats,
                )
                # =>
                mask_his_input_title = repeat_by_list_values_from_matrix(
                    self.X['article_id_fixed'].to_list(),
                    matrix=self.lookup_article_matrix_mask,
                    repeats=repeats,
                )
                # =>
                pred_input_title = self.lookup_article_matrix[
                    self.X['article_ids_inview'].explode().to_list()
                ]
                mask_pred_input_title = self.lookup_article_matrix_mask[
                    self.X['article_ids_inview'].explode().to_list()
                ]
                     
            else:                
                self.y = np.array(self.y.to_list())
                # self.c_y = np.array(self.c_y.to_list()) 
                    
                his_input_title = self.lookup_article_matrix[
                    self.X['article_id_fixed'].to_list()
                ]
                mask_his_input_title = self.lookup_article_matrix_mask[
                    self.X['article_id_fixed'].to_list()
                ]
                pred_input_title = self.lookup_article_matrix[
                    self.X['article_ids_inview'].to_list()
                ]
                mask_pred_input_title = self.lookup_article_matrix_mask[
                    self.X['article_ids_inview'].to_list()
                ]
            
            pred_input_title = np.squeeze(pred_input_title, axis=2)
            mask_pred_input_title = np.squeeze(mask_pred_input_title, axis=2)
            
            his_input_title = np.squeeze(his_input_title, axis=2)
            mask_his_input_title = np.squeeze(mask_his_input_title, axis=2)
            
                        
            return (his_input_title, mask_his_input_title, pred_input_title, mask_pred_input_title), (self.y, self.c_y)
    
    def create_category_labels(self):       

        # Get unique categories and their corresponding indices
        unique_categories = self.df_articles.select(pl.col('category_str').unique()).to_series().to_list()

        # Create a mapping dictionary for category to one-hot vectors
        self.name_dict = {name: [0] * 25 for name in unique_categories}
        for i, name in enumerate(unique_categories):
            self.name_dict[name][i] = 1

        # Map categories to vectors directly in the DataFrame
        self.df_articles = self.df_articles.with_columns(
            pl.col('category_str').apply(self.map_category_to_vector).alias('category_vector')
        )

        # Convert article_id to category vectors using a dictionary for quick lookup
        article_to_vector = {row['article_id']: row['category_vector'] for row in self.df_articles.to_dicts()}

        # Generate labels using the precomputed dictionary
        labels = []
        for elem in self.df_behaviors['article_ids_inview']:
            vectors = []
            for id in elem:
                if id == 0:
                    vectors.append([0] * 25)
                else:
                    vectors.append(article_to_vector.get(id, [0] * 25))
            labels.append(vectors)

        self.c_y = np.array(labels)
              
        
    def map_category_to_vector(self, category_str):
            return self.name_dict[category_str]
        
    def generate_ner_tag(self):
        # print(self.df_articles.columns)
        test1 = self.df_articles.select(pl.col('ner_clusters'))
        test2 = self.df_articles.select(pl.col('title'))
        ner_labels = []
        internal = False               
        for elem, elem2 in zip(test2.to_series().to_list(), test1.to_series().to_list()):
            vector = []            
            for i in elem.split():
                if internal and i in elem2:
                    vector.append(2)
                elif i in elem2:
                    vector.append(1)
                    internal = True
                
                else:
                    vector.append(0)
                    internal = False
            ner_labels.append(vector)   
        self.ner_y = ner_labels

def pad_list(lst, length):
    lst = lst.to_list()
    lst.extend([0] * (length - len(lst)))
        
    return lst      

def convert_text2encoding_with_transformers_tokenizers(
    df: pl.DataFrame,
    tokenizer: AutoTokenizer,
    column: str,
    max_length: int = None,
) -> pl.DataFrame:
    """Converts text in a specified DataFrame column to tokens using a provided tokenizer.
    Args:
        df (pl.DataFrame): The input DataFrame containing the text column.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text. (from transformers import AutoTokenizer)
        column (str): The name of the column containing the text.
        max_length (int, optional): The maximum length of the encoded tokens. Defaults to None.
    Returns:
        pl.DataFrame: A new DataFrame with an additional column containing the encoded tokens.
    Example:
    >>> from transformers import AutoTokenizer
    >>> import polars as pl
    >>> df = pl.DataFrame({
            'text': ['This is a test.', 'Another test string.', 'Yet another one.']
        })
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> encoded_df, new_column = convert_text2encoding_with_transformers(df, tokenizer, 'text', max_length=20)
    >>> print(encoded_df)
        shape: (3, 2)
        ┌──────────────────────┬───────────────────────────────┐
        │ text                 ┆ text_encode_bert-base-uncased │
        │ ---                  ┆ ---                           │
        │ str                  ┆ list[i64]                     │
        ╞══════════════════════╪═══════════════════════════════╡
        │ This is a test.      ┆ [2023, 2003, … 0]             │
        │ Another test string. ┆ [2178, 3231, … 0]             │
        │ Yet another one.     ┆ [2664, 2178, … 0]             │
        └──────────────────────┴───────────────────────────────┘
    >>> print(new_column)
        text_encode_bert-base-uncased
    """
    text = df[column].to_list()
    # set columns
    new_column_id = f"{column}_encode_{tokenizer.name_or_path}"
    new_column_mask = f"{column}_mask_{tokenizer.name_or_path}"
    # If 'max_length' is provided then set it, else encode each string its original length
    padding = "max_length" if max_length else False
    encoded_tokens = tokenizer(
        text,
        add_special_tokens=False,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )
    input_ids_series = encoded_tokens['input_ids']
    token_masks = encoded_tokens['attention_mask']
    df = df.with_columns(pl.Series(new_column_id, input_ids_series))
    df = df.with_columns(pl.Series(new_column_mask, token_masks))
    return df, new_column_id, new_column_mask
    

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