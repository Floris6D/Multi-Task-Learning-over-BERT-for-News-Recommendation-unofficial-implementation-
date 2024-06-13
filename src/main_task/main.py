from pathlib import Path
import polars as pl

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from dataloader import ArticleDataset
from encoders import UserEncoder, NewsEncoder
from NeRD_data import EB_NeRDDataset
import yaml

with open('/home/cedrik/Documents/Master_AI/Jaar1/RC/RecSys/src/mtrec/configs/config.yml') as f:
    cfg = yaml.safe_load(f)
    
def normalize(tensor):
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True)

def get_article_embedding(id):
    return df_articles.filter(pl.col('article_id') == id)['title'][0]

def nce_loss(s_pos, s_neg):
    s_pos_exp = torch.exp(s_pos)
    s_neg_exp = torch.exp(s_neg).sum(dim=-1)
    print(s_neg_exp)
    print(s_pos_exp)
    loss = -torch.log(s_pos_exp / (s_pos_exp + s_neg_exp))
    
    return loss

def score(candidate_news, user_representation):
    return torch.dot(normalize(candidate_news), normalize(user_representation))

# Loading in data
PATHART = Path('/home/cedrik/Documents/Master_AI/Jaar1/RC/RecSys/data/articles_large_only')
PATHBEV = Path('/home/cedrik/Documents/Master_AI/Jaar1/RC/RecSys/data/ebnerd_demo/train')
# PATHTEST = Path('/home/cedrik/Documents/Master_AI/Jaar1/RC/RecSys/data/ebnerd_testset/ebnerd_testset')

test_n = 2
df_articles = pl.scan_parquet(PATHART.joinpath('articles.parquet')).collect().head(test_n*2)
df_behaviour = pl.scan_parquet(PATHBEV.joinpath('behaviors.parquet')).collect().head(test_n*2)
df_history = pl.scan_parquet(PATHBEV.joinpath('history.parquet')).collect().head(test_n*2)
print(df_articles['article_id'])
print(df_articles.columns)
print(df_behaviour.columns)
print(df_history.columns)
# print(df_behaviour['article_ids_clicked'])

# Encoding article titles
batch_size = test_n
news_encoder = NewsEncoder()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
eb_nerd_dataset = EB_NeRDDataset(tokenizer, **cfg['dataset'])
dataloader1 = DataLoader(eb_nerd_dataset, batch_size=test_n, shuffle=True)


dataset = ArticleDataset(df_articles['title'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
user_encoder = UserEncoder(input_dim=768, hidden_dim=200)
encoded_titles = []

for data in tqdm(dataloader1):
    content, clicked = data
    history, masked_his, view, masked_view = content
    print(history.shape)
    print(view.shape)
    cls = news_encoder(history[0])
    user_representation = user_encoder(cls.unsqueeze(0))
    for article in view[0]:        
        cls = news_encoder(article.unsqueeze(0))
        print(score(cls, user_representation))
        
    
    print(cls.shape)
    
    
   
 
user_representation = user_encoder(encoded_batch.unsqueeze(0))
print(user_representation.shape)
test_score1 = score(news_encoder(get_article_embedding(3000022)), user_representation)
test_score2 = score(user_representation, user_representation)
print(test_score1)
print(test_score2)
print(nce_loss(test_score1, test_score2))

# # Testing with save and loading embeddings  
# df_articles.write_parquet('/home/cedrik/Documents/Master_AI/Jaar1/RC/RecSys/data/articles_with_embedding/articles.parquet')
# tensor_file_path = '/home/cedrik/Documents/Master_AI/Jaar1/RC/RecSys/data/articles_with_embedding/encoded_titles.pt'
# torch.save(encoded_titles, tensor_file_path)
# encoded_titles = torch.load(tensor_file_path)
# df_articles = df_articles.with_columns(pl.Series('encoded_title', encoded_titles))

# # Get the user representation for a history of articles
# clicked_news_vectors = get_article_embedding(3000022)
# print(clicked_news_vectors.shape)


# user_representation = user_encoder(df_articles['encoded_title'])
# print("User Representation Shape:", user_representation.shape) 
# print(user_representation.shape)

# # Click predictor
# test_score1 = score(get_article_embedding(3000022), user_representation)
# test_score2 = score(user_representation, user_representation)
# print(test_score1)
# print(test_score2)

# # Loss
# print(nce_loss(test_score1, test_score2))




