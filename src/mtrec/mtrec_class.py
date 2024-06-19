from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from transformers import BertTokenizer, BertModel
from peft import LoraConfig, get_peft_model
from trainer import get2device, category_loss, NER_loss, cross_product, main_loss
import torch

class Mtrec:
    def __init__(self, cfg, device:str = "cpu"):
        bert = BertModel.from_pretrained(cfg['model']['pretrained_model_name'])
        # Get the embedding dimension
        embedding_dim = bert.config.hidden_size
        self.bert = get_peft_model(bert, LoraConfig(cfg["lora_config"]))
        
        self.user_encoder = UserEncoder(**cfg['user_encoder'], embedding_dim=embedding_dim)
        self.news_encoder = NewsEncoder(**cfg['news_encoder'], bert=bert, embedding_dim=embedding_dim)
        
        self.device = device
    
    
    def train(self, dataloader_train, optimizer, print_flag, scoring_function:callable = cross_product, criterion:callable = main_loss):
        total_loss = 0
        total_main_loss = 0
        self.news_encoder.train()
        self.user_encoder.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            # Get the data
            (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = get2device(data, self.device)
            # Get the embeddings
            inview_news_embeddings, inview_news_cat, inview_news_ner = self.news_encoder(news_tokens, news_mask)  
            history_news_embeddings, history_news_cat, history_news_ner = self.news_encoder(user_histories, user_mask) 
            user_embeddings = self.user_encoder(history_news_embeddings)
            # AUX task: Category prediction            
            cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
            # AUX task: NER 
            ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his, news_mask, user_mask)
            # MAIN task: Click prediction
            scores = scoring_function(user_embeddings, inview_news_embeddings)
            main_loss = criterion(scores, labels)
            # Backpropagation           
            #optimizer.pc_backward([main_loss, cat_loss, ner_loss]) #TODO: PCGrad
            main_loss.backward()
            optimizer.step()
            total_loss += main_loss.item() + cat_loss.item() + ner_loss.item()
            total_main_loss += main_loss.item()
        
        # TODO: JE willen we delen door totaal aantal datapunten want dan moet je len(dataloader_train.dataset) doen
        total_loss /= len(dataloader_train.dataset)
        total_main_loss /= len(dataloader_train.dataset)
        if print_flag:
            print(f"Training total Loss: {total_loss}")
            print(f"Training main Loss: {total_main_loss}")
            
        return total_loss, total_main_loss
    
    def validate(self, dataloader_val, print_flag, scoring_function:callable = cross_product, criterion:callable = main_loss):
        #validation
        self.user_encoder.eval()
        self.news_encoder.eval()
        total_loss_val, total_main_loss_val = 0 , 0

        #df_val_data = dataloader_val.dataset.X
        for data in dataloader_val:
            # Get the data
            (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = get2device(data, self.device)
            
            # Get the embeddings
            inview_news_embeddings, inview_news_cat, inview_news_ner = self.news_encoder(news_tokens, news_mask)  
            history_news_embeddings, history_news_cat, history_news_ner = self.news_encoder(user_histories, user_mask) 
            user_embeddings = self.user_encoder(history_news_embeddings)

            # AUX task: Category prediction            
            cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
            # AUX task: NER 
            ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his, news_mask, user_mask)                    
            # MAIN task: Click prediction
            scores = scoring_function(user_embeddings, inview_news_embeddings) # batch_size * N
            main_loss = criterion(scores, labels)
            # Metrics
            total_loss_val += main_loss.item() + cat_loss.item() + ner_loss.item()
            total_main_loss_val += main_loss.item()
        
        
        # TODO: JE willen we delen door totaal aantal datapunten want dan moet je len(dataloader_train.dataset) doen
        total_loss /= len(dataloader_val.dataset)
        total_main_loss /= len(dataloader_val.dataset)
        if print_flag: 
            print(f"Validation total Loss: {total_loss_val}")
            print(f"Validation main Loss: {total_main_loss_val}")
            
        return total_loss_val, total_main_loss_val
    
    def predict(self, dataloader_test):
        # Set to eval mode
        self.user_encoder.eval()
        self.news_encoder.eval()
        
        total_scores, total_labels = torch.Tensor([]), torch.Tensor([])
        for data in dataloader_test:
            # if print_flag: print("SKIPPING VALIDATION FOR DEBUGGING")
            # break #TODO remove
            # Get the data
            (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview) = get2device(data, device)
            print("check1")
            # Get the embeddings
            inview_news_embeddings, inview_news_cat, inview_news_ner = news_encoder(news_tokens, news_mask)  
            print("check2")
            history_news_embeddings, history_news_cat, history_news_ner = news_encoder(user_histories, user_mask) 
            print("check3")
            user_embeddings = user_encoder(history_news_embeddings)
            print("check4")
            # AUX task: Category prediction            
            cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
            # AUX task: NER 
            ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his, news_mask, user_mask)                    
            # MAIN task: Click prediction
            scores = scoring_function(user_embeddings, inview_news_embeddings) # batch_size * N
            main_loss = criterion(scores, labels)
        
        
        
        # Save the scores and labels
        total_scores = torch.cat([total_scores, scores], dim=0)
        total_labels = torch.cat([total_labels, labels], dim=0)
    
    def save_model(self, path):
        pass
    
    
    def load_checkpoint(self, path):
        pass