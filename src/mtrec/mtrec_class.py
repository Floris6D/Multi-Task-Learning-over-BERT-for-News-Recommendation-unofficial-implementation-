from user_encoder import UserEncoder
from news_encoder import NewsEncoder
from transformers import  BertModel
from peft import LoraConfig, get_peft_model
from trainer import get2device, category_loss, NER_loss, cross_product, main_loss
import torch
import polars as pl
from utils import timer

class Mtrec(torch.nn.Module):
    def __init__(self, cfg, device:str = "cpu"):
        super().__init__()
        bert = BertModel.from_pretrained(cfg['model']['pretrained_model_name']).to(device)
        # Get the embedding dimension
        embedding_dim = bert.config.hidden_size
        self.bert = get_peft_model(bert, LoraConfig(cfg["lora_config"]))
        
        self.user_encoder = UserEncoder(**cfg['user_encoder'], embedding_dim=embedding_dim, device=device).to(device)
        self.news_encoder = NewsEncoder(**cfg['news_encoder'], bert=bert, embedding_dim=embedding_dim, extended_NER = cfg['dataset']['extended_NER'], device=device).to(device)
        
        self.device = device
    
    @timer
    def train(self, dataloader_train, optimizer, print_flag, cfg, scoring_function:callable = cross_product, criterion:callable = main_loss):
        total_loss, total_main_loss, total_cat_loss, total_ner_loss = 0, 0, 0, 0
        self.news_encoder.train()
        self.user_encoder.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            # Get the data
            (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview), _ = get2device(data, self.device)
            # Get the embeddings
            inview_news_embeddings, inview_news_cat, inview_news_ner, inview_mask_ner = self.news_encoder(news_tokens, news_mask)  
            history_news_embeddings, history_news_cat, history_news_ner, history_mask_ner = self.news_encoder(user_histories, user_mask) 
            user_embeddings = self.user_encoder(history_news_embeddings)
            # MAIN task: Click prediction
            scores = scoring_function(user_embeddings, inview_news_embeddings)
            main_loss = criterion(scores, labels)
            losses = [main_loss] # List of losses to backpropagate for PCGrad
            total_loss += main_loss.item() 
            total_main_loss += main_loss.item()
            # AUX task: Category prediction            
            if not cfg["skip_cat"]:
                cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
                losses.append(cat_loss)
                total_cat_loss += cat_loss.item()
                total_loss += cat_loss.item()
            # AUX task: NER 
            if not cfg["skip_ner"]:
                ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his, inview_mask_ner, history_mask_ner)
                losses.append(ner_loss)
                total_ner_loss += ner_loss.item()
                total_loss += ner_loss.item()
            # Backpropagation
            if not cfg["skip_gs"]: 
                optimizer.pc_backward(losses) 
            else:
                total_loss.backward()
            optimizer.step()            
            break #TODO verwijderen!!!
        
        # TODO: JE willen we delen door totaal aantal datapunten want dan moet je len(dataloader_train.dataset) doen
        total_loss /= len(dataloader_train.dataset)
        total_main_loss /= len(dataloader_train.dataset)
        total_cat_loss /= len(dataloader_train.dataset)
        total_ner_loss /= len(dataloader_train.dataset)
        if print_flag:
            print(f"Training total Loss: {total_loss}")
            print(f"Training main Loss: {total_main_loss}")
            
        return total_loss, total_main_loss, total_cat_loss, total_ner_loss
    
    @timer
    def validate(self, dataloader_val, print_flag, cfg, scoring_function:callable = cross_product, criterion:callable = main_loss):
        #validation
        self.user_encoder.eval()
        self.news_encoder.eval()
        total_loss, total_main_loss, total_cat_loss, total_ner_loss = 0, 0, 0, 0
        #df_val_data = dataloader_val.dataset.X
        for data in dataloader_val:
            # Get the data
            (user_histories, user_mask, news_tokens, news_mask), (labels, c_labels_his, c_labels_inview, ner_labels_his, ner_labels_inview), _ = get2device(data, self.device)
            # No gradient calculation needed
            with torch.no_grad():
                # Get the embeddings
                inview_news_embeddings, inview_news_cat, inview_news_ner, inview_mask_ner = self.news_encoder(news_tokens, news_mask)  
                history_news_embeddings, history_news_cat, history_news_ner, history_mask_ner = self.news_encoder(user_histories, user_mask) 
                user_embeddings = self.user_encoder(history_news_embeddings)
                total_loss = 0
                # AUX task: Category prediction            
                if not cfg["skip_cat"]:
                    cat_loss = category_loss(inview_news_cat, history_news_cat, c_labels_inview, c_labels_his)
                    total_loss += cat_loss.item()
                    total_cat_loss += cat_loss.item()
                # AUX task: NER 
                if not cfg["skip_ner"]:
                    ner_loss = NER_loss(inview_news_ner, history_news_ner, ner_labels_inview, ner_labels_his, inview_mask_ner, history_mask_ner)
                    total_loss += ner_loss.item()
                    total_ner_loss += ner_loss.item()
                # MAIN task: Click prediction
                scores = scoring_function(user_embeddings, inview_news_embeddings) # batch_size * N
                main_loss = criterion(scores, labels).item()
                # Metrics
                total_loss += main_loss 
                total_main_loss += main_loss               
                break #TODO verwijderen!!!
        # TODO: JE willen we delen door totaal aantal datapunten want dan moet je len(dataloader_train.dataset) doen
        total_loss /= len(dataloader_val.dataset)
        total_main_loss /= len(dataloader_val.dataset)
        if print_flag: 
            print(f"Validation total Loss: {total_loss}")
            print(f"Validation main Loss: {total_main_loss}")
        return total_loss, total_main_loss, total_cat_loss, total_ner_loss
    
    def predict(self, dataloader_test, scoring_function:callable = cross_product, criterion:callable = main_loss):
        # Set to eval mode
        self.user_encoder.eval()
        self.news_encoder.eval()
        
        total_scores, total_labels, impression_ids = [], [], []
        for data in dataloader_test:
            # Get the data
            (user_histories, user_mask, news_tokens, news_mask), (labels, _, _, _, _), impression_id = get2device(data, self.device)

            with torch.no_grad():
                # Get the embeddings
                # news_tokens shape: batch_size * max_inview_articles * max_title length
                inview_news_embeddings, _, _ = self.news_encoder(news_tokens, news_mask, validation=True)  
                history_news_embeddings, _, _ = self.news_encoder(user_histories, user_mask, validation=True) 
                user_embeddings = self.user_encoder(history_news_embeddings)
                    
                # MAIN task: Click prediction
                scores = scoring_function(user_embeddings, inview_news_embeddings) # batch_size * N
                
                # Save the impression_ids
                impression_ids.extend(impression_id.tolist())
                
                # Now save the scores and labels in lists and remove the padding
                for row_idx in range(scores.shape[0]):
                    
                    
                    row_label_mask = news_mask[row_idx, :, :].sum(dim=1) > 0
                    row_score = scores[row_idx, row_label_mask]
                    row_label = labels[row_idx, row_label_mask]
                    
                    # Apply softmax
                    row_score = torch.nn.functional.softmax(row_score, dim=0)

                    total_scores.append(row_score.tolist())
                    total_labels.append(row_label.tolist())
                
        # Set the results in polars dataframe
        pred_df = pl.DataFrame(
            {
                "impression_id": impression_ids,
                "scores": total_scores,
                "labels": total_labels
            }
        )
        return pred_df
        

    
    def save_model(self, path):
        # We need to save the user_encoder and news_encoder
        torch.save(self.user_encoder.state_dict(), path + "/user_encoder.pth")
        torch.save(self.news_encoder.state_dict(), path + "/news_encoder.pth")
    
    
    def load_checkpoint(self, path):
        self.user_encoder.load_state_dict(torch.load(path + "/user_encoder.pth"))
        self.news_encoder.load_state_dict(torch.load(path + "/news_encoder.pth"))