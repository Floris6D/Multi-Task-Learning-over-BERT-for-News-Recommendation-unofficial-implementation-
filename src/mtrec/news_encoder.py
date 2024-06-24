import torch
from transformers import BertModel


class NewsEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, bert, cfg_cat, cfg_ner, extended_NER = False):
        super(NewsEncoder, self).__init__()
        if extended_NER:
            self.num_ner = cfg_ner["extended_output_size"]
        elif extended_NER is False:
            self.num_ner = cfg_ner["output_size"]
        
        self.cat_net = self.initialize_network(embedding_dim, cfg_cat, cfg_cat["output_size"])
        self.ner_net = self.initialize_network(embedding_dim, cfg_ner, self.num_ner)

        self.bert = bert

    def initialize_network(self, input_size, cfg, output_size, act_fn = torch.nn.ReLU()):
        num_layers, hidden_size = cfg["num_layers"], cfg["hidden_size"]
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        elif num_layers == 1:
            return torch.nn.Linear(input_size, output_size)
        layers = []
        for _ in range(num_layers-1):
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(act_fn)
            input_size = hidden_size
        layers.append(torch.nn.Linear(hidden_size, output_size))
        return torch.nn.Sequential(*layers)

    
    def forward_cat(self, last_hidden_state, mask):
        
        # Last hidden state has dimensions (batch_size, max_inview_articles, max_title_length, hidden_size)
        cls_tokens = last_hidden_state[:, :, 0, :]
        bs, n, _ = cls_tokens.shape
        cls_tokens = cls_tokens.reshape(bs*n, -1) # Reshape to (batch_size*max_inview_articles, hidden_size)
        
        # Again remove all rows with only zeros in the mask
        subset_idx = mask.sum(dim=1) > 0
        used_tokens = cls_tokens[subset_idx]
        
        # Get the category predictions logits
        logits = self.cat_net(used_tokens)
        
        # Now reshape the logits back to the original shape fill with nas when no data of batch size inview articles using the reversed subset_idx
        final_logits = torch.nan*torch.ones(bs*n, logits.shape[1])
        final_logits[subset_idx] = logits
        logits = final_logits.reshape(bs, n, -1)
        
        return torch.nn.functional.softmax(logits, dim=2)

    
    def forward_ner(self, last_hidden_state, mask):
        '''
        Kijk naar de code die weg gecomment is als je het wilt doen zoals bij forward_cat, ik raakte alleen in de war
        '''
        # Last hidden state has dimensions (batch_size, max_inview_articles, max_title_length, hidden_size)
        sentence_tokens = last_hidden_state[:, :, :, :]
        #mask = mask[:, 1:].reshape(sentence_tokens.shape[0], sentence_tokens.shape[1], -1) #all tokens but CLS
        mask_ner = mask.clone()
        mask_ner[:, 0] = 0 #CLS token is not used for NER
        last_idx = torch.argmax(torch.fliplr(mask_ner), dim=1)
        cols = mask_ner.shape[1] - 1 - last_idx
        mask_ner[torch.arange(mask_ner.shape[0]), cols] = 0 # Set the last token to 0, now cls and sep tokens are 0
        mask_ner = mask_ner.reshape(sentence_tokens.shape[0], sentence_tokens.shape[1], -1) #Reshape
        
        logits = torch.nan*torch.ones(sentence_tokens.shape[0], sentence_tokens.shape[1], sentence_tokens.shape[2], self.num_ner)
        
        for i in range(sentence_tokens.shape[0]): # Loop over batches
            for j in range(sentence_tokens.shape[1]): # Loop over inview articles
                if mask_ner[i, j, :].sum() == 0:
                    continue
                for k in range(sentence_tokens.shape[2]):
                    if mask_ner[i, j, k] == 0:
                        continue
                    logits[i, j, k, :] = self.ner_net(sentence_tokens[i, j, k, :])
        
        ######### Commented code below was without for loops I got confused with the reshaping and masking so for loops
        output = torch.nn.functional.softmax(logits, dim=3)
        return output, mask_ner
        
        # bs, n, n_tokens, _ = sentence_tokens.shape
        # sentence_tokens = sentence_tokens.reshape(bs*n, n_tokens, -1) # Reshape to (batch_size*max_inview_articles*max_title_length, hidden_size)
        
        # # Remove all rows with only zeros in the mask
        # subset_idx = mask.sum(dim=1) > 0
        # used_tokens = sentence_tokens[subset_idx]
        
        # # Again reshape for every token in the sentence
        # used_tokens = used_tokens.reshape(used_tokens.shape[0] * used_tokens.shape[1], -1) # Reshape to (batch_size*max_inview_articles*max_title_length, hidden_size)
        # used_mask = mask[subset_idx]
        # used_mask = used_mask.reshape(used_mask.shape[0] * used_mask.shape[1])
        # subset_idx2 = used_mask > 0
        
        # final_tokens = used_tokens[subset_idx2]
        
        # logits =  self.ner_net(final_tokens) 
        
        # # Now reverse the reshaping process
        # logits_reshape = torch.nan * torch.ones(bs*n, n_tokens, logits.shape[1])
        # logits_reshape[subset_idx2] = logits
        
        # logits_reshape2 = torch.nan * torch.ones(bs, n, n_tokens, logits.shape[1])
        # logits_reshape2[subset_idx] = logits_reshape
        
        # output = torch.nn.functional.softmax(logits_reshape2, dim=3)
        # return output

    
    def forward(self, tokens, mask = False, validation = False):
        # Reshape the tokens and mask
        bs, n, ts = tokens.shape # batch size, max inview articles, max_title length
        tokens = tokens.reshape(bs*n, ts)
        mask = mask.reshape(bs*n, ts)
        
        # Next remove all rows with only zeros in the mask
        subset_idx = mask.sum(dim=1) > 0
        tokens = tokens[subset_idx]
        bert_mask = mask[subset_idx]
        
        x = self.bert(tokens, bert_mask)
        print(f"x device: {x.get_device()}") 
        last_hidden_state = x.last_hidden_state
        news_embeddings = last_hidden_state[:, 0 , :] #CLS token at index 0
        
        # Restructure the news embeddings in the original shape and using the reversed subset_idx if subset_idx was false, news embeddings will be zero
        final_news_embeddings = torch.zeros(bs*n, news_embeddings.shape[1])
        final_news_embeddings[subset_idx] = news_embeddings
        final_news_embeddings = final_news_embeddings.reshape(bs, n, -1)
        
        if validation:
            return final_news_embeddings, None, None
        
        # Also restructure the last_hidden_state and use the reversed subset_idx
        final_last_hidden_state = torch.zeros(bs*n, last_hidden_state.shape[1], last_hidden_state.shape[2])
        final_last_hidden_state[subset_idx] = last_hidden_state
        final_last_hidden_state = final_last_hidden_state.reshape(bs, n, ts, -1) # Batch size, max inview articles, max title length, hidden size (BERT)
        
        # Get the category and ner predictions # TODO: JE Check if this deals correctly with padding
        cat = self.forward_cat(final_last_hidden_state, mask) #when there were no articles in the inview articles, the output will be nan
        ner, mask_ner = self.forward_ner(final_last_hidden_state, mask)      
        return final_news_embeddings, cat, ner, mask_ner