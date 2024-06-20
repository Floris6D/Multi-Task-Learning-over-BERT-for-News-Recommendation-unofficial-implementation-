import torch
from transformers import BertModel


class NewsEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, bert, cfg_cat, cfg_ner):
        super(NewsEncoder, self).__init__()
        self.cat_net = self.initialize_network(embedding_dim, cfg_cat)
        self.ner_net = self.initialize_network(embedding_dim, cfg_ner)
        self.num_ner = cfg_ner["output_size"]
        self.bert = bert

    def initialize_network(self, input_size, cfg, act_fn = torch.nn.ReLU()):
        num_layers, hidden_size, output_size = cfg["num_layers"], cfg["hidden_size"], cfg["output_size"]
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

    
    def forward_cat(self, last_hidden_state):
        cls_tokens = last_hidden_state[:, 0, :]
        logits = self.cat_net(cls_tokens)
        return torch.nn.functional.softmax(logits, dim=1)

    
    def forward_ner(self, last_hidden_state):
        sentence_tokens = last_hidden_state[:, 1:-1, :] #all tokens but SEP and CLS #TODO: check how this works with padding
        logits =  self.ner_net(sentence_tokens) 
        output = torch.nn.functional.softmax(logits, dim=2)
        return output

    
    def forward(self, tokens, mask = False, validation = False):
        # Reshape the tokens and mask
        bs, n, ts = tokens.shape # batch size, max inview articles, max_title length
        tokens = tokens.reshape(bs*n, ts)
        mask = mask.reshape(bs*n, ts)
        
        # Next remove all rows with only zeros in the mask
        subset_idx = mask.sum(dim=1) > 0
        tokens = tokens[subset_idx]
        mask = mask[subset_idx]
        
        x = self.bert(tokens, mask)
        last_hidden_state = x.last_hidden_state
        news_embeddings = last_hidden_state[:, 0 , :] #CLS token at index 0
        
        # Restructure the news embeddings in the original shape and using the reversed subset_idx if subset_idx was false, news embeddings will be zero
        final_news_embeddings = torch.zeros(bs*n, news_embeddings.shape[1])
        final_news_embeddings[subset_idx] = news_embeddings
        final_news_embeddings = final_news_embeddings.reshape(bs, n, -1)
        
        if validation:
            return final_news_embeddings
        
        # Also restructure the last_hidden_state and use the reversed subset_idx
        final_last_hidden_state = torch.zeros(bs*n, last_hidden_state.shape[1], last_hidden_state.shape[2])
        final_last_hidden_state[subset_idx] = last_hidden_state
        final_last_hidden_state = final_last_hidden_state.reshape(bs, n, ts, -1) 
        
        # Get the category and ner predictions # TODO: JE Check if this deals correctly with padding
        cat = self.forward_cat(final_last_hidden_state).reshape(bs, n, -1)
        ner = self.forward_ner(final_last_hidden_state).reshape(bs, n, -1, self.num_ner)        
        return final_news_embeddings, cat, ner