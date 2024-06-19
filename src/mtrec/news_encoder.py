import torch
from transformers import BertModel

class NewsEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, bert, cfg_cat, cfg_ner):
        super(NewsEncoder, self).__init__()
        self.cat_net = self.initialize_network(embedding_dim, cfg_cat)
        self.ner_net = self.initialize_network(embedding_dim, cfg_ner)
        self.num_ner = num_ner
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
        bs, n, ts = tokens.shape
        tokens = tokens.reshape(bs*n, ts)
        mask = mask.reshape(bs*n, ts)
        x = self.bert(tokens, mask)
       
        last_hidden_state = x.last_hidden_state
        news_embeddings = last_hidden_state[:, 0 , :].reshape(bs, n, -1)
        if validation:
            return news_embeddings
        cat = self.forward_cat(last_hidden_state).reshape(bs, n, -1)
        ner = self.forward_ner(last_hidden_state).reshape(bs, n, -1, self.num_ner)        
          
        
        return news_embeddings, cat, ner