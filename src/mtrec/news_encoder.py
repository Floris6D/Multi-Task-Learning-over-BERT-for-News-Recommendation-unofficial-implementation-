import torch
from transformers import BertModel
#TODO implement bert uncased from huggingface

class NewsEncoder(torch.nn.Module):
    def __init__(self, input_size, embedding_dim, num_classes, num_ner, bert):
        super(NewsEncoder, self).__init__()
        #TODO: add configuration for the number of layers and hidden size
        self.cat_net = self.initialize_network(input_size, num_classes)
        self.ner_net = self.initialize_network(input_size, num_ner)
        self.bert = bert

    def initialize_network(self, input_size, output_size, hidden_size=124, num_layers=2, act_fn = torch.nn.ReLU()):
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
        return self.cat_net(cls_tokens)

    def forward_ner(self, last_hidden_state):
        sentence_tokens = last_hidden_state[:, 1:-1, :] #all tokens but SEP and CLS
        output =  self.ner_net(sentence_tokens)
        return output


    def forward(self, tokens, mask = False):
        bs, n, ts = tokens.shape
        tokens = tokens.reshape(bs*n, ts)
        if mask:
            x = self.bert(tokens, mask)
        else:
            x = self.bert(tokens)
        last_hidden_state = x.last_hidden_state
        cat = self.forward_cat(last_hidden_state)
        ner = self.forward_ner(last_hidden_state).reshape(bs, n, -1)
        return last_hidden_state[:, 0 , :], cat, ner