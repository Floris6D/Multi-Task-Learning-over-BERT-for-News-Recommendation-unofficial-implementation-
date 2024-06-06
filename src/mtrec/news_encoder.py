import torch
from transformers import BertModel
#TODO implement bert uncased from huggingface

class NewsEncoder(torch.nn.Module):
    def __init__(self, tok_size, embedding_dim, num_classes, num_ner):
        super(NewsEncoder, self).__init__()
        #TODO: add configuration for the number of layers and hidden size
        self.cat_net = self.initialize_network(tok_size, num_classes)
        self.ner_net = ...
        self.bert = BertModel.from_pretrained('bert-base-uncased')

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

    def forward_cat(self, x):
        cls_embedding = ...
        return self.cat_net(cls_embedding)

    def forward_ner(self, x):
        title_embeddings = ...
        result = []
        for e in title_embeddings:
            result.append(self.ner_net(e))
        return result


    def forward(self, x):
        x = self.bert(**x)
        cat = self.cat_net(x)
        ner = self.ner_net(x)
        return x, cat, ner