import torch

class NewsEncoder(torch.nn.Module):
    def __init__(self, tok_size, embedding_dim, num_classes):
        super(NewsEncoder, self).__init__()
        self.cat_net = self.initialize_network(tok_size, num_classes)
        self.ner_net = ...
        self.bert = ...

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
        
    def forward(self, x):
        x = self.bert(x)
        cat = self.cat_net(x)
        ner = self.ner_net(x)
        return x, cat, ner