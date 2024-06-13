import torch
from torch.nn.functional import softmax, tanh
from transformers import BertModel, BertTokenizer

class UserEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UserEncoder, self).__init__()
        self.q_u = torch.nn.Parameter(torch.randn(hidden_dim))
        self.W_u = torch.nn.Parameter(torch.randn(hidden_dim, input_dim))
    
    def forward(self, R):
        user_representation = torch.zeros_like(R[0])
        for r_i in R:      
            Wh = tanh(r_i @ self.W_u.T)
            a_i = softmax(Wh @ self.q_u, dim=0)  
            user_representation += r_i*a_i.unsqueeze(-1)    
        
        return user_representation

class NewsEncoder(torch.nn.Module):
    def __init__(self):
        super(NewsEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')       

    def forward(self, inputs):
        # inputs = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device), output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        return last_hidden_states[0,0,:].cpu()
