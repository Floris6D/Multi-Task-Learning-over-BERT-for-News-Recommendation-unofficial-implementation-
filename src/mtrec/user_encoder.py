import torch
import math

from torch.nn.functional import softmax, tanh


class UserEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, device:str = "cpu"):
        super(UserEncoder, self).__init__()
        #parameters for calculating attention
        self.device = device
        self.W = torch.nn.Parameter(torch.empty(hidden_size, embedding_dim))
        self.q = torch.nn.Parameter(torch.empty(1,hidden_size,1))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.q, a=math.sqrt(5))
    
    def calc_att(self, R):
        tanhWR = tanh(torch.einsum("ij,bjk->bik", self.W, R.transpose(1, 2)))
        return softmax(torch.sum(self.q * tanhWR, axis = 1).squeeze(), dim = 1)
    
    
    def forward(self, R_h): #new iteration where we feed R_h directly from news encoder
        att = self.calc_att(R_h) #batch_size * h
        user_embeddings = torch.einsum("bi,bik->bk", att, R_h) #batch_size * token_size
        return user_embeddings.squeeze()