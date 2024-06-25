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
        # Get all empty news embeddings
        M = torch.all(R==0, dim =2).int() #batch_size * hist
        M_float = M.float()
        M_float[M == 1] = float('-inf')
        M_float.to(self.device)

        tanhWR = tanh(torch.einsum("ij,bjk->bik", self.W, R.transpose(1, 2)))
        unnormalized_att = torch.sum(self.q * tanhWR, axis = 1).squeeze() + M_float #batch_size * h
        return softmax(unnormalized_att, dim = 1)
    
    
    def forward(self, R_h): #new iteration where we feed R_h directly from news encoder
        att = self.calc_att(R_h) #batch_size * hist * bert_embedding_size
        user_embeddings = torch.einsum("bi,bik->bk", att, R_h) #batch_size * token_size
        return user_embeddings.squeeze()