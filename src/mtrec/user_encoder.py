import torch
import math

from torch.nn.functional import softmax, tanh

class UserEncoder(torch.nn.Module):
    def __init__(self, input_size, embedding_dim, bert):
        super(UserEncoder, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.bert = bert

        #parameters for calculating attention
        self.W = torch.nn.Parameter(torch.empty(embedding_dim, input_size))
        self.q = torch.nn.Parameter(torch.empty(1,embedding_dim,1))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.q, a=math.sqrt(5))

    def calc_att(self, R):
        tanhWR = tanh(torch.einsum("ij,bjk->bik", self.W, R.transpose(1, 2)))
        return softmax(torch.sum(self.q * tanhWR, axis = 1).squeeze(), dim = 1)
        
    def forward(self, tokens_h, mask_h = False):
        bs, h, ts = tokens_h.shape
        tokens_h = tokens_h.reshape(bs * h, ts)
        mask_h = mask_h.reshape(bs * h, ts)
        R_h = self.bert(tokens_h, mask_h).last_hidden_state[:, 0, :]
        R_h = R_h.reshape(bs, h, -1) #batch_size * h * token_size
        att = self.calc_att(R_h) #batch_size * h
        user_embeddings = torch.einsum("bi,bik->bk", att, R_h) #batch_size * token_size
        return user_embeddings.squeeze()