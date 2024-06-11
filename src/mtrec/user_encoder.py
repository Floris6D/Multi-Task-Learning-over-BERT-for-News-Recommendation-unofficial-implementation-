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
        self.q = torch.nn.Parameter(torch.empty(embedding_dim, 1))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.q, a=math.sqrt(5))



    def calc_att(self, R):
        tanhWR = tanh(torch.einsum("bij,bjk->bik", self.W, R.transpose(1, 2)))
        return softmax(torch.einsum("i,bik-> bk", self.q. tanhWR ), axis = 1)
        

    def forward(self, tokens_h, mask_h):
        """
        Forward pass of the user encoder module.
        
        Args:
            R_h (torch.Tensor) : Batch_size * input_size Input tensor representing the user's interaction history.
            
        Returns:
            torch.Tensor: User embedding tensor.
        """

        bs, h, ts = tokens_h.shape
        tokens_h = tokens_h.reshape(bs * h, ts)
        R_h = self.bert(tokens_h, mask_h).last_hidden_state[:, 0, :]
        R_h = R_h.reshape(bs, h, -1) #batch_size * h * token_size
        att = self.calc_att(R_h) #batch_size * h
        user_embeddings = torch.einsum("bi,bik->bk", att, R_h) #batch_size * token_size
        return user_embeddings.squeeze()