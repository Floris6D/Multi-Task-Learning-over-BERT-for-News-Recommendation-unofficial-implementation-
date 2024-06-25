import torch
import math
from torch.nn.functional import softmax, tanh

class UserEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, device:str = "cpu"):
        super(UserEncoder, self).__init__()
        self.device = device
        # Parameters for attention layer
        self.W = torch.nn.Parameter(torch.empty(hidden_size, embedding_dim))
        self.q = torch.nn.Parameter(torch.empty(1,hidden_size,1))
        # Initialize the parameters
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.q, a=math.sqrt(5))
    
    def calc_att(self, R):
        """
        Calculates the attention weights for the given input tensor R.

        Args:
            R (torch.Tensor): Input tensor of shape (batch_size, history_size, embedding_dim).

        Returns:
            torch.Tensor: Attention weights of shape (batch_size, history_size).

        """
        # Make a mask of -inf for the padded values s.t. they get 0 weight after softmax
        M = torch.all(R==0, dim=2).int()  # batch_size * hist
        M_float = M.float()
        M_float[M == 1] = float('-inf')
        M_float.to(self.device)
        # Attention mechanism as described in paper
        tanhWR = tanh(torch.einsum("ij,bjk->bik", self.W, R.transpose(1, 2)))
        unnormalized_att = torch.sum(self.q * tanhWR, axis=1).squeeze() + M_float  # batch_size * h
        return softmax(unnormalized_att, dim=1)
    
    
    def forward(self, R_h):
        """
        Forward pass of the user encoder module.

        Args:
            R_h (torch.Tensor): Input tensor of shape (batch_size, hist, bert_embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, token_size).
        """
        attention = self.calc_att(R_h) #batch_size * hist * bert_embedding_size
        user_embeddings = torch.einsum("bi,bik->bk", attention, R_h) #batch_size * token_size
        return user_embeddings.squeeze()