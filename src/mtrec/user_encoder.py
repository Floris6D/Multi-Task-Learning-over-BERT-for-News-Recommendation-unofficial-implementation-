import torch
import math

from torch.nn.functional import softmax, tanh

class UserEncoder(torch.nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(UserEncoder, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim

        #parameters for calculating attention
        self.W = torch.nn.Parameter(torch.empty(embedding_dim, input_size))
        self.q = torch.nn.Parameter(torch.empty(embedding_dim, 1))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.q, a=math.sqrt(5))

    def calc_att(self, R):
        return softmax(self.q * tanh(self.W @ R.T), axis=1)
    
    def forward(self, R_h):
        """
        Forward pass of the user encoder module.
        
        Args:
            R_h (torch.Tensor) : Batch_size * input_size Input tensor representing the user's interaction history.
            
        Returns:
            torch.Tensor: User embedding tensor.
        """
        A = self.calc_att(R_h)
        return A * R_h