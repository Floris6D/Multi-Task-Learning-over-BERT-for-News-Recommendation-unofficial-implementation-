import torch
class PCGrad():
    def __init__(self, optimizer,  aux_scaler=0.3):
        self.optimizer = optimizer
        self.aux_scaler = aux_scaler
    
    def pc_backward(self, main_loss, aux_losses):
        # Get main gradients
        self.optimizer.zero_grad()
        main_loss.backward(retain_graph=True)
        self.main_grads = torch.tensor([param.grad.clone() for param in self.parameters()])

        # Get auxiliary gradients
        self.optimizer.zero_grad()
        aux_grads_list = []
        for aux_loss in aux_losses:
            aux_loss.backward(retain_graph=True)
            aux_grads = [param.grad.clone() for param in self.parameters()]
            aux_grads_list.append(aux_grads)
            self.zero_grad()
        # Aggregate auxiliary gradients 
        aux_grads_tensor = torch.stack(aux_grads_list)
        self.aux_grads = aux_grads_tensor.sum(dim=0)
        # Check and resolve conflict
        if self.is_conflict():
            self.resolve_conflict()
        # Combine main and aux gradients
        self.combine_grads()
        
    def is_conflict(self): 
        return torch.dot(self.main_grads, self.aux_grads) < 0
    
    def resolve_conflict(self):
        dot_ma = torch.dot(self.main_grads, self.aux_grads)
        mg, ag = self.main_grads, self.aux_grads
        new_aux_grad = ag - (dot_ma / torch.dot(mg, mg)) * mg
        new_main_grad = mg - (dot_ma / torch.dot(ag, ag)) * ag
        self.main_grad = new_main_grad
        self.aux_grad = new_aux_grad

    def combine_grads(self):
        combined_grads = self.main_grads + self.aux_scaler * self.aux_grads
        for i, param in enumerate(self.optimizer.parameters()):
                # Assign the combined gradient to the parameter's grad
                param.grad = combined_grads[i]
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self): #filler function, not necesarry
        self.optimizer.zero_grad()