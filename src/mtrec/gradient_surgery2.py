import torch
class PCGrad():
    def __init__(self, optimizer,  lamb=0.3):
        self.optimizer, self = optimizer
        self.lamb = lamb
        return
    
    def backward(self, main_loss, aux_losses):
        # Get main gradients
        self.optimizer.zero_grad()
        main_loss.backward(retain_graph=True)
        main_grads = [param.grad.clone() for param in self.parameters()]

        # Get auxiliary gradients
        self.optimizer.zero_grad()
        aux_grads_list = []
        for aux_loss in aux_losses:
            aux_loss.backward(retain_graph=True)
            aux_grads = [param.grad.clone() for param in self.parameters()]
            aux_grads_list.append(aux_grads)
            self.zero_grad()
        aux_grads_tensor = torch.stack(aux_grads_list)
        aux_grads = ...

        for i, param in enumerate(self.parameters()):
            combined_grad = main_grads[i]
            for aux_grads in aux_grads_list:
                combined_grad += aux_grads[i]
            # Assign the combined gradient to the parameter's grad
            param.grad = combined_grad