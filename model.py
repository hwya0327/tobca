import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

class EarlyStopping:
    def __init__(self, patience, loss_names, verbose=False, delta=0, path='Earlystopping.pt'):
        self.patience = patience
        self.loss_names = loss_names
        self.verbose = verbose
        self.counters = 0
        self.best_scores = {loss_name: None for loss_name in loss_names}
        self.early_stop = False
        self.loss_min = {loss_name: np.Inf for loss_name in loss_names}
        self.loss_min_past = {loss_name: np.Inf for loss_name in loss_names}
        self.delta = delta
        self.path = path

    def __call__(self, losses, model, epoch, num_epochs, avg_train_loss, avg_train_main_loss, avg_train_sub_loss,avg_valid_loss, avg_valid_main_loss,avg_valid_sub_loss):
        all_losses_improved = all(loss_value < self.loss_min[loss_name] for loss_name, loss_value in losses.items())
        
        if all_losses_improved:
            for loss_name, loss_value in losses.items():
                if self.best_scores[loss_name] is None:
                    self.best_scores[loss_name] = loss_value
                    self.loss_min[loss_name] = loss_value
                    self.save_checkpoint(model) 
                elif loss_value < self.best_scores[loss_name]:
                    self.best_scores[loss_name] = loss_value
                    self.loss_min_past[loss_name] = self.loss_min[loss_name]
                    self.loss_min[loss_name] = loss_value
                    self.save_checkpoint(model)
                    self.counters = 0
            if self.verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - main Loss: {avg_train_main_loss:.4f} - sub Loss: {avg_train_sub_loss:.4f} - Valid Loss: {avg_valid_loss:.4f} - main Loss: {avg_valid_main_loss:.4f} - sub Loss: {avg_valid_sub_loss:.4f}")
        else:
            self.counters += 1
            if self.counters >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        torch.cuda.empty_cache()

class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):

        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad  = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)

        merged_grad = torch.zeros_like(grads[0])

        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                        for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                        for g in pc_grad]).sum(dim=0)
            
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)

        return merged_grad.clone().detach()

    def _set_grad(self, grads):

        '''
        set the modified gradients to the network
        '''
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1

        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []

        self._optim.zero_grad(set_to_none=True)
        objectives[0].backward(retain_graph=True)
        grad, shape, has_grad = self._retrieve_grad()
        grads.append(self._flatten_grad(grad, shape))
        has_grads.append(self._flatten_grad(has_grad, shape))
        shapes.append(shape)
    
        self._optim.zero_grad(set_to_none=True)
        objectives[1].backward(retain_graph=False)
        grad, shape, has_grad = self._retrieve_grad()
        grads.append(self._flatten_grad(grad, shape))
        has_grads.append(self._flatten_grad(has_grad, shape))
        shapes.append(shape)

        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):

        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(Variable(grads[idx:(idx + length)].view(shape).clone(), requires_grad = False))
            idx += length
        
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):

        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad.clone().detach()

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue

                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
    
        return grad, shape, has_grad
    
class EmbeddingModule(nn.Module):
    def __init__(self, embedding_size, embedding_num_layers, activation_type, LN, seq_len, numeric_input_size, presence_input_size, CB):
        super(EmbeddingModule, self).__init__()

        self.CB = CB
        self.LN = LN
        self.num_layers = embedding_num_layers

        activation_functions = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'GELU': nn.GELU(),
        }

        activation_function = activation_functions.get(activation_type, nn.ReLU()) 

        self.embedding_numeric = nn.ModuleList([
            nn.Sequential(
                nn.Linear(numeric_input_size if i == 0 else embedding_size, embedding_size),
                activation_function,
            ) for i in range(self.num_layers)
        ])

        self.embedding_presence = nn.ModuleList([
            nn.Sequential(
                nn.Linear(presence_input_size if i == 0 else embedding_size, embedding_size),
                activation_function,
            ) for i in range(self.num_layers)
        ])
        
        self.LN_layer = nn.LayerNorm(normalized_shape=(seq_len, 2 * embedding_size if CB else embedding_size), eps=1e-05)

    def forward(self, x_numeric, x_presence):
        
        for i in range(len(self.embedding_numeric)):
            x_numeric = self.embedding_numeric[i](x_numeric)
            x_presence = self.embedding_presence[i](x_presence)

        if self.CB :
            embedded = torch.cat([x_numeric, x_presence], dim=2)

        else :
            embedded = x_numeric + x_presence

        if self.LN:
            embedded = self.LN_layer(embedded)

        return embedded
    
class RecurrentModule(nn.Module):
    def __init__(self, hidden_size, embedding_size, recurrent_num_layers, recurrent_type, highway_network,CB):
        super(RecurrentModule, self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = recurrent_num_layers
        self.highway_network = highway_network

        recurrent_modules = {
            'RNN': nn.RNN,
            'GRU': nn.GRU,
            'LSTM': nn.LSTM
        }

        recurrent_module = recurrent_modules.get(recurrent_type, nn.RNN)

        input_size =  2 * self.embedding_size if CB else self.embedding_size

        self.recurrent_layers = nn.ModuleList([
            recurrent_module(input_size if i == 0 else hidden_size, hidden_size, 1, batch_first=True) for i in range(self.num_layers)
        ])

        if self.highway_network:
            # Create a list to hold the highway layers dynamically (except the final layer)
            self.highway_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Linear(hidden_size, hidden_size)
                )
                for _ in range(self.num_layers - 1)
            ])

        self.fc_main = nn.Linear(hidden_size, 8)
        self.fc_sub = nn.Linear(hidden_size, 4)

    def forward(self, x):
        
        out = x

        if (self.num_layers != 1):
            if self.highway_network:
                for i in range(self.num_layers - 1):
                    out, _ = self.recurrent_layers[i](out)

                    # Apply the highway network
                    h = out
                    t = torch.sigmoid(self.highway_layers[i][0](h))
                    transformed = torch.relu(self.highway_layers[i][1](h))
                    out = t * transformed + (1 - t) * h
            else: 
                for i in range(self.num_layers - 1):
                    out, _ = self.recurrent_layers[i](out)
        else:
            out, _ = self.recurrent_layers[-1](out)

        out_main = self.fc_main(out)
        out_sub = self.fc_sub(out)

        out_main = out_main.transpose(1,2).contiguous()
        out_sub = out_sub.transpose(1,2).contiguous()

        return out_main, out_sub

class AKIPredictionModel(nn.Module):
    def __init__(self, hidden_size, embedding_size, recurrent_num_layers, embedding_num_layers, activation_type, recurrent_type, seq_len, LN, highway_network, numeric_input_size, presence_input_size, CB):
        super(AKIPredictionModel, self).__init__()

        self.embedding_module = EmbeddingModule(embedding_size, embedding_num_layers, activation_type, LN, seq_len, numeric_input_size, presence_input_size, CB)
        self.recurrent_module = RecurrentModule(hidden_size, embedding_size, recurrent_num_layers, recurrent_type, highway_network, CB)

    def forward(self, x_numeric, x_presence):

        embedded = self.embedding_module(x_numeric, x_presence)
        out_main, out_sub = self.recurrent_module(embedded)

        return out_main, out_sub
    
class CustomBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(CustomBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        eps = 1e-12
        input_clamped = torch.clamp(input, min=eps, max= (1 - eps))
        loss = - (self.pos_weight * target * torch.log(input_clamped) + (1 - target) * torch.log(1 - input_clamped))
        
        return loss.mean()