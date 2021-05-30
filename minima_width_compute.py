import torch
import torch.nn as nn

import copy

class ComputeKeskarSharpness:
    def __init__(self, final_model, optimizer, criterion, trainloader, epsilon=1e-4, lr=0.001, max_steps=1000)
        self.net = final_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.epsilon = epsilon
        self.lr = lr 
        self.max_steps = max_steps

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        
    def compute_loss(self):
        loss = 0
        self.net.eval()
        train_loss = 0
        total = 0
        with torch.no_grad():
            for i, (inputs,targets) in enumerate(self.trainloader):
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets).item()
                
                train_loss += loss * targets.size(0)
                total += targets.size(0)

        return train_loss/total

    def compute_sharpness(self):
        stop = False
        num_steps = 0
        final_loss = 0  # l_1: 

        self.net.eval()
        # init loss  --> l_0
        init_loss = self.compute_loss()
            
        # min and max parameter values
        init_sd = copy.deepcopy(dict(self.net.state_dict()))
        max_sd = {k: self.epsilon*(torch.abs(v)+1.) for k,v in init_sd.items()}

        proj = lambda v,k: torch.max(init_sd[k]-max_sd[k], torch.min(v, init_sd[k]+max_sd[k]))
        
        while not stop:
            for i, (x, y) in enumerate(self.trainloader):
                self.net.train()
                self.optimizer.zero_grad()

                # maximize loss
                out = self.net(x)
                loss = -1 * self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                # project
                proj_sd = {k: proj(v, k) for k, v in self.net.state_dict().items()}
                self.net.load_state_dict(proj_sd)
            
                num_steps += 1
                if num_steps == self.max_steps:
                    final_loss = self.compute_loss()
                    stop = True
                    break
            
        # keskarify = lambda x: ((x-init_loss) * 100) /(1+init_loss)
        print("sharpness = ", ((final_loss - init_loss) * 100) / (1 + init_loss))   
        return ((final_loss - init_loss) * 100) / (1 + init_loss)
