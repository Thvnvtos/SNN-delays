import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.build_model()
        self.init_model()

    def forward(self, x):
        return self.model(x)


    def optimizers(self):
        # returns a list of optimizers
        return [optim.Adam(self.model.parameters(), lr = self.config.lr, betas=(0.9,0.999))]


    def schedulers(self):
        pass

    def calc_loss(self, output, y):
        # probably better to add it in init, or in general do it one time only
        log_softmax_fn = nn.LogSoftmax(dim=1) 
        loss_fn = nn.NLLLoss()

        m = torch.mean(output, 0)
        log_p_y = log_softmax_fn(m) 
        
        return loss_fn(log_p_y, y)

    def calc_metric(self, output, y):
        # mean accuracy over batch
        m = torch.mean(output, 0)
        return np.mean((y==torch.max(m,1)[1]).detach().cpu().numpy())
    

    def train_model(self, train_loader, valid_loader, device):

        optimizers = self.optimizers()
        schedulers = self.schedulers()

        loss_epochs = {'train':[], 'valid':[]}
        metric_epochs = {'train':[], 'valid':[]}
        for epoch in range(self.config.epochs):
            self.model.train()
            #last element in the tuple corresponds to the collate_fn return
            loss_batch, metric_batch = [], []
            for i, (x, y, _) in enumerate(train_loader):
                # x for shd and ssc is: (batch, neurons, time)
                x = x.permute(1,0,2).float().to(device)
                y = y.to(device)

                for opt in optimizers: opt.zero_grad()
                
                output = self.forward(x)
                loss = self.calc_loss(output, y)

                loss.backward()
                for opt in optimizers: opt.step()

                metric = self.calc_metric(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)
            
            loss_epochs['train'].append(np.mean(loss_batch))
            metric_epochs['train'].append(np.mean(metric_batch))


            self.model.eval()
            with torch.no_grad():
                loss_batch, metric_batch = [], []
                for i, (x, y, _) in enumerate(valid_loader):
                    x = x.permute(1,0,2).float().to(device)
                    y = y.to(device)

                    output = self.forward(x)
                
                    loss = self.calc_loss(output, y)
                    metric = self.calc_metric(output, y)

                    loss_batch.append(loss.detach().cpu().item())
                    metric_batch.append(metric)

                loss_epochs['valid'].append(np.mean(loss_batch))
                metric_epochs['valid'].append(np.mean(metric_batch))
                
        
            print(f"=====> Epoch {epoch} : \nLoss Train = {loss_epochs['train'][-1]:.3f}  |  Best Acc Train = {100*max(metric_epochs['train']):.2f}% \nLoss Test = {loss_epochs['valid'][-1]:.3f}  |  Best Acc Test = {100*max(metric_epochs['valid']):.2f}%")

        



class ANN(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        
    def build_model(self):
        
        self.model = nn.Sequential(
            nn.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias=False),
            nn.BatchNorm1d(self.config.n_hidden_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=False),
        )


    def init_model(self):
        torch.nn.init.normal_(self.model[0].weight, mean=0.0, std = 1/np.sqrt(self.config.n_inputs))
        torch.nn.init.normal_(self.model[4].weight, mean=0.0, std = 1/np.sqrt(self.config.n_hidden_neurons))


    def forward(self, x):
        out = []
        for t in range(x.size()[0]):
            out.append(self.model(x[t]).unsqueeze(0))

        return torch.concat(out)