import numpy as np

import torch
import torch.nn as nn

from model import Model



class ANN(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        
    def build_model(self):


        self.layers = [[nn.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias=self.config.bias),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout_p)]]
        if self.config.use_batchnorm: self.layers[0].insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))


        for i in range(self.config.n_hidden_layers-1):
            self.layer = [  nn.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias),
                            nn.ReLU(),
                            nn.Dropout(self.config.dropout_p)]
            if self.config.use_batchnorm: self.layer.insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))
            self.layers.append(self.layer)


        self.layers.append([nn.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=self.config.bias)])

        self.model = [l for block in self.layers for l in block]
        self.model = nn.Sequential(*self.model)

        print(self.model)


    def init_model(self):
        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers):
                print(self.layers[i][0])
                torch.nn.init.kaiming_uniform_(self.layers[i][0].weight, nonlinearity='relu')



    def forward(self, x):
        out = []
        for t in range(x.size()[0]):
            out.append(self.model(x[t]).unsqueeze(0))

        return torch.concat(out)