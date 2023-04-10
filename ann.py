import numpy as np

import torch
import torch.nn as nn

from model import Model
from utils import set_seed


class ANN(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        
    def build_model(self):


        self.blocks = [[nn.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias=self.config.bias),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout_p)]]
        if self.config.use_batchnorm: self.blocks[0].insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))


        for i in range(self.config.n_hidden_layers-1):
            self.block = [  nn.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias),
                            nn.ReLU(),
                            nn.Dropout(self.config.dropout_p)]
            if self.config.use_batchnorm: self.block.insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))
            self.blocks.append(self.block)


        self.blocks.append([nn.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=self.config.bias)])

        self.model = [l for block in self.blocks for l in block]
        self.model = nn.Sequential(*self.model)

        print(self.model)


    def init_model(self):
        set_seed(self.config.seed)

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                torch.nn.init.kaiming_uniform_(self.blocks[i][0].weight, nonlinearity='relu')

    def reset_model(self):
        pass

    def decrease_sig(self, epoch):
        pass

    def forward(self, x):
        out = []
        for t in range(x.size()[0]):
            out.append(self.model(x[t]).unsqueeze(0))

        return torch.concat(out)