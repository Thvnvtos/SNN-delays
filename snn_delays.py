import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional

from DCLG.Conv import GDcls1d

from model import Model



class SnnDelays(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
    
    # Try factoring this method
    # Check ThresholdDependent batchnorm (in spikingjelly)
    def build_model(self):

        # self.blocks = (n_layers, 0:weights+bn | 1: lif+dropout,  0,1)

        self.blocks = [[[GDcls1d(self.config.n_inputs, self.config.n_hidden_neurons, kernel_count=self.config.kernel_count, groups = 1, 
                                dilated_kernel_size = self.config.max_delay, bias=self.config.bias, version=self.config.DCLSversion)],
                       
                        [layer.Dropout(self.config.dropout_p, step_mode='m')]]]
        
        if self.config.use_batchnorm: self.blocks[0][0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
        if self.config.spiking_neuron_type == 'lif': 
            self.blocks[0][1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))



        for i in range(self.config.n_hidden_layers-1):
            self.block = [[GDcls1d(self.config.n_hidden_neurons, self.config.n_hidden_neurons, kernel_count=self.config.kernel_count, groups = 1, 
                                dilated_kernel_size = self.config.max_delay, bias=self.config.bias, version=self.config.DCLSversion)],
                       
                            [layer.Dropout(self.config.dropout_p, step_mode='m')]]
        
            if self.config.use_batchnorm: self.block[0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
            if self.config.spiking_neuron_type == 'lif': 
                self.block[1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            
            self.blocks.append(self.block)


        self.final_block = [[GDcls1d(self.config.n_hidden_neurons, self.config.n_outputs, kernel_count=self.config.kernel_count, groups = 1, 
                                     dilated_kernel_size = self.config.max_delay, bias=self.config.bias, version=self.config.DCLSversion)]]
        if self.config.spiking_neuron_type == 'lif':
            self.final_block.append([neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])


        self.blocks.append(self.final_block)

        self.model = [l for block in self.blocks for sub_block in block for l in sub_block]
        self.model = nn.Sequential(*self.model)
        #print(self.model)

        self.weights, self.positions = [], []
        for name, param in self.named_parameters():
            if 'P' in name:     self.positions.append(param)
            else:               self.weights.append(param)


    def init_model(self):
        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.weights ?
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].weight, nonlinearity='relu')

        if self.config.init_pos_method == 'uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.positions?
                torch.nn.init.uniform_(self.blocks[i][0][0].P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0][0].clamp_parameters()

                if self.config.model_type == 'snn_delays_lr0':
                    self.blocks[i][0][0].P.requires_grad = False

        for i in range(self.config.n_hidden_layers+1):
            # can you replace with self.positions?
            torch.nn.init.constant_(self.blocks[i][0][0].SIG, self.config.sigInit)
            self.blocks[i][0][0].SIG.requires_grad = False



    def reset_model(self):
        functional.reset_net(self)

        # We use clamp_parameters of the GDcls1d modules
        for block in self.blocks:
            block[0][0].clamp_parameters()



    def forward(self, x):
        
        for block_id in range(self.config.n_hidden_layers):
            # x is permuted: (time, batch, neurons) => (batch, neurons, time)  in order to be processed by the convolution
            x = x.permute(1,2,0)
            x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)  # we use padding for the delays kernel

            # we use convolution of delay kernels
            x = self.blocks[block_id][0][0](x)

            # We permute again: (batch, neurons, time) => (time, batch, neurons) in order to be processed by batchnorm or Lif
            x = x.permute(2,0,1)

            if self.config.use_batchnorm:
                # we use x.unsqueeze(3) to respect the expected shape to batchnorm which is (time, batch, channels, length)
                # we do batch norm on the channels since length is the time dimension
                # we use squeeze to get rid of the channels dimension 
                x = self.blocks[block_id][0][1](x.unsqueeze(3)).squeeze()
            
            
            # we use our spiking neuron filter
            spikes = self.blocks[block_id][1][0](x)
            # we use dropout on generated spikes tensor
            x = self.blocks[block_id][1][1](spikes)
            
            # x is back to shape (time, batch, neurons)
        
        # Finally, we apply same transforms for the output layer
        x = x.permute(1,2,0)
        x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)
        
        # Apply final layer
        out = self.blocks[-1][0][0](x)

        # permute out: (batch, neurons, time) => (time, batch, neurons)  For final spiking neuron filter
        out = out.permute(2,0,1)
        out = self.blocks[-1][1][0](out)

        if self.config.loss != 'spike_count':
            out = self.blocks[-1][1][0].v_seq

        return out