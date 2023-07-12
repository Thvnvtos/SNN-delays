import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional

from DCLS.construct.modules import Dcls1d

from model import Model
from utils import set_seed


class SnnDelays(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
    
    # Try factoring this method
    # Check ThresholdDependent batchnorm (in spikingjelly)
    def build_model(self):

        ########################### Model Description :
        #
        #  self.blocks = (n_layers,  0:weights+bn  |  1: lif+dropout+(synapseFilter) ,  element in sub-block)
        #


        ################################################   First Layer    #######################################################

        self.blocks = [[[Dcls1d(self.config.n_inputs, self.config.n_hidden_neurons, kernel_count=self.config.kernel_count, groups = 1, 
                                dilated_kernel_size = self.config.max_delay, bias=self.config.bias, version=self.config.DCLSversion)],
                       
                        [layer.Dropout(self.config.dropout_p, step_mode='m')]]]
        
        if self.config.use_batchnorm: self.blocks[0][0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
        if self.config.spiking_neuron_type == 'lif': 
            self.blocks[0][1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))

        elif self.config.spiking_neuron_type == 'plif': 
            self.blocks[0][1].insert(0, neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
        
        elif self.config.spiking_neuron_type == 'heaviside': 
            self.blocks[0][1].insert(0, self.config.surrogate_function)


        if self.config.stateful_synapse:
            self.blocks[0][1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                         step_mode='m'))


        ################################################   Hidden Layers    #######################################################

        for i in range(self.config.n_hidden_layers-1):
            self.block = [[Dcls1d(self.config.n_hidden_neurons, self.config.n_hidden_neurons, kernel_count=self.config.kernel_count, groups = 1, 
                                dilated_kernel_size = self.config.max_delay, bias=self.config.bias, version=self.config.DCLSversion)],
                       
                            [layer.Dropout(self.config.dropout_p, step_mode='m')]]
        
            if self.config.use_batchnorm: self.block[0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
            if self.config.spiking_neuron_type == 'lif': 
                self.block[1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            elif self.config.spiking_neuron_type == 'plif': 
                self.block[1].insert(0, neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            
            elif self.config.spiking_neuron_type == 'heaviside': 
                self.block[1].insert(0, self.config.surrogate_function)
            
            if self.config.stateful_synapse:
                self.block[1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                             step_mode='m'))

            self.blocks.append(self.block)


        ################################################   Final Layer    #######################################################


        self.final_block = [[Dcls1d(self.config.n_hidden_neurons, self.config.n_outputs, kernel_count=self.config.kernel_count, groups = 1, 
                                     dilated_kernel_size = self.config.max_delay, bias=self.config.bias, version=self.config.DCLSversion)]]
        if self.config.spiking_neuron_type == 'lif':
            self.final_block.append([neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])
        elif self.config.spiking_neuron_type == 'plif': 
            self.final_block.append([neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])



        self.blocks.append(self.final_block)

        self.model = [l for block in self.blocks for sub_block in block for l in sub_block]
        self.model = nn.Sequential(*self.model)
        #print(self.model)

        self.positions = []
        self.weights = []
        self.weights_bn = []
        self.weights_plif = []
        for m in self.model.modules():
            if isinstance(m, Dcls1d):
                self.positions.append(m.P)
                self.weights.append(m.weight)
                if self.config.bias:
                    self.weights_bn.append(m.bias)
            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)



    def init_model(self):

        set_seed(self.config.seed)
        self.mask = []

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.weights ?
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].weight, nonlinearity='relu')
                
                if self.config.sparsity_p > 0:
                    with torch.no_grad():
                        self.mask.append(torch.rand(self.blocks[i][0][0].weight.size()).to(self.blocks[i][0][0].weight.device))
                        self.mask[i][self.mask[i]>self.config.sparsity_p]=1
                        self.mask[i][self.mask[i]<=self.config.sparsity_p]=0
                        #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                        self.blocks[i][0][0].weight *= self.mask[i]


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






    def reset_model(self, train=True):
        functional.reset_net(self)

        for i in range(self.config.n_hidden_layers+1):                
            if self.config.sparsity_p > 0:
                with torch.no_grad():
                    self.mask[i] = self.mask[i].to(self.blocks[i][0][0].weight.device)
                    #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                    self.blocks[i][0][0].weight *= self.mask[i]

        # We use clamp_parameters of the Dcls1d modules
        if train: 
            for block in self.blocks:
                block[0][0].clamp_parameters()






    def decrease_sig(self, epoch):

        # Decreasing to 0.23 instead of 0.5

        alpha = 0
        sig = self.blocks[-1][0][0].SIG[0,0,0,0].detach().cpu().item()
        if self.config.decrease_sig_method == 'exp':
            if epoch < self.config.final_epoch and sig > 0.23:
                if self.config.DCLSversion == 'max':
                    # You have to change this !!
                    alpha = (1/self.config.sigInit)**(1/(self.config.final_epoch))
                elif self.config.DCLSversion == 'gauss':
                    alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for block in self.blocks:
                    block[0][0].SIG *= alpha
                    # No need to clamp after modifying sigma
                    #block[0][0].clamp_parameters()




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
            if self.config.spiking_neuron_type != 'heaviside':
                spikes = self.blocks[block_id][1][0](x)
            else:
                spikes = self.blocks[block_id][1][0](x - self.config.v_threshold)
            # we use dropout on generated spikes tensor


            x = self.blocks[block_id][1][1](spikes)

            # we apply synapse filter
            if self.config.stateful_synapse:
                x = self.blocks[block_id][1][2](x)
            
            # x is back to shape (time, batch, neurons)
        
        # Finally, we apply same transforms for the output layer
        x = x.permute(1,2,0)
        x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)
        
        # Apply final layer
        out = self.blocks[-1][0][0](x)

        # permute out: (batch, neurons, time) => (time, batch, neurons)  For final spiking neuron filter
        out = out.permute(2,0,1)

        if self.config.spiking_neuron_type != 'heaviside':
            out = self.blocks[-1][1][0](out)

            if self.config.loss != 'spike_count':
                out = self.blocks[-1][1][0].v_seq

        return out#, self.blocks[0][1][0].v_seq
    


    def get_model_wandb_logs(self):


        sig = self.blocks[-1][0][0].SIG[0,0,0,0].detach().cpu().item()

        model_logs = {"sigma":sig}

        for i in range(len(self.blocks)):
            
            if self.config.spiking_neuron_type != 'heaviside':
                tau_m = self.blocks[i][1][0].tau if self.config.spiking_neuron_type == 'lif' else  1. / self.blocks[i][1][0].w.sigmoid()
            else: tau_m = 0
            
            if self.config.stateful_synapse and i<len(self.blocks)-1:
                tau_s = self.blocks[i][1][2].tau if not self.config.stateful_synapse_learnable else  1. / self.blocks[i][1][2].w.sigmoid()
            else: tau_s = 0
            
            w = torch.abs(self.blocks[i][0][0].weight).mean()

            model_logs.update({f'tau_m_{i}':tau_m*self.config.time_step, 
                               f'tau_s_{i}':tau_s*self.config.time_step, 
                               f'w_{i}':w})

        return model_logs


    def round_pos(self):
        with torch.no_grad():
            for i in range(len(self.blocks)):
                self.blocks[i][0][0].P.round_()
                self.blocks[i][0][0].clamp_parameters()
