import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.build_model()
        self.init_model()

    #def forward(self, x):
    #    return self.model(x)


    def optimizers(self):
        # returns a list of optimizers
        optimizers_return = []
        
        if self.config.model_type in ['snn_delays', 'snn_delays_lr0']:
            if self.config.optimizer_w == 'adam':
                optimizers_return.append(optim.Adam(self.weights, lr = self.config.lr_w, betas=(0.9,0.999)))
            if self.config.model_type == 'snn_delays':
                if self.config.optimizer_pos == 'adam':
                    optimizers_return.append(optim.Adam(self.positions, lr = self.config.lr_pos, betas=(0.9,0.999)))
        elif self.config.model_type == 'ann':
            if self.config.optimizer_w == 'adam':
                optimizers_return.append(optim.Adam(self.model.parameters(), lr = self.config.lr_w, betas=(0.9,0.999)))

        return optimizers_return

    def schedulers(self, optimizers):
        # returns a list of schedulers
        schedulers_return = []
        
        if self.config.model_type in ['snn_delays', 'snn_delays_lr0']:
            if self.config.scheduler_w == 'one_cycle':
                schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[0], max_lr=self.config.max_lr_w,
                                                                             total_steps=self.config.epochs))
            if self.config.model_type == 'snn_delays':
                if self.config.scheduler_pos == 'one_cycle':
                    schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[1], max_lr=self.config.max_lr_pos,
                                                                                total_steps=self.config.epochs))
                elif self.config.scheduler_pos == 'cosine_a':
                    schedulers_return.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1],  
                                                                                        T_max = self.config.t_max_pos))
       
        elif self.config.model_type == 'ann':
            if self.config.scheduler_w == 'one_cycle':
                schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[0], max_lr=self.config.max_lr_w,
                                                                             total_steps=self.config.epochs))

        return schedulers_return
    


    def calc_loss(self, output, y):

        if self.config.loss == 'mean': m = torch.mean(output, 0)
        elif self.config.loss == 'max': m, _ = torch.max(output, 0)

        # probably better to add it in init, or in general do it one time only
        if self.config.loss_fn == 'CEloss':
            #compare using this to directly using nn.CrossEntropyLoss
            log_softmax_fn = nn.LogSoftmax(dim=1) 
            loss_fn = nn.NLLLoss()

            log_p_y = log_softmax_fn(m) 
            return loss_fn(log_p_y, y)
        

    def calc_metric(self, output, y):
        # mean accuracy over batch
        if self.config.loss == 'mean': m = torch.mean(output, 0)
        elif self.config.loss == 'max': m, _ = torch.max(output, 0)

        return np.mean((y==torch.max(m,1)[1]).detach().cpu().numpy())
    


    def train_model(self, train_loader, valid_loader, device):

        if self.config.use_wandb:
            wandb.login(key="25f19d79982fd7c29f092981a100f187f2c706b4")

            wandb.init(
                project= self.config.wandb_project_name,
                name=self.config.wandb_run_name)
        

        optimizers = self.optimizers()
        schedulers = self.schedulers(optimizers)

        loss_epochs = {'train':[], 'valid':[]}
        metric_epochs = {'train':[], 'valid':[]}
        for epoch in range(self.config.epochs):
            self.train()
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

                self.reset_model(train=True)
                
            
            loss_epochs['train'].append(np.mean(loss_batch))
            metric_epochs['train'].append(np.mean(metric_batch))

            for scheduler in schedulers: scheduler.step()
            self.decrease_sig(epoch)

            self.eval()
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

                    self.reset_model(train=False)

                loss_epochs['valid'].append(np.mean(loss_batch))
                metric_epochs['valid'].append(np.mean(metric_batch))
                
        
            print(f"=====> Epoch {epoch} : \nLoss Train = {loss_epochs['train'][-1]:.3f}  |  Best Acc Train = {100*max(metric_epochs['train']):.2f}% \nLoss Test = {loss_epochs['valid'][-1]:.3f}  |  Best Acc Test = {100*max(metric_epochs['valid']):.2f}%")

            if self.config.use_wandb:
                lr_w = schedulers[0].get_last_lr()[0]

                if self.config.model_type in ['snn_delays', 'snn_delays_lro']:
                    sig = self.blocks[0][0][0].SIG[0,0,0,0].detach().cpu().item()
                if self.config.model_type  == 'snn_delays':
                    lr_pos = schedulers[1].get_last_lr()[0]
                
                
                if self.config.model_type == 'snn_delays':
                    wandb.log({"Epoch":epoch,"loss_train":loss_epochs['train'][-1], 
                            "acc_train" : metric_epochs['train'][-1], "acc_test" : metric_epochs['valid'][-1], 
                            "loss_test" : loss_epochs['valid'][-1],"lr_w":lr_w, "lr_p":lr_pos, "sigma":sig})
                
                elif self.config.model_type == 'snn_delays_lr0':
                    wandb.log({"Epoch":epoch,"loss_train":loss_epochs['train'][-1], 
                            "acc_train" : metric_epochs['train'][-1], "acc_test" : metric_epochs['valid'][-1], 
                            "loss_test" : loss_epochs['valid'][-1],
                            "lr_w":lr_w, "sigma":sig})
                else:
                    wandb.log({"Epoch":epoch,"loss_train":loss_epochs['train'][-1], 
                            "acc_train" : metric_epochs['train'][-1], "acc_test" : metric_epochs['valid'][-1], 
                            "loss_test" : loss_epochs['valid'][-1],"lr_w":lr_w})
        
        if self.config.use_wandb:
            wandb.run.finish()   