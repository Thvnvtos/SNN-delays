from datasets import SHD_dataloaders, SSC_dataloaders
from config import Config
from snn_delays import SnnDelays
import torch
from snn import SNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n===> Device = {device}")

config = Config()

if config.dataset == 'shd':
    train_loader, valid_loader = SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'ssc':
    train_loader, valid_loader, test_loader = SSC_dataloaders(config)
else:
    raise Exception(f'dataset {config.dataset} not implemented')

if config.model_type == 'snn':
    model = SNN(config).to(device)
else:
    model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()

model.train_model(train_loader, valid_loader, test_loader, device)
