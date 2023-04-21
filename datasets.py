from utils import set_seed

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets.shd import SpikingSpeechCommands
from spikingjelly.datasets import pad_sequence_collate



class RNoise(object):
  
  def __init__(self, sig):
    self.sig = sig
        
  def __call__(self, sample):
    print("Called")
    noise = np.abs(np.random.normal(0, self.sig, size=sample.shape).round())
    return sample + noise


class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config
  
  def __call__(self, sample):
    # Time mask
    if np.random.uniform() < self.config.TN_mask_aug_proba:
      mask_size = np.random.randint(0, self.config.time_mask_size)
      ind = np.random.randint(0, sample.shape[0])
      sample[ind:ind+mask_size, :] = 0

    # Freq mask
    if np.random.uniform() < self.config.TN_mask_aug_proba:
      mask_size = np.random.randint(0, self.config.freq_mask_size)
      ind = np.random.randint(0, sample.shape[1])
      sample[:, ind:ind+mask_size] = 0

    return sample


class Augs(object):

  def __init__(self, config):
    self.config = config
    self.augs = [TimeNeurons_mask_aug(config)]
  
  def __call__(self, batch):
    for sample in batch:
      for aug in self.augs:
        sample = aug(sample)
    
    return batch



def SHD_dataloaders(config):
  set_seed(config.seed)

  train_dataset = SpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= SpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)

  return train_loader, valid_loader, test_loader




def SSC_dataloaders(config):
  set_seed(config.seed)

  train_dataset = SpikingSpeechCommands(config.datasets_path, config.n_bins, split='train', data_type='frame', duration=config.time_step)
  valid_dataset = SpikingSpeechCommands(config.datasets_path, config.n_bins, split='valid', data_type='frame', duration=config.time_step)
  test_dataset = SpikingSpeechCommands(config.datasets_path, config.n_bins, split='test', data_type='frame', duration=config.time_step)


  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)

  return train_loader, valid_loader, test_loader