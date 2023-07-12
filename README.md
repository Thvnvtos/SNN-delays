#  Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings

This repository contains the code used to obtain the results in [Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings](https://arxiv.org/abs/2306.17670)

## Dependencies
### SpikingJelly
Install SpikingJelly using:
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```
Installing SpikingJelly using ```pip``` is not yet compatible with this repo.

### DCLS
Install DCLS following the instruction from the [official repo](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch#installation) using:
```
pip install dcls
```

### PyTorch
Install PyTorch version ```torch>=2.0.0``` by following instructions on the official [website](https://pytorch.org/)

### Others
- wandb
- h5py
