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

## Usage
The first thing to do after installing all the dependencies is to specify the ```datasets_path``` in ```config.py```. Simply create a empty data directory, preferably with two subdirectories, one for SHD and the other SSC. The ```datasets_path``` should correspond to these subdirectories.
The datasets will then be downloaded and preprocessed automatically.

To train a new model as defined by the ```config.py``` simply use:
```
python main.py
```

The loss and accuracy for the training and validation at every epoch will be printed to ```stdout``` and the best model will be saved to the current directory.
If the ```use_wandb``` parameter is set to ```True```, a more detailed log will be available at the wandb project specified in the configuration.
