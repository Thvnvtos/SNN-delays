[![arXiv](https://img.shields.io/badge/arXiv-2306.17670-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2306.17670)
[![openreview](https://img.shields.io/badge/read%20it%20on-OpenReview-b31b1b?style=plastic)](https://openreview.net/forum?id=4r2ybzJnmN)


#  Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings

![](https://github.com/Thvnvtos/SNN-delays/blob/master/delays_figures.png)

This is the official implementation of [Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings](https://openreview.net/forum?id=4r2ybzJnmN) by Ilyass Hammouamri, Ismail Khalfaoui Hassani and Timothée Masquelier.

## Dependencies
### Python
Please use Python 3.9 and above ```python>=3.9```

### SpikingJelly
Install SpikingJelly using:
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install --user
```
Installing SpikingJelly using ```pip``` is not yet compatible with this repo.

### Other
Install the other dependencies from the requirements.txt file using:
```
pip install -r requirements.txt
```


## Usage
The first thing to do after installing all the dependencies is to specify the ```datasets_path``` in ```config.py```. Simply create an empty data directory, preferably with two subdirectories, one for SHD and the other SSC. The ```datasets_path``` should correspond to these subdirectories.
The datasets will then be downloaded and preprocessed automatically. For example:
```
cd SNN-delays
mkdir -p Datasets/SHD
mkdir -p Datasets/SSC
```

To train a new model as defined by the ```config.py``` simply use:
```
python main.py
```

The loss and accuracy for the training and validation at every epoch will be printed to ```stdout``` and the best model will be saved to the current directory.
If the ```use_wandb``` parameter is set to ```True```, a more detailed log will be available at the wandb project specified in the configuration.


## Publications and Citation

If you use DCLS-Delays in your work, please consider to cite it as follows:

```
@inproceedings{
  hammouamri2024learning,
  title={Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings},
  author={Ilyass Hammouamri and Ismail Khalfaoui-Hassani and Timoth{\'e}e Masquelier},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=4r2ybzJnmN}
}

```
