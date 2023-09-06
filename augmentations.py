# Copyright 2023 by Ismail Khalfaoui-Hassani, ANITI Toulouse.
#
# All rights reserved.
#
# This file is part of the ConvNeXt-Dcls-Audio package, and 
# is released under the "MIT License Agreement".
# Please see the LICENSE file that should have been included as part 
# of this package.

import random

from typing import Any, List, Tuple, Union

import torch

from torch import nn, Tensor
from torch.distributions import Uniform
from torch.nn import functional as F
from torchaudio.transforms import Resample as TorchAudioResample


class Crop(nn.Module):
    def __init__(
        self, target_length: int, align: str = "left", dim: int = -1, p: float = 1.0
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.align = align
        self.dim = dim
        self.p = p

    def extra_repr(self) -> str:
        return (
            f"target_length={self.target_length}, "
            f"align={self.align}, "
            f"dim={self.dim}"
        )

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.process(x)
        else:
            return x

    def process(self, data: Tensor) -> Tensor:
        if self.align == "center":
            return self.crop_align_center(data)
        elif self.align == "left":
            return self.crop_align_left(data)
        elif self.align == "random":
            return self.crop_align_random(data)
        elif self.align == "right":
            return self.crop_align_right(data)
        else:
            raise ValueError(
                f'Unknown alignment "{self.align}". Must be one of {str(["left", "right", "center", "random"])}.'
            )

    def crop_align_center(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            diff = data.shape[self.dim] - self.target_length
            start = diff // 2 + diff % 2
            end = start + self.target_length
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, end)
            data = data[slices]
            data = data.contiguous()
        return data

    def crop_align_left(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(self.target_length)
            data = data[slices]
            data = data.contiguous()
        return data

    def crop_align_random(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            diff = data.shape[self.dim] - self.target_length
            start = torch.randint(low=0, high=diff, size=()).item()
            end = start + self.target_length
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, end)
            data = data[slices]
            data = data.contiguous()
        return data

    def crop_align_right(self, data: Tensor) -> Tensor:
        if data.shape[self.dim] > self.target_length:
            start = data.shape[self.dim] - self.target_length
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, None)
            data = data[slices]
            data = data.contiguous()
        return data


class Pad(nn.Module):
    def __init__(
        self,
        target_length: int,
        align: str = "left",
        fill_value: float = 0.0,
        dim: int = -1,
        mode: str = "constant",
        p: float = 1.0,
    ) -> None:
        """
        Example :

        >>> import torch; from torch import tensor
        >>> x = torch.ones(6)
        >>> zero_pad = F.pad(10, align='left')
        >>> x_pad = zero_pad(x)
        ... tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

        :param target_length: The target length of the dimension.
        :param align: The alignment type. Can be 'left', 'right', 'center' or 'random'. (default: 'left')
        :param fill_value: The fill value used for constant padding. (default: 0.0)
        :param dim: The dimension to pad. (default: -1)
        :param mode: The padding mode. Can be 'constant', 'reflect', 'replicate' or 'circular'. (default: 'constant')
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__()
        self.target_length = target_length
        self.align = align
        self.fill_value = fill_value
        self.dim = dim
        self.mode = mode
        self.p = p

    def extra_repr(self) -> str:
        return (
            f"target_length={self.target_length}, "
            f"align={self.align}, "
            f"fill_value={self.fill_value}, "
            f"dim={self.dim}, "
            f"mode={self.mode}"
        )

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.process(x)
        else:
            return x

    def process(self, data: Tensor) -> Tensor:
        if self.align == "left":
            return self.pad_align_left(data)
        elif self.align == "right":
            return self.pad_align_right(data)
        elif self.align == "center":
            return self.pad_align_center(data)
        elif self.align == "random":
            return self.pad_align_random(data)
        else:
            raise ValueError(
                f'Unknown alignment "{self.align}". Must be one of {str(["left", "right", "center", "random"])}.'
            )

    def pad_align_left(self, x: Tensor) -> Tensor:
        # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        pad_seq[idx * 2 + 1] = missing

        x = F.pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def pad_align_right(self, x: Tensor) -> Tensor:
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        pad_seq[idx * 2] = missing

        x = F.pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def pad_align_center(self, x: Tensor) -> Tensor:
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        missing_left = missing // 2 + missing % 2
        missing_right = missing // 2

        pad_seq[idx * 2] = missing_left
        pad_seq[idx * 2 + 1] = missing_right

        x = F.pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x

    def pad_align_random(self, x: Tensor) -> Tensor:
        idx = len(x.shape) - (self.dim % len(x.shape)) - 1
        pad_seq = [0 for _ in range(len(x.shape) * 2)]

        missing = max(self.target_length - x.shape[self.dim], 0)
        missing_left = torch.randint(low=0, high=missing + 1, size=()).item()
        missing_right = missing - missing_left

        pad_seq[idx * 2] = missing_left  # type: ignore
        pad_seq[idx * 2 + 1] = missing_right  # type: ignore

        x = F.pad(x, pad_seq, mode=self.mode, value=self.fill_value)
        return x


class Resample(nn.Module):
    INTERPOLATIONS = ("nearest", "linear")

    def __init__(
        self,
        rates: Tuple[float, float] = (0.5, 1.5),
        interpolation: str = "nearest",
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """Resample an audio waveform signal.

        :param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2. (default: (0.5, 1.5))
        :param interpolation: Interpolation for resampling. Can be one of ("nearest", "linear").
            (default: "nearest")
        :param dim: The dimension to modify. (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        if interpolation not in self.INTERPOLATIONS:
            raise ValueError(
                f'Invalid argument mode interpolation={interpolation}. Must be one of {self.INTERPOLATIONS}.'
            )

        super().__init__()
        self.rates = rates
        self.interpolation = interpolation
        self.dim = dim
        self.p = p

    def extra_repr(self) -> str:
        return f"rates={str(self.rates)}"

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.process(x)
        else:
            return x

    def process(self, data: Tensor) -> Tensor:
        if self.rates[0] == self.rates[1]:
            rate = self.rates[0]
        else:
            sampler = Uniform(*self.rates)
            rate = sampler.sample().item()

        if self.interpolation == "nearest":
            data = self.resample_nearest(data, rate)
        elif self.interpolation == "linear":
            sampling_rate = 32000
            tchaudio_resample = TorchAudioResample(sampling_rate, int(sampling_rate * rate))
            data = tchaudio_resample(data)
        else:
            raise ValueError(
                f"Invalid argument interpolation={self.interpolation}. Must be one of {self.INTERPOLATIONS}."
            )

        return data

    def resample_nearest(self, data: Tensor, rate: float) -> Tensor:
        length = data.shape[self.dim]
        step = 1.0 / rate
        indexes = torch.arange(0, length, step)
        indexes = indexes.round().long().clamp(max=length - 1)
        slices: List[Any] = [slice(None)] * len(data.shape)
        slices[self.dim] = indexes
        data = data[slices]
        data = data.contiguous()
        return data


class SpeedPerturbation(nn.Module):
    def __init__(
        self,
        rates: Tuple[float, float] = (0.9, 1.1),
        target_length: Union[int, str] = "same",
        align: str = "random",
        fill_value: float = 0.0,
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """
        Resample, Pad and Crop the signal.

        :param rates: The ratio of the signal used for resize. (default: (0.9, 1.1))
        :param target_length: Optional target length of the signal dimension.
                If 'auto', the output will have the same shape than the input.
                (default: 'auto')
        :param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
                (default: 'random')
        :param fill_value: The fill value when padding the waveform. (default: 0.0)
        :param dim: The dimension to stretch and pad or crop. (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        super().__init__()
        self.rates = rates
        self._target_length = target_length
        self.align = align
        self.fill_value = fill_value
        self.dim = dim
        self.p = p

        target_length = self.target_length if isinstance(self.target_length, int) else 1
        self.stretch = Resample(rates, dim=dim)
        self.pad = Pad(target_length, align, fill_value, dim, mode="constant")
        self.crop = Crop(target_length, align, dim)

    def forward(self, x):
        if self.p >= 1.0 or random.random() <= self.p:
            return self.process(x)
        else:
            return x

    def process(self, data: Tensor) -> Tensor:
        if self.target_length == "same":
            target_length = data.shape[self.dim]
            self.pad.target_length = target_length
            self.crop.target_length = target_length

        data = self.stretch(data)
        data = self.pad(data)
        data = self.crop(data)
        return data

    @property
    def target_length(self) -> Union[int, str]:
        return self._target_length


class PadOrTruncate(object):
    """Pad all audio to specific length."""
    def __init__(self, audio_length):
        self.audio_length = audio_length

    def __call__(self, sample):
        if len(sample) <= self.audio_length:
            return F.pad(sample, (0, self.audio_length - sample.size(-1)))
        else:
            return sample[0: self.audio_length]
    def __repr__(self):
        return f"PadOrTruncate(audio_length={self.audio_length})"

class RandomRoll(object):
    def __init__(self, dims):
        self.dims = dims
    def __call__(self, sample):
        shifts = [torch.randint(-sample.size(dim),
                                sample.size(dim),
                                size=(1,)) for dim in self.dims]
        return torch.roll(sample, shifts, self.dims)
    def __repr__(self):
        return f"RandomRoll(dims={self.dims})"
