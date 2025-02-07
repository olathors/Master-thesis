from torchvision.models import EfficientNet, EfficientNet_V2_M_Weights
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.efficientnet import _efficientnet_conf, MBConvConfig, FusedMBConvConfig
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial



def CustomEfficientnet(
    *, weights: Optional[EfficientNet_V2_M_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_M_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_M_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_M_Weights
        :members:
    """
    weights = EfficientNet_V2_M_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")

    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )

def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))


    model.classifier= nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1280, 3),
        )

    return model