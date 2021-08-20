# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric

from flash.core.adapter import Adapter, AdapterTask
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.core.data.process import Preprocess

if _IMAGE_AVAILABLE:
    from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
else:
    IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")


class ImageEmbedder(AdapterTask):
    """The ``ImageEmbedder`` is a :class:`~flash.Task` for obtaining feature vectors (embeddings) from images. For
    more details, see :ref:`image_embedder`.

    Args:
        embedding_dim: Dimension of the embedded vector. ``None`` uses the default from the backbone. Default value
            or setting it to ``None`` ignores the heads args.
        backbone: A model to use to extract image features, defaults to ``"swav-imagenet"``.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        heads: A list of heads to be applied to the backbones for training, validation, test or predict. Defaults
            to ``None``.
        loss_fn: Loss function for training and finetuning, defaults to :func:`torch.nn.functional.cross_entropy`
        optimizer: Optimizer to use for training and finetuning, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
    """

    backbones: FlashRegistry = IMAGE_CLASSIFIER_BACKBONES

    required_extras: str = "image"

    def __init__(
        self,
        adapter: Adapter,
        embedding_dim: Optional[int] = None,
        backbone: str = "resnet50",
        pretrained: bool = True,
        heads: Optional[Union[nn.Module, nn.ModuleList]] = None,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Optional[Union[Metric, Callable, Mapping, Sequence]] = None,
        learning_rate: float = 1e-3,
        preprocess: Optional[Preprocess] = None,
    ):
        self.save_hyperparameters()

        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.heads = heads

        self.backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained)

        # TODO: figure out how embedding_dim and head or a list of heads will interact
        # if embedding_dim is None:
        #     self.head = nn.Identity()
        # else:
        #     self.head = nn.Sequential(
        #         nn.Flatten(),
        #         nn.Linear(num_features, embedding_dim),
        #     )
        #     rank_zero_warn("Adding linear layer on top of backbone. Remember to finetune first before using!")

        super().__init__(
            adapter=adapter,
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            preprocess=preprocess,
        )
