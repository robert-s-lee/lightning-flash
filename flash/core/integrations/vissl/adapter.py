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
from typing import Any, Callable, Optional

from flash.core.adapter import Adapter
from flash.core.model import Task
from flash.core.utilities.imports import _VISSL_AVAILABLE
from flash.core.utilities.url_error import catch_url_error

if _VISSL_AVAILABLE:
    from classy_vision.losses import build_loss, ClassyLoss
    from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel


class VISSLAdapter(Adapter):
    """The ``VISSLAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with VISSL."""

    required_extras: str = "image"

    def __init__(self, vissl_model: BaseSSLMultiInputOutputModel, vissl_loss: ClassyLoss):
        super().__init__()

        self.vissl_model = vissl_model
        self.vissl_loss = vissl_loss

    def forward(self, batch) -> Any:
        return self.vissl_model.forward(batch)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        vissl_input, target = batch
        out = self(vissl_input)

        # out can be torch.Tensor/List target is torch.Tensor
        loss = self.vissl_loss(out, target)

        # TODO: log
        # TODO: Include call to ClassyHooks during training
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        vissl_input, target = batch
        out = self(vissl_input)

        # out can be torch.Tensor/List target is torch.Tensor
        loss = self.vissl_loss(out, target)

        # TODO: log
        # TODO: Include call to ClassyHooks during training
        pass

    def test_step(self, batch: Any, batch_idx: int) -> None:
        vissl_input, target = batch
        out = self(vissl_input)

        # out can be torch.Tensor/List target is torch.Tensor
        loss = self.vissl_loss(out, target)

        # TODO: log
        # TODO: Include call to ClassyHooks during training
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # TODO: return embedding here
        pass
<<<<<<< HEAD



class MockVISSLTask:
    def __init__(self, loss) -> None:
        pass
=======
>>>>>>> b2788038e0f924f6c905c892aa9200ccc0cd977e
