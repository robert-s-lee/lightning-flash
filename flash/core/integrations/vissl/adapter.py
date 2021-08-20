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
from flash.core.utilities.url_error import catch_url_error


class VISSLAdapter(Adapter):
    """The ``VISSLAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with VISSL."""

    required_extras: str = "image"

    def __init__(self):
        super().__init__()

    def forward(self, x: Any) -> Any:
        # TODO: Adapt VISSL BaseSSLMultiInputOutputModel to process input batch here
        pass

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        # TODO: Call to forward and then
        # TODO: Call ClassyLoss on forward
        # TODO: Include call to ClassyHooks during training
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # TODO: Call to forward and then
        # TODO: Call ClassyLoss on forward
        # TODO: Include call to ClassyHooks during training
        pass

    def test_step(self, batch: Any, batch_idx: int) -> None:
        # TODO: Call to forward and then
        # TODO: Call ClassyLoss on forward
        # TODO: Include call to ClassyHooks during training
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # TODO: return embedding here
        pass
