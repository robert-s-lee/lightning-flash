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
from typing import Any, Callable, Dict, Tuple

from torch import nn
from flash.core.adapter import AdapterTransform


class VISSLTransformAdapter(AdapterTransform):
    def __init__(self, train_transform, val_transform, test_transform):
        super().__init__(train_transform, val_transform, test_transform)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def forward(self, batch):
        return self.train_transform(batch)
