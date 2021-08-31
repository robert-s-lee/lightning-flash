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
from flash.core.integrations.vissl.adapter import VISSLAdapter
from flash.core.utilities.imports import _VISSL_AVAILABLE
from flash.core.registry import FlashRegistry

if _VISSL_AVAILABLE:
    from classy_vision.losses import ClassyLoss, LOSS_REGISTRY


def dino_loss(
    num_crops,
    momentum=0.996,
    student_temp=0.1,
    teacher_temp_min=0.04,
    teacher_temp_max=0.07,
    teacher_temp_warmup_iters=37530, # convert this to 30 epochs
    crops_for_teacher=[0, 1],
    ema_center=0.9,
    normalize_last_layer=False,
) -> ClassyLoss:
    cfg = {
        "num_crops": num_crops,
        "momentum": momentum,
        "student_temp": student_temp,
        "teacher_temp_min": teacher_temp_min,
        "teacher_temp_max": teacher_temp_max,
        "teacher_temp_warmup_iters": teacher_temp_warmup_iters,
        "crops_for_teacher": crops_for_teacher,
        "ema_center": ema_center,
        "normalize_last_layer": normalize_last_layer,
    }
    loss_fn = LOSS_REGISTRY['dino_loss'](cfg)
    return loss_fn


def register_vissl_losses(register: FlashRegistry):
    for loss_fn in (dino_loss):
        register(loss_fn, adapter=VISSLAdapter)
