"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("SCHEMA_VERSION", "DeviceProfile", "AnomalyModel")

import dataclasses
import datetime
import typing

import mlflow.pyfunc

if typing.TYPE_CHECKING:
    # For type hints only: the detectors need not be loaded with the model.
    from .detectors.extreme_value import ValueStats
    from .detectors.receive_time import GapStats

# 2: without the curve model of a device.
SCHEMA_VERSION = 2


@dataclasses.dataclass
class DeviceProfile:
    values: typing.Optional["ValueStats"]
    gaps: typing.Optional["GapStats"]


class AnomalyModel(mlflow.pyfunc.PythonModel):
    def __init__(self, schema_version: int, trained_until: datetime.datetime, devices: typing.Dict[str, DeviceProfile]):
        self.schema_version = schema_version
        self.trained_until = trained_until
        self.devices = devices

    def profile(self, device_key) -> typing.Optional[DeviceProfile]:
        return self.devices.get(device_key)

    # No type hints: mlflow checks them only where present, and the model is never called through predict.
    def predict(self, context, model_input, params=None):
        return list(self.devices)
