"""
   Copyright 2022 InfAI (CC SES)

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

__all__ = ("OperatorConfig", )

import simple_struct
import json
import typing


class Selector(simple_struct.Structure):
    name: str = None
    args: typing.Set[str] = None

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.args = set(self.args)


class Config(simple_struct.Structure):
    device_id: str = None
    data_path = "/opt/data"
    device_name: str = None
    logger_level = "warning"
    selectors: typing.List[Selector] = None
    check_data_anomalies: bool = False
    check_data_extreme_outlier: bool = True
    check_data_schema: bool = True
    check_receive_time_outlier: bool = True

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        if self.selectors:
            self.selectors = [Selector(s) for s in json.loads(self.selectors)]
        
        for boolean_prop in [self.check_data_anomalies, self.check_data_extreme_outlier, self.check_data_schema, self.check_receive_time_outlier]:
            if boolean_prop:
                boolean_prop = (boolean_prop == "True" or boolean_prop == "true" or boolean_prop == "1")


class Mapping(simple_struct.Structure):
    dest: str = None
    source: str = None


class InputTopic(simple_struct.Structure):
    name: str = None
    filterType: str = None
    filterValue: str = None
    mappings: typing.List[Mapping] = None

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.mappings = [Mapping(m) for m in self.mappings]


class OperatorConfig(simple_struct.Structure):
    config = Config
    inputTopics: typing.List[InputTopic] = None

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.inputTopics = [InputTopic(it) for it in self.inputTopics]
