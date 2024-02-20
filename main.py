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
 
import dotenv
dotenv.load_dotenv()
import json 

from algo.operator import Operator

from operator_lib.operator_lib import OperatorLib

def handle_schema_error(error, message, produce_func):
    # catches cases when middle keys are missing like ENERGY, but not when last key like power is missing 
    msg_str = json.dumps(message)
    produce_func({
        "type": "schema",
        "sub_type": "",
        "value": msg_str, 
    })
    

if __name__ == "__main__":
    OperatorLib(
        Operator(), 
        name="anomaly-detector-operator", 
        git_info_file='git_commit',
        result_error_handler=handle_schema_error
    )
