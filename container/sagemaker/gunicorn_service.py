# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from model_manager import ModelManagerResource
from python_service import PingResource, InvocationResource

import falcon
import os

INFERENCE_SCRIPT_PATH = '/opt/ml/model/code/inference.py'


class ServiceResources(object):
    def __init__(self):
        self._enable_python_service = os.path.exists(INFERENCE_SCRIPT_PATH)
        self._enable_model_manager = os.environ.get('SAGEMAKER_TFS_ENABLE_DYNAMIC_ENDPOINT')

    def add_routes(self, app):
        if self._enable_python_service:
            ping_resource = PingResource()
            invocation_resource = InvocationResource()
            app.add_route('/ping', ping_resource)
            app.add_route('/invocations', invocation_resource)

        if self._enable_model_manager:
            model_manager_resource = ModelManagerResource()
            app.add_route('/models', model_manager_resource)


app = falcon.API()
resources = ServiceResources()
resources.add_routes(app)
