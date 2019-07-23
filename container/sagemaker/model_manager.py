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

import falcon
import json
import logging
import os

from proxy_client import GRPCProxyClient

TFS_GRPC_PORT = os.environ.get('TFS_GRPC_PORT')
DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ModelManagerResource(object):

    def __init__(self):
        self.grpc_client = GRPCProxyClient(TFS_GRPC_PORT)

    def on_get(self, req, res):  # pylint: disable=W0613
        with open('/sagemaker/model-config.cfg') as f:
            models = f.read()
            res.body = models
            res.status = falcon.HTTP_200

    def on_post(self, req, res):
        res.status = falcon.HTTP_200
        try:
            data = json.loads(req.stream.read().decode('utf-8'))
            model_name = data['name']
            base_path = data['uri']
            msg = self.grpc_client.add_model(model_name, base_path)
            res.body = msg
            res.status = falcon.HTTP_200
        except Exception as e:
            res.status = falcon.HTTP_507
            res.body = json.dumps({
                'error': str(e)
            }).encode('utf-8')

    def on_delete(self, req, res):
        pass

    def _parse_request(self, req):
        content_type = req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE
        accept_header = req.get_header('Accept') or DEFAULT_ACCEPT_HEADER
        data = req.stream.read()
        return data, content_type, accept_header
