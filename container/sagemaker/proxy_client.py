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

import grpc
import logging

from google.protobuf import text_format
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2

# TFS_GRPC_PORT = os.environ['TFS_GRPC_PORT']
MODEL_CONFIG_FILE = '/sagemaker/model-config.cfg'

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GRPCProxyClient(object):
    def __init__(self, port, host='0.0.0.0'):
        self.channel = grpc.insecure_channel('{}:{}'.format(host, port))
        self.stub = model_service_pb2_grpc.ModelServiceStub(self.channel)

    def add_model(self, model_name, base_path, model_platform='tensorflow'):
        # read model configs from existing model config file
        config_file = self._read_model_config(MODEL_CONFIG_FILE)
        model_server_config = model_server_config_pb2.ModelServerConfig()
        config_list = model_server_config_pb2.ModelConfigList()
        model_server_config = text_format.Parse(text=config_file, message=model_server_config)

        new_model_config = config_list.config.add()
        new_model_config.name = model_name
        new_model_config.base_path = base_path
        new_model_config.model_platform = model_platform

        # send HandleReloadConfigRequest to tensorflow model server
        model_server_config.model_config_list.MergeFrom(config_list)
        req = model_management_pb2.ReloadConfigRequest()
        req.config.CopyFrom(model_server_config)

        try:
            res = self.stub.HandleReloadConfigRequest(req)
        except Exception as e:
            log.exception('Exception handling request to load model {}: {}'
                          .format(model_name, res.status.error_message))
            raise Exception('error: {}; message: {}'
                            .format(res.status.error_code, res.status.error_message))

        # update local model-config.cfg file
        self._add_model_to_config_file(model_name, base_path, model_platform)
        return 'Successfully loaded model {}'.format(model_name)

    def _read_model_config(self, model_config_file):
        with open(model_config_file, 'r') as f:
            model_config = f.read()
        return model_config

    def _add_model_to_config_file(self, model_name, base_path, model_platform):
        # read existing model-config.cfg, excluding the closing bracket
        config = ''
        with open(MODEL_CONFIG_FILE, 'r') as f:
            line = f.readline()
            while line:
                if not line.startswith('}'):  # closing bracket of model-config.cfg
                    config += line
                line = f.readline()

        # add new model config, append the closing bracket
        config += '  config: {\n'
        config += '    name: "{}",\n'.format(model_name)
        config += '    base_path: "{}",\n'.format(base_path)
        config += '    model_platform: "{}"\n'.format(model_platform)
        config += '  },\n'
        config += '}\n'

        # write back to model-config.cfg
        with open(MODEL_CONFIG_FILE, 'w') as f:
            f.write(config)
