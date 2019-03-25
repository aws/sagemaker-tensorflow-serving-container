# Copyrasight 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import re
from collections import namedtuple

import falcon
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TFS_BASE_URI = 'http://localhost:9001/v1/models/'
TFS_DEFAULT_MODEL = 'hal_plus_three'
TFS_DEFAULT_METHOD = 'predict'
DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')


class InvocationResource(object):

    def __init__(self):
        self.handler = None
        self.input_handler = None
        self.output_handler = None
        self._import_handlers()

    def on_post(self, req, res):
        context, data = self._parse_http_request(req)

        if self.handler:
            log.warning('Using handler, input_handler and output_handler will be ignored.')
            log.info('handler type: ' + str(type(self.handler)))
            res.status = falcon.HTTP_200
            res.body = self.handler(data, context)
        elif self.input_handler and self.output_handler:
            processed_input = self.input_handler(data, context)
            response = requests.post(context.rest_uri, data=processed_input)
            res.status = falcon.HTTP_200
            res.body, res.content_type = self.output_handler(response, context)
        else:
            res.status = falcon.HTTP_500
            res.body = 'Handlers are not implemented correctly.'

    def _import_handlers(self):
        # import handlers
        import importlib.util
        spec = importlib.util.spec_from_file_location('inference',
                                                      '/opt/ml/model/inference_example.py')
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        self.handler = inference.handler
        self.input_handler = inference.input_handler
        self.output_handler = inference.output_handler

    def _parse_http_request(self, req):
        content_type = req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE
        accept = req.get_header('Accept') or DEFAULT_ACCEPT_HEADER
        custom_attributes = req.get_header('X-Amzn-SageMaker-Custom-Attributes')
        model_name = req.get_header('model_name') or TFS_DEFAULT_MODEL
        model_version = req.get_header('model_version')
        method = req.get_header('method') or TFS_DEFAULT_METHOD

        uri = self._tfs_rest_uri(self._parse_custom_attributes(custom_attributes))
        rest_uri = req.get_header('rest_uri') or uri
        grpc_uri = req.get_header('grpc_uri')

        custom_attributes = self._make_custom_attributes(custom_attributes,
                                                         model_name,
                                                         model_version,
                                                         method)
        context = Context(model_name=model_name, model_version=model_version, method=method,
                          rest_uri=rest_uri, grpc_uri=grpc_uri, custom_attributes=custom_attributes,
                          request_content_type=content_type, accept_header=accept)
        data = req.stream.read()

        return context, data

    def _make_custom_attributes(self, custom_attributes, model_name, model_version, tfs_method):
        attribute_list = []
        if custom_attributes:
            attribute_list = re.findall("(tfs-[a-z\-]+=[^,]+)",  # pylint: disable=W1401; # noqa: W605, E501
                                        custom_attributes)
        name = False
        method = False
        version = False
        for attribute in attribute_list:
            k, _ = attribute.split('=')
            if k == 'tfs-model-name':
                name = True
            if k == 'tfs-method':
                method = True
            if k == 'tfs-model-version':
                version = True

        if not name:
            attribute_list.append('tfs-model-name={}'.format(model_name))
        if not method:
            attribute_list.append('tfs-method={}'.format(tfs_method))
        if not version and model_version:
            attribute_list.append('tfs-model-version={}'.format(model_version))

        return ','.join(attribute_list)

    def _parse_custom_attributes(self, custom_attributes):
        attribute_list = re.findall("(tfs-[a-z\-]+=[^,]+)",  # pylint: disable=W1401; # noqa: W605
                                    custom_attributes)
        attributes = {}
        for attribute in attribute_list:
            k, v = attribute.split('=')
            attributes[k] = v
        return attributes

    def _tfs_rest_uri(self, attributes):
        uri = TFS_BASE_URI + (attributes['tfs-model-name'] or TFS_DEFAULT_MODEL)
        if 'tfs-model-version' in attributes:
            uri += '/versions/' + attributes['tfs-model-version']
        uri += ':' + (attributes['tfs-method'] or TFS_DEFAULT_METHOD)
        return uri


class PingResource(object):
    def on_get(self, req, res):  # pylint: disable=W0613
        res.status = falcon.HTTP_200
        res.body = 'OK!\n'


# receiving requests to /invocations and /ping
ping_resource = PingResource()
invocation_resource = InvocationResource()

app = falcon.API()

app.add_route('/ping', ping_resource)
app.add_route('/invocations', invocation_resource)
