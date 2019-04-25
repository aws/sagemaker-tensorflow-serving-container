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
import os

import importlib.util
import re
from collections import namedtuple

import falcon
import requests

DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'
CUSTOM_ATTRIBUTES = 'X-Amzn-SageMaker-Custom-Attributes'

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_port, '
                     'custom_attributes, request_content_type, accept_header, content_length')


class InvocationResource(object):

    def __init__(self):
        if 'TFS_GRPC_PORT' not in os.environ:
            raise EnvironmentError('GRPC port not set')
        if 'TFS_REST_PORT' not in os.environ:
            raise EnvironmentError('REST API port not set.')

        self._tfs_grpc_port = os.getenv('TFS_GRPC_PORT')
        self._tfs_rest_port = os.getenv('TFS_REST_PORT')

        self._handler, self._input_handler, self._output_handler = self._import_handlers()
        self._handlers = self._make_handler(self._handler,
                                            self._input_handler,
                                            self._output_handler)

    def on_post(self, req, res):
        data, context = self._parse_request(req)
        try:
            prediction, content_type = self._handlers(data, context)
            res.status = falcon.HTTP_200
            res.body, res.content_type = prediction, content_type
        except Exception as e:
            raise Exception("Error in handling request: {}".format(str(e)))

    def _import_handlers(self):
        spec = importlib.util.spec_from_file_location('inference',
                                                      '/opt/ml/model/script/inference.py')
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        _custom_handler, _custom_input_handler, _custom_output_handler = None, None, None
        try:
            _custom_handler = inference.handler
        except AttributeError:
            try:
                _custom_input_handler = inference.input_handler
                _custom_output_handler = inference.output_handler
            except AttributeError:
                raise NotImplementedError('Handlers are not implemented correctly in user script.')
        return _custom_handler, _custom_input_handler, _custom_output_handler

    def _make_handler(self, custom_handler, custom_input_handler, custom_output_handler):
        def handler(data, context):
            if custom_handler:
                return custom_handler(data, context)
            else:
                processed_input = custom_input_handler(data, context)
                response = requests.post(context.rest_uri, data=processed_input)
                return custom_output_handler(response, context)
        return handler

    def _parse_request(self, req):
        attributes = self._parse_custom_attributes(req)
        context = self._make_context(attributes,
                                     req.get_header(CUSTOM_ATTRIBUTES),
                                     req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE,
                                     req.get_header('Accept') or DEFAULT_ACCEPT_HEADER,
                                     req.content_length)
        data = req.stream
        return data, context

    def _parse_custom_attributes(self, req):
        attributes = {}
        header = req.get_header(CUSTOM_ATTRIBUTES)
        if header:
            for attribute in re.findall(r'(tfs-[a-z\-]+=[^,]+)', header):
                k, v = attribute.split('=')
                attributes[k] = v
        return attributes

    def _make_context(self, attributes, custom_attributes, content_type, accept, content_length):
        model_name = attributes.get('tfs-model-name')
        model_version = attributes.get('tfs-model-version')
        method = attributes.get('tfs-method')

        rest_uri = self._tfs_uri(self._tfs_rest_port, attributes)
        grpc_port = self._tfs_grpc_port
        request_content_type = content_type
        accept_header = accept

        return Context(model_name, model_version, method, rest_uri, grpc_port,
                       custom_attributes, request_content_type, accept_header, content_length)

    def _tfs_uri(self, port, attributes):
        uri = 'http://localhost:{}/v1/models/{}'.format(port, attributes['tfs-model-name'])
        if 'tfs-model-version' in attributes:
            uri += '/versions/' + attributes['tfs-model-version']
        uri += ':' + (attributes['tfs-method'])
        return uri


class PingResource(object):
    def on_get(self, req, res):  # pylint: disable=W0613
        res.status = falcon.HTTP_200


# receiving requests to /invocations and /ping
ping_resource = PingResource()
invocation_resource = InvocationResource()

app = falcon.API()

app.add_route('/ping', ping_resource)
app.add_route('/invocations', invocation_resource)
