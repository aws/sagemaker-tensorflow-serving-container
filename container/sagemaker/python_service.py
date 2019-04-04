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
import json
import os

import importlib.util
import logging
import re
from collections import namedtuple

import falcon
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'
CUSTOM_ATTRIBUTES = 'X-Amzn-SageMaker-Custom-Attributes'

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')


class InvocationResource(object):

    def __init__(self):
        if 'SAGEMAKER_SAFE_PORT_RANGE' in os.environ:
            port_range = os.environ['SAGEMAKER_SAFE_PORT_RANGE']
            parts = port_range.split('-')
            low = int(parts[0])
            high = int(parts[1])
            if low + 2 > high:
                raise ValueError('not enough ports available in SAGEMAKER_SAFE_PORT_RANGE ({})'
                                 .format(port_range))
            self._tfs_grpc_port = str(low)
            self._tfs_rest_port = str(low + 1)
        else:
            self._tfs_grpc_port = '9000'
            self._tfs_rest_port = '8501'

        self.handler = None
        self.input_handler = None
        self.output_handler = None
        self._import_handlers()

    def on_post(self, req, res):
        context, data = self._parse_request(req)

        if self.handler:
            res.status = falcon.HTTP_200
            res.body = self.handler(data, context)
        elif self.input_handler and self.output_handler:
            processed_input = self.input_handler(data, context)
            response = requests.post(context.rest_uri, data=json.dumps(processed_input))
            prediction, content_type = self.output_handler(response, context)
            prediction = json.dumps(prediction)

            res.status = falcon.HTTP_200
            res.body, res.content_type = prediction, content_type

    def _import_handlers(self):
        spec = importlib.util.spec_from_file_location('inference', '/opt/ml/model/inference.py')
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        try:
            self.handler = inference.handler
            log.warning('Using handler, input_handler and output_handler will be ignored.')
        except AttributeError:
            try:
                self.input_handler = inference.input_handler
                self.output_handler = inference.output_handler
            except AttributeError:
                log.error('Handlers are not implemented correctly.')

    def _parse_request(self, req):
        content_type = req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE
        accept = req.get_header('Accept') or DEFAULT_ACCEPT_HEADER
        attributes = self._parse_custom_attributes(req.get_header(CUSTOM_ATTRIBUTES))

        context = self._make_context(attributes, content_type, accept)
        data = req.stream.read().decode('utf-8')
        return context, data

    def _parse_custom_attributes(self, custom_attributes):
        attribute_list = []
        attributes = {}
        if custom_attributes:
            attribute_list = re.findall(r'(tfs-[a-z\-]+=[^,]+)', custom_attributes)
        for attribute in attribute_list:
            k, v = attribute.split('=')
            attributes[k] = v
        return attributes

    def _make_context(self, attributes, content_type, accept):
        model_name = None
        model_version = None
        method = None
        custom_attributes = {}
        for key in attributes:
            if key == 'tfs-model-name':
                model_name = attributes[key]
            elif key == 'tfs-model-version':
                model_version = attributes[key]
            elif key == 'tfs-method':
                method = attributes[key]
            else:
                custom_attributes[key] = attributes[key]

        rest_uri = self._tfs_uri(self._tfs_rest_port, attributes)
        grpc_uri = self._tfs_uri(self._tfs_grpc_port, attributes)
        request_content_type = content_type
        accept_header = accept

        return Context(model_name, model_version, method, rest_uri, grpc_uri,
                       custom_attributes, request_content_type, accept_header)

    def _tfs_uri(self, port, attributes):
        uri = 'http://localhost:{}/v1/models/'.format(port) + (attributes['tfs-model-name'])
        if 'tfs-model-version' in attributes:
            uri += '/versions/' + attributes['tfs-model-version']
        uri += ':' + (attributes['tfs-method'])
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
