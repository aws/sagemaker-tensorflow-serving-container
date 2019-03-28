# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
import re
from collections import namedtuple

import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

DEFAULT_TFS_MODEL = 'half_plus_three'
DEFAULT_TFS_METHOD = 'predict'
DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'
BASE_URI = 'http://localhost:8080/invocations'


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == 'application/json':
        data = _parse_json(data)
    elif context.request_content_type == 'application/jsons' or \
            context.request_content_type == 'application/jsonlines':
        data = _parse_jsonlines(data)

    elif context.request_content_type == 'text/csv':
        data = _parse_csv(data)
    else:
        _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))

    return data


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (dict, string): data to return to client, response content type
    """
    processed_output = json.loads(data.text)
    response_content_type = context.accept_header or DEFAULT_CONTENT_TYPE
    return processed_output, response_content_type


def handler(data, context):
    """Pre-process request input, and post request to TensorFlow REST API, then post-process output before it is
    sent to the client

    Args:
        data (obj): the request data, in the format of dict or string
        context (Context): an object containing request and configuration details

    Returns:
        (dict, string): data to return to client, response content type
    """
    processed_input = input_handler(data, context)
    processed_output = _post_request(processed_input, context)
    return str(processed_output['outputs'])


def _post_request(data, context):
    response = requests.post(context.rest_uri, data=data['data'], headers=data['headers'])

    processed_output, _ = output_handler(response, context)
    if 'error' in processed_output:
        raise Exception('Invocation is not successful.')

    return processed_output


def _make_headers(context):
    headers = {
        'Content-Type': context.request_content_type,
        'Accept': context.accept_header,
        'X-Amzn-SageMaker-Custom-Attributes': _parse_custom_attributes(context)
    }
    return headers


def _parse_custom_attributes(context):
    attribute_list = []
    if context.custom_attributes:
        attribute_list = re.findall("(tfs-[a-z\-]+=[^,]+)", context.custom_attributes)
    model_name = False
    method = False
    model_version = False
    for attribute in attribute_list:
        k, _ = attribute.split('=')
        if k == 'tfs-model-name':
            model_name = True
        if k == 'tfs-method':
            method = True
        if k == 'tfs-model-version':
            model_version = True

    if not model_name:
        attribute_list.append('tfs-model-name={}'.format(context.model_name or DEFAULT_TFS_MODEL))
    if not method:
        attribute_list.append('tfs-method={}'.format(context.method or DEFAULT_TFS_METHOD))
    if not model_version and context.model_version:
        attribute_list.append('tfs-model-version={}'.format(context.model_version))

    return ','.join(attribute_list)


def _parse_json(data):
    data_str = ''
    data = str(data)

    if isinstance(data, dict):
        inputs = data[next(iter(data))]
        if isinstance(inputs, list):
            data_str = str(inputs)
        else:
            data_str = inputs
    elif isinstance(data, str):
        multi_lines = False
        for line in data.splitlines():
            if multi_lines:
                data_str += ','
            data_str += '[' + re.findall(r'\[([^\]]+)', line)[0] + ']'
            multi_lines = '\n' in data
        if multi_lines:
            data_str = '[' + data_str + ']'
    return '{\"inputs\": ' + data_str + '}'


def _parse_jsonlines(data):
    data_str = ''
    data = data.decode("utf-8")
    multi_lines = False
    for line in data.splitlines():
        if multi_lines:
            data_str += ','
        data_str += line
        multi_lines = '\n' in data
    if multi_lines:
        data_str = '[' + data_str + ']'
    return '{\"inputs\": ' + data_str + '}'


def _parse_csv(data):
    json = ''
    data = data.decode("utf-8")
    multi_lines = False
    for line in data.splitlines():
        if multi_lines:
            json += ','
        json += '[' + line + ']'
        multi_lines = '\n' in data

    if multi_lines:
        json = '[' + json + ']'

    return '{\"inputs\": ' + json + '}'


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
