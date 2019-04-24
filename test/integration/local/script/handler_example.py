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
import re
from collections import namedtuple

import requests

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')


def handler(data, context):
    """Handle request.

    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=input)
    return _process_output(response, context)


def _process_input(data, context):
    if context.request_content_type == 'application/json':
        data = _parse_json(data)
    elif context.request_content_type == 'text/csv':
        data = _parse_csv(data)
    else:
        _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))

    return data


def _process_output(data, context):
    if data.status_code != 200:
        raise Exception(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type


def _parse_json(data):
    data = data.read().decode('utf-8')
    if len(data) == 0:
        return data
    data_str = ''
    for line in data.splitlines():
        data_str += re.findall(r'\[([^\]]+)', line)[0]
    return json.dumps({"instances": [float(i) for i in data_str.split(',')]})


def _parse_csv(data):
    data = data.read().decode('utf-8')
    return json.dumps({"instances": [float(x) for x in data.split(',')]})


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
