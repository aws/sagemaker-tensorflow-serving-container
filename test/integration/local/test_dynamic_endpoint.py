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
import subprocess
import sys
import time

import pytest
import requests

INVOCATION_URL = 'http://localhost:8080/invocations'
MODELS_URL = 'http://localhost:8080/models'


@pytest.fixture(scope='session', autouse=True)
def volume():
    try:
        model_dir = os.path.abspath('test/resources/models')
        subprocess.check_call(
            'docker volume create --name dynamic_endpoint_model_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume rm dynamic_endpoint_model_volume'.split())


@pytest.fixture(scope='module', autouse=True)
def container(request, docker_base_name, tag, runtime_config):
    try:
        command = (
            'docker run {}--name sagemaker-tensorflow-serving-test -p 8080:8080'
            ' --mount type=volume,source=dynamic_endpoint_model_volume,target=/opt/ml/models,readonly'
            ' -e SAGEMAKER_TFS_DEFAULT_MODEL_NAME=half_plus_three'
            ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' -e SAGEMAKER_TFS_ENABLE_DYNAMIC_ENDPOINT=true'
            ' {}:{} serve'
        ).format(runtime_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 5:
            time.sleep(3)
            try:
                requests.get('http://localhost:8080/ping')
                break
            except:
                attempts += 1
                pass

        yield proc.pid
    finally:
        subprocess.check_call('docker rm -f sagemaker-tensorflow-serving-test'.split())


def make_invocation_request(data, model_name, content_type='application/json'):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes':
            'tfs-model-name={},tfs-method=predict'.format(model_name)
    }
    response = requests.post(INVOCATION_URL, data=data, headers=headers)
    return json.loads(response.content.decode('utf-8'))


def make_load_model_request(data, content_type='application/json'):
    headers = {
        'Content-Type': content_type
    }
    response = requests.post(MODELS_URL, data=data, headers=headers)
    return response.content.decode('utf-8')


def test_container_start_invocation_fail():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    y = make_invocation_request(json.dumps(x), 'half_plus_three')
    assert y['error'].startswith('Servable not found for request')


def test_load_one_model():
    model_name = 'half_plus_three'
    model_data = {
        'name': model_name,
        'uri': '/opt/ml/models/half_plus_three'
    }
    res = make_load_model_request(json.dumps(model_data))
    assert res == 'Successfully loaded model {}'.format(model_name)

    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    y = make_invocation_request(json.dumps(x), model_name)
    assert y == {'predictions': [3.5, 4.0, 5.5]}


def test_load_two_models():
    model_name_1 = 'half_plus_two'
    model_data_1 = {
        'name': model_name_1,
        'uri': '/opt/ml/models/half_plus_two'
    }
    res1 = make_load_model_request(json.dumps(model_data_1))
    assert res1 == 'Successfully loaded model {}'.format(model_name_1)

    # load second model
    command = 'curl -d\'{"name":"half_plus_three", "uri":"/opt/ml/models/half_plus_three"}\'' \
              ' -X POST http://localhost:8080/models'
    subprocess.check_call(command.split())

    # make invocation request to the first model
    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    y1 = make_invocation_request(json.dumps(x), model_name_1)
    assert y1 == {'predictions': [2.5, 3.0, 4.5]}

    # make invocation request to the second model
    y2 = make_invocation_request(json.dumps(x), 'half_plus_three')
    assert y2 == {'predictions': [3.5, 4.0, 5.5]}


def test_inference_unloaded_model():
    model_name = 'cifar'
    model_data = {
        'name': model_name,
        'uri': '/opt/ml/models/cifar'
    }
    res = make_load_model_request(json.dumps(model_data))
    assert res == 'Successfully loaded model {}'.format(model_name)

    # make invovation request to unloaded model
    x = {
        'instances': [1.0, 2.0, 5.0]
    }
    y = make_invocation_request(json.dumps(x), 'unloaded_model')
    assert y['error'].startswith('Servable not found for request')
