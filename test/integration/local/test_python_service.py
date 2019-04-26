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

import os
import shutil
import subprocess
import sys
import time

import pytest

import requests


PING_URL = 'http://localhost:8080/ping'
INVOCATIONS_URL = 'http://localhost:8080/invocations'


@pytest.fixture(scope='session', autouse=True, params=['1', '2', '3', '4', '5'])
def volume(tmpdir_factory, request):
    try:
        model_dir = os.path.join(tmpdir_factory.mktemp('test'), 'model')
        code_dir = os.path.join(model_dir, 'code')
        test_example = 'test/resources/examples/test{}'.format(request.param)

        models_dir = 'test/resources/models'
        shutil.copytree(models_dir, model_dir)
        shutil.rmtree(code_dir)  # clear existing /code dir
        shutil.copytree(test_example, code_dir)

        subprocess.check_call('ls -lR {}'.format(model_dir).split())

        subprocess.check_call(
            'docker volume create --name model_inference_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume prune -f'.split())


@pytest.fixture(scope='module', autouse=True, params=['1.11', '1.12'])
def container(request):
    try:
        command = (
            'docker run --name sagemaker-tensorflow-serving-test -p 8080:8080'
            ' --mount type=volume,source=model_inference_volume,target=/opt/ml/model'
            ' -e SAGEMAKER_TFS_DEFAULT_MODEL_NAME=half_plus_three'
            ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
            ' -e SAGEMAKER_BIND_TO_PORT=8080'
            ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
            ' sagemaker-tensorflow-serving:{}-cpu serve'
        ).format(request.param)

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


def make_headers(content_type, method):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes': 'tfs-model-name=half_plus_three,tfs-method=%s' % method
    }
    return headers


def test_predict_json():
    headers = make_headers('application/json', 'predict')
    data = '{"instances": [1.0, 2.0, 5.0]}'
    response = requests.post(INVOCATIONS_URL, data=data, headers=headers).json()
    assert response == {'predictions': [3.5, 4.0, 5.5]}


def test_zero_content():
    headers = make_headers('application/json', 'predict')
    data = ''
    response = requests.post(INVOCATIONS_URL, data=data, headers=headers)
    assert response.status_code == 500
    assert 'Internal Server Error' in response.text


def test_large_input():
    headers = make_headers('text/csv', 'predict')
    data_file = 'test/resources/inputs/test-large.csv'

    with open(data_file, 'r') as file:
        large_data = file.read()
        response = requests.post(INVOCATIONS_URL, data=large_data, headers=headers).json()
        predictions = response['predictions']
        assert len(predictions) == 753936


def test_csv_input():
    headers = make_headers('text/csv', 'predict')
    data = '1.0,2.0,5.0'
    response = requests.post(INVOCATIONS_URL, data=data, headers=headers).json()
    assert response == {'predictions': [3.5, 4.0, 5.5]}


def test_unsupported_content_type():
    headers = make_headers('unsupported-type', 'predict')
    data = 'aW1hZ2UgYnl0ZXM='
    with pytest.raises(ValueError, message='Error: 415, Unsupported content type "unsupported-type"'):
        requests.post(INVOCATIONS_URL, data=data, headers=headers).json()


def test_ping_service():
    response = requests.get(PING_URL)
    assert response.status_code == 200
