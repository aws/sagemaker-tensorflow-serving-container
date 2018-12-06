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
import os
import subprocess
import sys
import time

import pytest
import requests

BASE_URL = 'http://localhost:8080/invocations'


@pytest.fixture(scope='module', autouse=True, params=['1.11', '1.12'])
def container(request):
    model_dir = os.path.abspath('test/resources/models')
    command = 'docker run --name sagemaker-tensorflow-serving-test -v {}:/opt/ml/model:ro -p 8080:8080'.format(
        model_dir)
    command += ' -e SAGEMAKER_TFS_DEFAULT_MODEL_NAME=half_plus_three'
    command += ' -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info'
    command += ' -e SAGEMAKER_BIND_TO_PORT=8080'
    command += ' -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999'
    command += ' sagemaker-tensorflow-serving:{}-cpu serve'.format(request.param)
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
    subprocess.check_call('docker rm -f sagemaker-tensorflow-serving-test'.split())


def make_request(data, content_type='application/json', method='predict'):
    headers = {
        'Content-Type': content_type,
        'X-Amzn-SageMaker-Custom-Attributes':
            'tfs-model-name=half_plus_three,tfs-method=%s' % method
    }
    response = requests.post(BASE_URL, data=data, headers=headers)
    return json.loads(response.content.decode('utf-8'))


def test_predict():
    x = {
        'instances': [1.0, 2.0, 5.0]
    }

    y = make_request(json.dumps(x))
    assert y == {'predictions': [3.5, 4.0, 5.5]}


def test_predict_two_instances():
    x = {
        'instances': [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0]]
    }

    y = make_request(json.dumps(x))
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsons_json_content_type():
    x = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    y = make_request(x)
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsonlines():
    x = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    y = make_request(x, 'application/jsonlines')
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_jsons():
    x = '[1.0, 2.0, 5.0]\n[1.0, 2.0, 5.0]'
    y = make_request(x, 'application/jsons')
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}

def test_predict_jsons_2():
    x = '{"x": [1.0, 2.0, 5.0]}\n{"x": [1.0, 2.0, 5.0]}'
    y = make_request(x)
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_generic_json():
    x = [1.0, 2.0, 5.0]
    y = make_request(json.dumps(x))
    assert y == {'predictions': [[3.5, 4.0, 5.5]]}


def test_predict_generic_json_two_instances():
    x = [[1.0, 2.0, 5.0], [1.0, 2.0, 5.0]]
    y = make_request(json.dumps(x))
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_predict_csv():
    x = '1.0, 2.0, 5.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [3.5, 4.0, 5.5]}


def test_predict_csv_two_instances():
    x = '1.0, 2.0, 5.0\n1.0, 2.0, 5.0'
    y = make_request(x, 'text/csv')
    assert y == {'predictions': [[3.5, 4.0, 5.5], [3.5, 4.0, 5.5]]}


def test_regress():
    x = {
        'signature_name': 'tensorflow/serving/regress',
        'examples': [{'x': 1.0}, {'x': 2.0}]
    }

    y = make_request(json.dumps(x), method='regress')
    assert y == {'results': [3.5, 4.0]}


def test_regress_one_instance():
    # tensorflow serving docs indicate response should have 'result' key,
    # but it is actually 'results'
    # this test will catch if they change api to match docs (unlikely)
    x = {
        'signature_name': 'tensorflow/serving/regress',
        'examples': [{'x': 1.0}]
    }

    y = make_request(json.dumps(x), method='regress')
    assert y == {'results': [3.5]}


def test_predict_bad_input():
    y = make_request('whatever')
    assert 'error' in y


def test_predict_bad_input_instances():
    x = json.dumps({'junk': 'data'})
    y = make_request(x)
    assert y['error'].startswith('Failed to process element: 0 key: junk of \'instances\' list.')
