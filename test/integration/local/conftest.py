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

import pytest

FRAMEWORK_LATEST_VERSION = '1.12'
TFS_DOCKER_BASE_NAME = 'sagemaker-tensorflow-serving'


def pytest_addoption(parser):
    parser.addoption('--docker-base-name', default=TFS_DOCKER_BASE_NAME)
    parser.addoption('--framework-version', default=FRAMEWORK_LATEST_VERSION, required=True)
    parser.addoption('--processor', default='cpu')
    parser.addoption('--enable-batch', default='False', choices=['True', 'False'])


@pytest.fixture(scope='module')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='module')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='module')
def processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='module')
def enable_batch(request):
    return request.config.getoption('--enable-batch') == 'True'
