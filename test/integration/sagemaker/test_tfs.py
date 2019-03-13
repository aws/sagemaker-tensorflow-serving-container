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

import boto3
import botocore
import pytest

import util

NON_P3_REGIONS = ['ap-southeast-1', 'ap-southeast-2', 'ap-south-1',
                  'ca-central-1', 'eu-west-2', 'us-west-1']


@pytest.fixture(params=os.environ['TEST_VERSIONS'].split(','))
def version(request):
    return request.param


@pytest.fixture(scope='session')
def repo(request):
    return request.config.getoption('--repo') or 'sagemaker-tensorflow-serving'


@pytest.fixture
def tag(request, version, instance_type):
    if request.config.getoption('--tag'):
        return request.config.getoption('--tag')

    arch = 'gpu' if instance_type.startswith('ml.p') else 'cpu'
    return f'{version}-{arch}'


@pytest.fixture
def image_uri(registry, region, repo, tag):
    return util.image_uri(registry, region, repo, tag)


@pytest.fixture(params=os.environ['TEST_INSTANCE_TYPES'].split(','))
def instance_type(request, region):
    return request.param

@pytest.fixture(scope='module')
def accelerator_type():
    return None

@pytest.fixture(scope='session')
def model_data(region):
    account = boto3.client('sts').get_caller_identity()['Account']
    bucket = f'sagemaker-{region}-{account}'
    key = 'test-tfs/test-model.tar.gz'

    s3 = boto3.client('s3')

    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise

        # bucket doesn't exist, create it
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={'LocationConstraint': region})


    try:
        s3.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise

        # file doesn't exist - upload it
        file = 'test/data/test-model.tar.gz'
        s3.upload_file(file, bucket, key)

    return f's3://{bucket}/{key}'


@pytest.fixture
def input_data():
    return {'instances': [1.0, 2.0, 5.0]}


@pytest.fixture
def skip_if_p3_in_unsupported_region(region, instance_type):
    if 'p3' in instance_type and region in NON_P3_REGIONS:
        pytest.skip('Skipping because accelerator type was not provided')


@pytest.mark.skip_if_p3_in_unsupported_region()
def test_invoke_endpoint(region, boto_session, sagemaker_client, sagemaker_runtime_client,
                         model_name, model_data, image_uri, instance_type, accelerator_type,
                         input_data):
    util.create_and_invoke_endpoint(region, boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, model_data, image_uri,
                                    instance_type, accelerator_type, input_data)
