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
import io
import json
import logging
import time

import boto3
import numpy as np

import pytest

EI_SUPPORTED_REGIONS = ['us-east-1', 'us-east-2', 'us-west-2', 'eu-west-1', 'ap-northeast-1', 'ap-northeast-2']

logger = logging.getLogger(__name__)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)
logging.getLogger('session.py').setLevel(logging.DEBUG)
logging.getLogger('functional').setLevel(logging.DEBUG)


@pytest.fixture(autouse=True)
def skip_if_no_accelerator(accelerator_type):
    if accelerator_type is None:
        pytest.skip('Skipping because accelerator type was not provided')


@pytest.fixture(autouse=True)
def skip_if_non_supported_ei_region(region):
    if region not in EI_SUPPORTED_REGIONS:
        pytest.skip('EI is not supported in {}'.format(region))


@pytest.fixture
def pretrained_model_data(region):
    return 's3://sagemaker-sample-data-{}/tensorflow/model/resnet/resnet_50_v2_fp32_NCHW.tar.gz'.format(region)


def _timestamp():
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def _execution_role(session):
    return session.resource('iam').Role('SageMakerRole').arn


def _production_variants(model_name, instance_type, accelerator_type):
    production_variants = [{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': instance_type,
        'AcceleratorType': accelerator_type
    }]
    return production_variants


def _create_model(session, client, docker_image_uri, pretrained_model_data):
    model_name = 'test-tfs-ei-model-{}'.format(_timestamp())
    client.create_model(ModelName=model_name,
                        ExecutionRoleArn=_execution_role(session),
                        PrimaryContainer={
                            'Image': docker_image_uri,
                            'ModelDataUrl': pretrained_model_data
                        })


def _create_endpoint(client, endpoint_config_name, endpoint_name, model_name, instance_type, accelerator_type):
    client.create_endpoint_config(EndpointConfigName=endpoint_config_name,
                                  ProductionVariants=_production_variants(model_name, instance_type, accelerator_type))

    client.create_endpoint(EndpointName=endpoint_name,
                           EndpointConfigName=endpoint_config_name)

    logger.info('deploying model to endpoint: {}'.format(endpoint_name))

    try:
        client.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
    finally:
        status = client.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
        if status != 'InService':
            logger.error('failed to create endpoint: {}'.format(endpoint_name))
            raise Exception('Failed to create endpoint.')


@pytest.mark.skip_if_non_supported_ei_region
@pytest.mark.skip_if_no_accelerator
def test_deploy_elastic_inference_with_pretrained_model(pretrained_model_data,
                                                        docker_image_uri,
                                                        instance_type,
                                                        accelerator_type):
    endpoint_name = 'test-tfs-ei-deploy-model-{}'.format(_timestamp())
    endpoint_config_name = 'test-tfs-endpoint-config-{}'.format(_timestamp())
    model_name = 'test-tfs-ei-model-{}'.format(_timestamp())

    session = boto3.Session()
    client = session.client('sagemaker')
    runtime_client = session.client('runtime.sagemaker')

    _create_model(session, client, docker_image_uri, pretrained_model_data)
    _create_endpoint(client, endpoint_config_name, endpoint_name, model_name, instance_type, accelerator_type)

    input_data = {'instances': np.random.rand(1, 1, 3, 3).tolist()}

    try:
        response = runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                                  ContentType='application/json',
                                                  Body=json.dumps(input_data))
        result = json.loads(response['Body'].read().decode())
        assert result['predictions'] is not None
    finally:
        logger.info('deleting endpoint, endpoint config and model.')
        client.delete_endpoint(EndpointName=endpoint_name)
        client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        client.delete_model(ModelName=model_name)
