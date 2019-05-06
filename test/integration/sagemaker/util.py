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
import contextlib
import json
import logging
import random
import time
import os

import botocore

logger = logging.getLogger(__name__)


def image_uri(registry, region, repo, tag):
    return f'{registry}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}'


def _execution_role(boto_session):
    return boto_session.resource('iam').Role('SageMakerRole').arn


@contextlib.contextmanager
def sagemaker_model(region, boto_session, sagemaker_client, image_uri, model_name, model_data):
    model = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=_execution_role(boto_session),
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data
        })

    try:
        yield model
    finally:
        logger.info('deleting model %s', model_name)
        sagemaker_client.delete_model(ModelName=model_name)


def _production_variants(model_name, instance_type, accelerator_type):
    production_variants = [{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': instance_type
    }]

    if accelerator_type:
        production_variants[0]['AcceleratorType'] = accelerator_type

    return production_variants

def find_or_put_model_data(region, boto_session, local_path):
    model_file = os.path.basename(local_path)

    account = boto_session.client('sts').get_caller_identity()['Account']
    bucket = f'sagemaker-{region}-{account}'
    key = f'test-tfs/{model_file}'

    s3 = boto_session.client('s3')

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
        s3.upload_file(local_path, bucket, key)

    return f's3://{bucket}/{key}'


@contextlib.contextmanager
def sagemaker_endpoint(sagemaker_client, model_name, instance_type, accelerator_type=None):
    logger.info('creating endpoint %s', model_name)

    # Add jitter so we can run tests in parallel without running into service side limits.
    delay = round(random.random()*5, 3)
    logger.info('waiting for {} seconds'.format(delay))
    time.sleep(delay)

    production_variants = _production_variants(model_name, instance_type, accelerator_type)
    sagemaker_client.create_endpoint_config(EndpointConfigName=model_name,
                                            ProductionVariants=production_variants)

    sagemaker_client.create_endpoint(EndpointName=model_name, EndpointConfigName=model_name)

    try:
        sagemaker_client.get_waiter('endpoint_in_service').wait(EndpointName=model_name)
    finally:
        status = sagemaker_client.describe_endpoint(EndpointName=model_name)['EndpointStatus']
        if status != 'InService':
            raise ValueError(f'failed to create endpoint {model_name}')

    try:
        yield model_name  # return the endpoint name
    finally:
        logger.info('deleting endpoint and endpoint config %s', model_name)
        sagemaker_client.delete_endpoint(EndpointName=model_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=model_name)


def invoke_endpoint(sagemaker_runtime_client, endpoint_name, input_data):
    response = sagemaker_runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                                        ContentType='application/json',
                                                        Body=json.dumps(input_data))
    result = json.loads(response['Body'].read().decode())
    assert result['predictions'] is not None
    return result


def create_and_invoke_endpoint(region, boto_session, sagemaker_client, sagemaker_runtime_client,
                               model_name, model_data, image_uri, instance_type, accelerator_type,
                               input_data):
    with sagemaker_model(region, boto_session, sagemaker_client, image_uri, model_name, model_data):
        with sagemaker_endpoint(sagemaker_client, model_name, instance_type, accelerator_type):
            return invoke_endpoint(sagemaker_runtime_client, model_name, input_data)
