# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import tarfile

import boto3
import pytest
import sagemaker
import urllib.request

import util

from packaging.version import Version

from sagemaker.tensorflow import TensorFlowModel
from sagemaker.utils import name_from_base

from timeout import timeout_and_delete_endpoint

NON_P3_REGIONS = ["ap-southeast-1", "ap-southeast-2", "ap-south-1",
                  "ca-central-1", "eu-central-1", "eu-west-2", "us-west-1"]

RESOURCES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resources"))


@pytest.fixture(params=os.environ["TEST_VERSIONS"].split(","))
def version(request):
    return request.param


@pytest.fixture(scope="session")
def repo(request):
    return request.config.getoption("--repo") or "sagemaker-tensorflow-serving"


@pytest.fixture
def tag(request, version, instance_type):
    if request.config.getoption("--tag"):
        return request.config.getoption("--tag")

    arch = "gpu" if instance_type.startswith("ml.p") else "cpu"
    return f"{version}-{arch}"


@pytest.fixture
def image_uri(registry, region, repo, tag):
    return util.image_uri(registry, region, repo, tag)


@pytest.fixture(params=os.environ["TEST_INSTANCE_TYPES"].split(","))
def instance_type(request, region):
    return request.param


@pytest.fixture(scope="module")
def accelerator_type():
    return None


@pytest.fixture(scope="session")
def tfs_model(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "test/data/tfs-model.tar.gz")


@pytest.fixture(scope='session')
def python_model_with_requirements(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "test/data/python-with-requirements.tar.gz")


@pytest.fixture(scope='session')
def python_model_with_lib(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "test/data/python-with-lib.tar.gz")


@pytest.fixture(scope="session")
def resnet_model_tar_path():
    model_path = os.path.join(RESOURCES_PATH, "models", "resnet50_v1")
    model_tar_path = os.path.join(model_path, "model.tar.gz")
    if os.path.exists(model_tar_path):
        os.remove(model_tar_path)
    s3_resource = boto3.resource("s3")
    models_bucket = s3_resource.Bucket("aws-dlc-sample-models")
    model_s3_location = "tensorflow/resnet50_v1/model"
    for obj in models_bucket.objects.filter(Prefix=model_s3_location):
        local_file = os.path.join(model_path, "model", os.path.relpath(obj.key, model_s3_location))
        if not os.path.isdir(os.path.dirname(local_file)):
            os.makedirs(os.path.dirname(local_file))
        if obj.key.endswith("/"):
            continue
        models_bucket.download_file(obj.key, local_file)
    with tarfile.open(model_tar_path, "w:gz") as model_tar:
        model_tar.add(os.path.join(model_path, "code"), arcname="code")
        model_tar.add(os.path.join(model_path, "model"), arcname="model")
    return model_tar_path


def test_tfs_model(boto_session, sagemaker_client,
                   sagemaker_runtime_client, model_name, tfs_model,
                   image_uri, instance_type, accelerator_type):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, tfs_model,
                                    image_uri, instance_type, accelerator_type, input_data)


def test_batch_transform(region, boto_session, sagemaker_client,
                         model_name, tfs_model, image_uri,
                         instance_type):
    results = util.run_batch_transform_job(region=region,
                                           boto_session=boto_session,
                                           model_data=tfs_model,
                                           image_uri=image_uri,
                                           model_name=model_name,
                                           sagemaker_client=sagemaker_client,
                                           instance_type=instance_type)
    assert len(results) == 10
    for r in results:
        assert r == [3.5, 4.0, 5.5]


def test_python_model_with_requirements(boto_session, sagemaker_client,
                                        sagemaker_runtime_client, model_name,
                                        python_model_with_requirements, image_uri, instance_type,
                                        accelerator_type):

    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    # the python service needs to transform this to get a valid prediction
    input_data = {"x": [1.0, 2.0, 5.0]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name,
                                                  python_model_with_requirements, image_uri,
                                                  instance_type, accelerator_type, input_data)

    # python service adds this to tfs response
    assert output_data["python"] is True
    assert output_data["pillow"] == "6.0.0"


def test_python_model_with_lib(boto_session, sagemaker_client,
                               sagemaker_runtime_client, model_name, python_model_with_lib,
                               image_uri, instance_type, accelerator_type):

    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    # the python service needs to transform this to get a valid prediction
    input_data = {"x": [1.0, 2.0, 5.0]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name, python_model_with_lib,
                                                  image_uri, instance_type, accelerator_type, input_data)

    # python service adds this to tfs response
    assert output_data["python"] is True
    assert output_data["dummy_module"] == "0.1"


def test_resnet_with_inference_handler(
    boto_session, image_uri, instance_type, resnet_model_tar_path, version
):
    if Version(version) >= Version("2.6"):
        pytest.skip("The inference script currently uses v1 compat features, making it incompatible with TF>=2.6")

    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    model_data = sagemaker_session.upload_data(
        path=resnet_model_tar_path, key_prefix=os.path.join("tensorflow-inference", "resnet")
    )
    endpoint_name = name_from_base("tensorflow-inference")

    tensorflow_model = TensorFlowModel(
        model_data=model_data,
        role="SageMakerRole",
        entry_point="inference.py",
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        tensorflow_predictor = tensorflow_model.deploy(
            initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name
        )
        kitten_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
        kitten_local_path = "kitten.jpg"
        urllib.request.urlretrieve(kitten_url, kitten_local_path)
        with open(kitten_local_path, "rb") as f:
            kitten_image = f.read()

        runtime_client = sagemaker_session.sagemaker_runtime_client
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType='application/x-image', Body=kitten_image
        )
        result = json.loads(response['Body'].read().decode('ascii'))

        assert len(result["outputs"]["probabilities"]["floatVal"]) == 3
