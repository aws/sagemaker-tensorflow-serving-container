# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

FRAMEWORK_LATEST_VERSION = "1.13"
TFS_DOCKER_BASE_NAME = "sagemaker-tensorflow-serving"


def pytest_addoption(parser):
    """
    Adds a pytest to the pytest.

    Args:
        parser: (todo): write your description
    """
    parser.addoption("--docker-base-name", default=TFS_DOCKER_BASE_NAME)
    parser.addoption("--framework-version", default=FRAMEWORK_LATEST_VERSION, required=True)
    parser.addoption("--processor", default="cpu", choices=["cpu", "gpu"])
    parser.addoption("--tag")


@pytest.fixture(scope="module")
def docker_base_name(request):
    """
    Return the docker name for a docker container.

    Args:
        request: (todo): write your description
    """
    return request.config.getoption("--docker-base-name")


@pytest.fixture(scope="module")
def framework_version(request):
    """
    Return the version of the request.

    Args:
        request: (todo): write your description
    """
    return request.config.getoption("--framework-version")


@pytest.fixture(scope="module")
def processor(request):
    """
    Return the processor for the request.

    Args:
        request: (todo): write your description
    """
    return request.config.getoption("--processor")


@pytest.fixture(scope="module")
def runtime_config(request, processor):
    """
    Return runtime config.

    Args:
        request: (todo): write your description
        processor: (todo): write your description
    """
    if processor == "gpu":
        return "--runtime=nvidia "
    else:
        return ""


@pytest.fixture(scope="module")
def tag(request, framework_version, processor):
    """
    Create a tag.

    Args:
        request: (todo): write your description
        framework_version: (todo): write your description
        processor: (str): write your description
    """
    image_tag = request.config.getoption("--tag")
    if not image_tag:
        image_tag = "{}-{}".format(framework_version, processor)
    return image_tag


@pytest.fixture(autouse=True)
def skip_by_device_type(request, processor):
    """
    Skip the mark marker for a device.

    Args:
        request: (todo): write your description
        processor: (str): write your description
    """
    is_gpu = processor == "gpu"
    if (request.node.get_closest_marker("skip_gpu") and is_gpu) or \
            (request.node.get_closest_marker("skip_cpu") and not is_gpu):
        pytest.skip("Skipping because running on \"{}\" instance".format(processor))
