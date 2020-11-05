# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import unittest

from docker.build_artifacts import deep_learning_container as deep_learning_container_to_test
import pytest
import requests


@pytest.fixture(name='fixture_valid_instance_id')
def fixture_valid_instance_id(requests_mock):
    """
    Fixture id of a valid instance.

    Args:
        requests_mock: (todo): write your description
    """
    return requests_mock.get('http://169.254.169.254/latest/meta-data/instance-id',
                             text='i-123t32e11s32t1231')


@pytest.fixture(name='fixture_invalid_instance_id')
def fixture_invalid_instance_id(requests_mock):
    """
    Fixture id of an instance.

    Args:
        requests_mock: (todo): write your description
    """
    return requests_mock.get('http://169.254.169.254/latest/meta-data/instance-id', text='i-123')


@pytest.fixture(name='fixture_none_instance_id')
def fixture_none_instance_id(requests_mock):
    """
    Return the instance id of a mock instance.

    Args:
        requests_mock: (todo): write your description
    """
    return requests_mock.get('http://169.254.169.254/latest/meta-data/instance-id', text=None)


@pytest.fixture(name='fixture_invalid_region')
def fixture_invalid_region(requests_mock):
    """
    Retrieve the region of the given requests.

    Args:
        requests_mock: (dict): write your description
    """
    return requests_mock.get('http://169.254.169.254/latest/dynamic/instance-identity/document',
                             json={'region': 'test'})


@pytest.fixture(name='fixture_valid_region')
def fixture_valid_region(requests_mock):
    """
    Retrieve the region details.

    Args:
        requests_mock: (dict): write your description
    """
    return requests_mock.get('http://169.254.169.254/latest/dynamic/instance-identity/document',
                             json={'region': 'us-east-1'})


def test_retrieve_instance_id(fixture_valid_instance_id):
    """
    Retrieves id of a container id

    Args:
        fixture_valid_instance_id: (str): write your description
    """
    result = deep_learning_container_to_test._retrieve_instance_id()
    assert 'i-123t32e11s32t1231' == result


def test_retrieve_none_instance_id(fixture_none_instance_id):
    """
    Retrieves the runtime id is_id id of the container

    Args:
        fixture_none_instance_id: (str): write your description
    """
    result = deep_learning_container_to_test._retrieve_instance_id()
    assert result is None


def test_retrieve_invalid_instance_id(fixture_invalid_instance_id):
    """
    Retrieve the test id of the container id of a valid.

    Args:
        fixture_invalid_instance_id: (str): write your description
    """
    result = deep_learning_container_to_test._retrieve_instance_id()
    assert result is None


def test_retrieve_invalid_region(fixture_invalid_region):
    """
    Determine whether a valid region is in a valid region.

    Args:
        fixture_invalid_region: (str): write your description
    """
    result = deep_learning_container_to_test._retrieve_instance_region()
    assert result is None


def test_retrieve_valid_region(fixture_valid_region):
    """
    Determine if a valid region is valid.

    Args:
        fixture_valid_region: (str): write your description
    """
    result = deep_learning_container_to_test._retrieve_instance_region()
    assert 'us-east-1' == result


def test_query_bucket(requests_mock, fixture_valid_region, fixture_valid_instance_id):
    """
    Returns true if a valid bucket query exists.

    Args:
        requests_mock: (todo): write your description
        fixture_valid_region: (str): write your description
        fixture_valid_instance_id: (str): write your description
    """
    fixture_valid_instance_id.return_value = 'i-123t32e11s32t1231'
    fixture_valid_region.return_value = 'us-east-1'
    requests_mock.get(('https://aws-deep-learning-containers-us-east-1.s3.us-east-1.amazonaws.com'
                       '/dlc-containers.txt?x-instance-id=i-123t32e11s32t1231'),
                      text='Access Denied')
    actual_response = deep_learning_container_to_test.query_bucket()
    assert 'Access Denied' == actual_response.text


def test_query_bucket_region_none(fixture_invalid_region, fixture_valid_instance_id):
    """
    Returns true if the bucket exist.

    Args:
        fixture_invalid_region: (str): write your description
        fixture_valid_instance_id: (str): write your description
    """
    fixture_valid_instance_id.return_value = 'i-123t32e11s32t1231'
    fixture_invalid_region.return_value = None
    actual_response = deep_learning_container_to_test.query_bucket()
    assert actual_response is None


def test_query_bucket_instance_id_none(requests_mock, fixture_valid_region, fixture_none_instance_id):
    """
    Returns true if a bucket id exists.

    Args:
        requests_mock: (todo): write your description
        fixture_valid_region: (str): write your description
        fixture_none_instance_id: (str): write your description
    """
    fixture_none_instance_id.return_value = None
    fixture_valid_region.return_value = 'us-east-1'
    actual_response = deep_learning_container_to_test.query_bucket()
    assert actual_response is None


def test_query_bucket_instance_id_invalid(requests_mock, fixture_valid_region, fixture_invalid_instance_id):
    """
    Returns true if the bucket id is a valid.

    Args:
        requests_mock: (todo): write your description
        fixture_valid_region: (str): write your description
        fixture_invalid_instance_id: (str): write your description
    """
    fixture_invalid_instance_id.return_value = None
    fixture_valid_region.return_value = 'us-east-1'
    actual_response = deep_learning_container_to_test.query_bucket()
    assert actual_response is None


def test_HTTP_error_on_S3(requests_mock, fixture_valid_region, fixture_valid_instance_id):
    """
    Test for http : attributeerror is raised. http : class :.

    Args:
        requests_mock: (dict): write your description
        fixture_valid_region: (str): write your description
        fixture_valid_instance_id: (str): write your description
    """
    fixture_valid_instance_id.return_value = 'i-123t32e11s32t1231'
    fixture_valid_region.return_value = 'us-east-1'
    query_s3_url = ('https://aws-deep-learning-containers-us-east-1.s3.us-east-1.amazonaws.com'
                    '/dlc-containers.txt?x-instance-id=i-123t32e11s32t1231')

    requests_mock.get(
        query_s3_url,
        exc=requests.exceptions.HTTPError)
    requests_mock.side_effect = requests.exceptions.HTTPError

    with pytest.raises(requests.exceptions.HTTPError):
        actual_response = requests.get(query_s3_url)
        assert actual_response is None


def test_connection_error_on_S3(requests_mock, fixture_valid_region, fixture_valid_instance_id):
    """
    Test for connection pool connection to connection.

    Args:
        requests_mock: (dict): write your description
        fixture_valid_region: (str): write your description
        fixture_valid_instance_id: (str): write your description
    """
    fixture_valid_instance_id.return_value = 'i-123t32e11s32t1231'
    fixture_valid_region.return_value = 'us-east-1'
    query_s3_url = ('https://aws-deep-learning-containers-us-east-1.s3.us-east-1.amazonaws.com'
                    '/dlc-containers.txt?x-instance-id=i-123t32e11s32t1231')

    requests_mock.get(
        query_s3_url,
        exc=requests.exceptions.ConnectionError)

    with pytest.raises(requests.exceptions.ConnectionError):
        actual_response = requests.get(
            query_s3_url)

        assert actual_response is None


def test_timeout_error_on_S3(requests_mock, fixture_valid_region, fixture_valid_instance_id):
    """
    Return a valid http error response.

    Args:
        requests_mock: (todo): write your description
        fixture_valid_region: (str): write your description
        fixture_valid_instance_id: (str): write your description
    """
    fixture_valid_instance_id.return_value = 'i-123t32e11s32t1231'
    fixture_valid_region.return_value = 'us-east-1'
    query_s3_url = ('https://aws-deep-learning-containers-us-east-1.s3.us-east-1.amazonaws.com'
                    '/dlc-containers.txt?x-instance-id=i-123t32e11s32t1231')

    requests_mock.get(
        query_s3_url,
        exc=requests.Timeout)

    with pytest.raises(requests.exceptions.Timeout):
        actual_response = requests.get(
            query_s3_url)

        assert actual_response is None


if __name__ == '__main__':
    unittest.main()
