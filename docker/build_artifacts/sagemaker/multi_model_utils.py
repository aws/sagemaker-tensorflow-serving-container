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
import fcntl
import signal
import time
from contextlib import contextmanager

MODEL_CONFIG_FILE = "/sagemaker/model-config.cfg"
DEFAULT_LOCK_FILE = "/sagemaker/lock-file.lock"


@contextmanager
def lock(path=DEFAULT_LOCK_FILE):
    """
    A context manager.

    Args:
        path: (str): write your description
        DEFAULT_LOCK_FILE: (str): write your description
    """
    f = open(path, "w")
    fd = f.fileno()
    fcntl.lockf(fd, fcntl.LOCK_EX)

    try:
        yield
    finally:
        time.sleep(1)
        fcntl.lockf(fd, fcntl.LOCK_UN)


@contextmanager
def timeout(seconds=60):
    """
    A context manager.

    Args:
        seconds: (float): write your description
    """
    def _raise_timeout_error(signum, frame):
        """
        Raise an exception if the timeout.

        Args:
            signum: (int): write your description
            frame: (todo): write your description
        """
        raise Exception(408, "Timed out after {} seconds".format(seconds))

    try:
        signal.signal(signal.SIGALRM, _raise_timeout_error)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)


class MultiModelException(Exception):
    def __init__(self, code, msg):
        """
        Initialize code object

        Args:
            self: (todo): write your description
            code: (int): write your description
            msg: (str): write your description
        """
        Exception.__init__(self, code, msg)
        self.code = code
        self.msg = msg
