import json
import os
import subprocess
import sys
import time

import pytest
import requests

BASE_URL = "http://localhost:8090/invocations"


@pytest.fixture(scope="module", autouse=True, params=[True, False])
def container(request, docker_base_name, tag, runtime_config):
    try:
        if request.param:
            batching_config = " -e SAGEMAKER_TFS_ENABLE_BATCHING=true"
            port_config = " -e SAGEMAKER_SAFE_PORT_RANGE=9000-9999"
        else:
            batching_config = ""
            port_config = ""
        command = (
            "docker run {}--name sagemaker-tensorflow-serving-test -p 8090:8090"
            " --mount type=volume,source=model_volume,target=/opt/ml/model,readonly"
            " -e SAGEMAKER_TFS_NGINX_LOGLEVEL=info"
            " -e SAGEMAKER_BIND_TO_PORT=8090"
            " -e SAGEMAKER_TFS_INSTANCE_COUNT=8"
            " -e SAGEMAKER_GUNICORN_WORKERS=36"
            " -e SAGEMAKER_TFS_INTER_OP_PARALLELISM=1"
            " -e SAGEMAKER_TFS_INTRA_OP_PARALLELISM=1"
            " {}"
            " {}"
            " {}:{} serve"
        ).format(runtime_config, batching_config, port_config, docker_base_name, tag)

        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0

        while attempts < 40:
            time.sleep(3)
            try:
                res_code = requests.get("http://localhost:8090/ping").status_code
                if res_code == 200:
                    break
            except:
                attempts += 1
                pass

        yield proc.pid
    finally:
        subprocess.check_call("docker rm -f sagemaker-tensorflow-serving-test".split())


def make_request(data, content_type="application/json", method="predict", version=None):
    custom_attributes = "tfs-model-name=half_plus_three,tfs-method={}".format(method)
    if version:
        custom_attributes += ",tfs-model-version={}".format(version)

    headers = {
        "Content-Type": content_type,
        "X-Amzn-SageMaker-Custom-Attributes": custom_attributes,
    }
    response = requests.post(BASE_URL, data=data, headers=headers)
    return json.loads(response.content.decode("utf-8"))


def test_predict():
    x = {
        "instances": [1.0, 2.0, 5.0]
    }

    y = make_request(json.dumps(x))
    assert y == {"predictions": [3.5, 4.0, 5.5]}
