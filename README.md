# SageMaker TensorFlow Serving Container

SageMaker TensorFlow Serving Container is an a open source project that builds 
docker images for running TensorFlow Serving on 
[Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/).

This documentation covers building and testing these docker images. 

For information about using TensorFlow Serving on SageMaker, see: 
[Deploying to TensorFlow Serving Endpoints](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst)
in the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) documentation.

For notebook examples, see: [Amazon SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Building your image](#building-your-image)
3. [Running the tests](#running-the-tests)
4. [Pre/Post-Processing](#pre/post-processing)
5. [Packaging SageMaker Models for TensorFlow Serving](#packaging-sagemaker-models-for-tensorflow-serving)

## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your 
development machine:

- [Docker](https://www.docker.com/)
- [AWS CLI](https://aws.amazon.com/cli/)

For testing, you will also need:

- [Python 3.6](https://www.python.org/)
- [tox](https://tox.readthedocs.io/en/latest/)
- [npm](https://npmjs.org/)
- [jshint](https://jshint.com/about/)

To test GPU images locally, you will also need:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

**Note:** Some of the build and tests scripts interact with resources in your AWS account. Be sure to 
set your default AWS credentials and region using `aws configure` before using these scripts. 

## Building your image

Amazon SageMaker uses Docker containers to run all training jobs and inference endpoints.

The Docker images are built from the Dockerfiles in 
[docker/](https://github.com/aws/sagemaker-tensorflow-serving-container/tree/master/docker>).

The Dockerfiles are grouped based on the version of TensorFlow Serving they support. Each supported
processor type (e.g. "cpu", "gpu", "ei") has a different Dockerfile in each group.  

To build an image, run the `./scripts/build.sh` script:

```bash
./scripts/build.sh --version 1.11 --arch cpu
./scripts/build.sh --version 1.11 --arch gpu
./scripts/build.sh --version 1.11 --arch eia
```


If your are testing locally, building the image is enough. But if you want to your updated image
in SageMaker, you need to publish it to an ECR repository in your account. The 
`./scripts/publish.sh` script makes that easy:
 
```bash
./scripts/publish.sh --version 1.11 --arch cpu
./scripts/publish.sh --version 1.11 --arch gpu
./scripts/publish.sh --version 1.11 --arch eia
```

Note: this will publish to ECR in your default region. Use the `--region` argument to 
specify a different region.

### Running your image in local docker

You can also run your container locally in Docker to test different models and input 
inference requests by hand. Standard `docker run` commands (or `nvidia-docker run` for 
GPU images) will work for this, or you can use the provided `start.sh` 
and `stop.sh` scripts:

```bash
./scripts/start.sh [--version x.xx] [--arch cpu|gpu|eia|...]
./scripts/stop.sh [--version x.xx] [--arch cpu|gpu|eia|...]
```

When the container is running, you can send test requests to it using any HTTP client. Here's
and an example using the `curl` command:

```bash
curl -X POST --data-binary @test/resources/inputs/test.json \
     -H 'Content-Type: application/json' \ 
     -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=half_plus_three' \ 
     http://localhost:8080/invocations
```

Additional `curl` examples can be found in `./scripts/curl.sh`. 

## Running the tests

The package includes automated tests and code checks. The tests use Docker to run the container 
image locally, and do not access resources in AWS. You can run the tests and static code 
checkers using `tox`:

```bash
tox
```

To run local tests against a single container or with other options, you can use the following command:

```bash
python -m pytest test/integration/local 
    [--docker-name-base <docker_name_base>]
    [--framework-version <framework_version>]
    [--processor-type <processor_type>]
    [--enable-batch <ues_bactch>]
```

To test against Elastic Inference with Accelerator, you will need an AWS account, publish your built image to ECR repository and run the following command:

    tox -e py36 -- test/integration/sagemaker/test_ei.py
        [--repo <ECR_repository_name>]
        [--instance-types <instance_type>,...]
        [--accelerator-type <accelerator_type>]
        [--versions <version>,...]

For example:
    
    tox -e py36 -- test/integration/sagemaker/test_ei.py \
        --repo sagemaker-tensorflow-serving-eia \
        --instance_type ml.m5.xlarge \
        --accelerator-type ml.eia1.medium \
        --versions 1.12.0

## Pre/Post-Processing

You can add your customized Python code to process your input and output data. To make it work, here are some few things you need to pay attention:
1. The customized Python code file should be named `inference.py` and it should be under `code` directory of your model archive.
2. `inference.py` should implement either a pair of `input_handler` and `output_handler` functions or a single `handler` function. Note that if `handler` function is implemented, `input_handler` and `output_handler` will be ignored.

To implement pre/post-processing handler(s), you will need to make use of the `Context` object created by Python service. The `Context` is a `namedtuple` with following attributes:
- `model_name (string)`: the name of the model you will to use for inference, for example 'half_plus_three'
- `model_version (string)`: version of the model, for example '5'
- `method (string)`: inference method, for example, 'predict', 'classify' or 'regress', for more information on methods, please see [Classify and Regress API](https://www.tensorflow.org/tfx/serving/api_rest#classify_and_regress_api) and [Predict API](https://www.tensorflow.org/tfx/serving/api_rest#predict_api)
- `rest_uri (string)`: the TFS REST uri generated by the Python service, for example, 'http://localhost:8501/v1/models/half_plus_three:predict'
- `grpc_uri (string)`: the GRPC port number generated by the Python service, for example, '9000'
- `custom_attributes (string)`: content of 'X-Amzn-SageMaker-Custom-Attributes' header from the original request, for example, 'tfs-model-name=half_plus_three,tfs-method=predict'
- `request_content_type (string)`: the original request content type, defaulted to 'application/json' if not provided
- `accept_header (string)`: the original request accept type, defaulted to 'application/json' if not provided
- `content_length (int)`: content length of the original request

Here's a code example of implementing `input_handler` and `output_handler`. By providing these, the Python service will post the request to TFS REST uri with the data pre-processed by `input_handler` and pass the response to `output_handler` for post-processing.

```python
import json

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
```

There are occasions when you might want to have complete controls over request. For example, making TFS request (REST or GRPC) to first model, inspecting results and making request to a second model. In this case, you may implement the `handler` instead of the `input_handler` and `output_handler` pair:

```python
import json
import requests

    
def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _process_input(data, context):
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
```
    
You can also bring in external dependencies to help with your data processing. There are 2 ways to do this:
1. If your model archive contains `code/requirements.txt`, the container will install the Python dependencies at runtime using `pip install -r`.
2. If you are working in a network-isolation situation or if you don't want to install dependencies at runtime everytime your Endpoint starts or Batch Transform job runs, you may want to put pre-downloaded dependencies under `code/lib` directory in your model archive, the container will then add the modules to the Python path. Note that if both `code/lib` and `code/requirements.txt` are present in the model archive, the `requirements.txt` will be ignored.

Your untarred model directory structure may look like this if you are using `requirements.txt`:

        model1
            |__[model_version_number]
                |__variables
                |__saved_model.pb
        model2
            |__[model_version_number]
                |__assets
                |__variables
                |__saved_model.pb
        code
            |__inference.py
            |__requirements.txt

Your untarred model directory structure may look like this if you have downloaded modules under `code/lib`:

        model1
            |__[model_version_number]
                |__variables
                |__saved_model.pb
        model2
            |__[model_version_number]
                |__assets
                |__variables
                |__saved_model.pb
        code
            |__lib
                |__external_module
            |__inference.py

### Packaging SageMaker Models for TensorFlow Serving

If you are not using SageMaker Python SDK, you should package the contents in model directory (including models, inference.py and external modules) in .tar.gz format in a file named "model.tar.gz" and upload it to S3. If you're on a Unix-based operating system, you can create a "model.tar.gz" using the `tar` utility:

```
tar -czvf model.tar.gz 12345 code
```

where "12345" is your TensorFlow serving model version which contains your SavedModel.

After uploading your `model.tar.gz` to an S3 URI, such as `s3://your-bucket/your-models/model.tar.gz`, create a [SageMaker Model](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html) which will be used to generate inferences. Set `PrimaryContainer.ModelDataUrl` to the S3 URI where you uploaded the `model.tar.gz`, and set `PrimaryContainer.Image` to an image following this format:

```
520713654638.dkr.ecr.{REGION}.amazonaws.com/sagemaker-tensorflow-serving:{TENSORFLOW_SERVING_VERSION}-{cpu|gpu|eia}
```

Where `REGION` is your AWS region, such as "us-east-1" or "eu-west-1"; `TENSORFLOW_SERVING_VERSION` is one of the supported versions: "1.11" or "1.12"; and "gpu" for use on GPU-based instance types like ml.p3.2xlarge, or "cpu" for use on CPU-based instances like `ml.c5.xlarge`.


After creating your SageMaker Model, you can use it to create [SageMaker Batch Transform Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html)
 for offline inference, or create [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html) for real-time inference.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/CONTRIBUTING.md) 
for details on our code of conduct, and the process for submitting pull requests to us.

## License

This library is licensed under the Apache 2.0 License. 

