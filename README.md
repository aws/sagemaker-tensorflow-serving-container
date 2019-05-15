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

You can also bring in external dependencies to help with your data processing. There are 2 ways to do this:
1. If your model archive contains `code/requirements.txt`, the container will install the python dependencies at runtime using `pip install -r`.
2. If you have pre-installed modules under `code/lib` directory in your model archive, the container will add the modules to the Python path. Note that if both `code/lib` and `code/requirements.txt` are present in the model archive, the `requirements.txt` will be ignored.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/CONTRIBUTING.md) 
for details on our code of conduct, and the process for submitting pull requests to us.

## License

This library is licensed under the Apache 2.0 License. 

