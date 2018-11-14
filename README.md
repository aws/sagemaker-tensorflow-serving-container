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

## Getting Started

### Prerequisites

Make sure you have installed all of the following prerequisites on your 
development machine:

- [Docker](https://www.docker.com/)
- [AWS CLI](https://aws.amazon.com/cli/)

For testing, you will also need:

- [Python 3.5+](https://www.python.org/)
- [pytest](https://docs.pytest.org/en/latest/)
- The Python [requests](http://docs.python-requests.org/en/master/) library

To test GPU images locally, you will also need:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

**Note:** Some of the build and tests scripts interact with resources in your AWS account. Be sure to 
set your default AWS credentials and region using `aws configure` before using these scripts. 

## Building your image

Amazon SageMaker uses Docker containers to run all training jobs and inference endpoints.

The Docker images are built from the Dockerfiles in 
[docker/](https://github.com/aws/sagemaker-tensorflow-serving-container/tree/master/docker>).

The Dockerfiles are grouped based on the version of TensorFlow Serving they support. Each supported
processor type (e.g. "cpu", "gpu") has a different Dockerfile in each group.  

To build an image, run the `./scripts/build.sh` script:

```bash
./scripts/build.sh --version 1.11 --arch cpu
./scripts/build.sh --version 1.11 --arch gpu
```


If your are testing locally, building the image is enough. But if you want to your updated image
in SageMaker, you need to publish it to an ECR repository in your account. The 
`./scripts/publish.sh` script makes that easy:
 
```bash
./scripts/publish.sh --version 1.11 --arch cpu
./scripts/publish.sh --version 1.11 --arch gpu
```

Note: this will publish to ECR in your default region. Use the `--region` argument to 
specify a different region.

### Running your image in local docker

You can also run your container locally in Docker to test different models and input 
inference requests by hand. Standard `docker run` commands (or `nvidia-docker run` for 
GPU images) will work for this, or you can use the provided `start.sh` 
and `stop.sh` scripts:

```bash
./scripts/start.sh [--version x.xx] [--arch cpu|gpu|...]
./scripts/stop.sh [--version x.xx] [--arch cpu|gpu|...]
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

The package includes some automated unit and integration tests. These tests use Docker to run
your image locally, and do not access resources in AWS. You can run them using `pytest`:

```bash
pytest ./test
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/CONTRIBUTING.md) 
for details on our code of conduct, and the process for submitting pull requests to us.

## License

This library is licensed under the Apache 2.0 License. 

