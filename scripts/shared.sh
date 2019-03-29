#!/bin/bash
#
# Utility functions for build/test scripts.

function error() {
    >&2 echo $1
    >&2 echo "usage: $0 [--version <major-version>] [--arch (cpu*|gpu|eia)] [--region <aws-region>]"
    exit 1
}

function get_default_region() {
    if [ -n "${AWS_DEFAULT_REGION:-}" ]; then
        echo "$AWS_DEFAULT_REGION"
    else
        aws configure get region
    fi
}

function get_full_version() {
    echo $1 | sed 's#^\([0-9][0-9]*\.[0-9][0-9]*\)$#\1.0#'
}

function get_short_version() {
    echo $1 | sed 's#\([0-9][0-9]*\.[0-9][0-9]*\)\.[0-9][0-9]*#\1#'
}

function get_aws_account() {
    aws sts get-caller-identity --query 'Account' --output text
}

function get_tfs_executable() {
    if [[ -z $(aws s3 ls 's3://amazonei-tensorflow/Tensorflow Serving/v'${short_version}'/Ubuntu/') ]]; then
        echo 'ERROR: cannot find this version in S3 bucket.'
        exit 1
    fi

    zip_file=$(aws s3 ls 's3://amazonei-tensorflow/Tensorflow Serving/v'${short_version}'/Ubuntu/' | awk '{print $4}')
    aws s3 cp 's3://amazonei-tensorflow/Tensorflow Serving/v'${short_version}'/Ubuntu/'${zip_file} .

    mkdir exec_dir
    unzip ${zip_file} -d exec_dir

    find . -name amazonei_tensorflow_model_server -exec mv {} container/ \;
    rm ${zip_file} && rm -rf exec_dir
}

function get_device_type() {
    if [ $1 = 'eia' ]; then
        echo 'cpu'
    else
        echo $1
    fi
}

function parse_std_args() {
    # defaults
    arch='cpu'
    version='1.12.0'
    repository='sagemaker-tensorflow-serving'

    aws_region=$(get_default_region)
    aws_account=$(get_aws_account)

    while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -v|--version)
        version="$2"
        shift
        shift
        ;;
        -a|--arch)
        arch="$2"
        shift
        shift
        ;;
        -r|--region)
        aws_region="$2"
        shift
        shift
        ;;
        *) # unknown option
        error "unknown option: $1"
        shift
        ;;
    esac
    done

    [[ -z "${version// }" ]] && error 'missing version'
    [[ "$arch" =~ ^(cpu|gpu|eia)$ ]] || error "invalid arch: $arch"
    [[ -z "${aws_region// }" ]] && error 'missing aws region'

    [[ "$arch" = eia ]] && repository=$repository'-'$arch

    full_version=$(get_full_version $version)
    short_version=$(get_short_version $version)
    device=$(get_device_type $arch)

    true
}


