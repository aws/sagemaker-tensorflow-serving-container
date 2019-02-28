#!/bin/bash
#
# Utility functions for build/test scripts.

function error() {
    >&2 echo $1
    >&2 echo "usage: $0 [--version <major-version>] [--arch (cpu*|gpu|ei)] [--region <aws-region>]"
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
    # default to v1.12 in accordance with defaults below
    s3_object='tfs_ei_v1_12_ubuntu'
    unzipped='v1_12_Ubuntu'

    if [ ${version} = '1.11' ]; then
        s3_object='Ubuntu'
        unzipped='Ubuntu'
    fi

    aws s3 cp 's3://amazonei-tensorflow/Tensorflow Serving/v'${version}'/Ubuntu/'${s3_object}'.zip' .
    unzip ${s3_object} && mv ${unzipped}/AmazonEI_Tensorflow_Serving_v${version}_v1 container/
    rm ${s3_object}.zip && rm -rf ${unzipped}
}

function parse_std_args() {
    # defaults
    arch='cpu'
    version='1.12.0'

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
    [[ "$arch" =~ ^(cpu|gpu|ei)$ ]] || error "invalid arch: $arch"
    [[ -z "${aws_region// }" ]] && error 'missing aws region'

    full_version=$(get_full_version $version)
    short_version=$(get_short_version $version)

    true
}


