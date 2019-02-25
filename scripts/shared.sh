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

function parse_std_args() {
    # defaults
    arch='cpu'
    version='1.12.0'
    model='AmazonEI_TensorFlow_Serving_v1.12_v1'

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
        -m|--model)
        model="$2"
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


