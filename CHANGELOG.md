# Changelog

## v1.5.0 (2019-06-20)

### Features

 * add 1.13 EIA containers

## v1.4.6 (2019-06-18)

### Bug fixes and other changes

 * fix broken ei tests

## v1.4.5 (2019-06-17)

### Bug fixes and other changes

 * add batch transform integration test

## v1.4.4 (2019-06-12)

### Bug fixes and other changes

 * move SageMaker tests to release build

## v1.4.3 (2019-06-12)

### Bug fixes and other changes

 * use p2.xlarge by default in tests

## v1.4.2 (2019-06-12)

### Bug fixes and other changes

 * add Tensorflow 1.13

## v1.4.1 (2019-06-11)

### Bug fixes and other changes

 * make tox run any pytest tests

## v1.4.0 (2019-06-03)

### Features

 * support jsonlines output

### Documentation changes

 * update README.md for EI image

## v1.3.2 (2019-05-29)

### Bug fixes and other changes

 * change Dockerfile directory structure
 * allow local test against single container

## v1.3.1.post1 (2019-05-20)

### Documentation changes

 * update README.md
 * add pre/post-processing usage examples

## v1.3.1.post0 (2019-05-16)

### Documentation changes

 * add pre-post-processing documentation

## v1.3.1 (2019-05-08)

### Bug fixes and other changes

 * add build-all.sh, publish-all.sh scripts

## v1.3.0 (2019-05-06)

### Features

 * add tensorflow serving batching

### Bug fixes and other changes

 * install requirements.txt in writable dir

## v1.2.1 (2019-04-29)

### Bug fixes and other changes

 * make njs code handle missing custom attributes header

## v1.2.0 (2019-04-29)

### Features

 * add python service for pre/post-processing

## v1.1.9 (2019-04-09)

### Bug fixes and other changes

 * improve handling of ei binary during builds

## v1.1.8 (2019-04-08)

### Bug fixes and other changes

 * add data generator and perf tests
 * remove per-line parsing

## v1.1.7 (2019-04-05)

### Bug fixes and other changes

 * add additional csv test case

## v1.1.6 (2019-04-04)

### Bug fixes and other changes

 * handle zero values correctly

## v1.1.5 (2019-04-04)

### Bug fixes and other changes

 * update EI binary directory.

## v1.1.4 (2019-04-01)

### Changes

 * Support payloads with many csv rows and change CSV parsing behavior

## v1.1.3 (2019-03-29)

### Bug fixes

 * update EI binary location

## v1.1.2 (2019-03-14)

### Bug fixes

 * remove tfs deployment tests

## v1.1.1 (2019-03-13)

### Bug fixes

 * create bucket during test
 * fix argname in deployment test
 * fix repository name in buildspec
 * add deployment tests and run them concurrently
 * report error for missing ei version
 * remove extra commma in buildspec

### Other changes

 * add eia images to release build
 * update buildspec to output deployments.json
 * Modify EI image repository and tag to match Python SDK.
 * Change test directory to be consistent with PDT pipeline.
 * Add EI support to TFS container.
 * simplify tfs versioning
 * add buildspec.yml for codebuild
 * add tox, pylint, flake8, jshint
