# Changelog

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
