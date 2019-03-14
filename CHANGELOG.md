# Changelog

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
