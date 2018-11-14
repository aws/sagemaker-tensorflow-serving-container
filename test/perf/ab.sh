#!/bin/bash

ab -k -n 10000 -c 16 -p test/resources/inputs/test.json -T 'application/json' http://localhost:8080/tfs/v1/models/half_plus_three:predict
ab -k -n 10000 -c 16 -p test/resources/inputs/test.json -T 'application/json' http://localhost:8080/invocations
ab -k -n 10000 -c 16 -p test/resources/inputs/test.jsons -T 'application/json' http://localhost:8080/invocations
ab -k -n 10000 -c 16 -p test/resources/inputs/test.csv -T 'text/csv' http://localhost:8080/invocations
ab -k -n 10000 -c 16 -p test/resources/inputs/test-cifar.json -T 'application/json' \
    -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=cifar' \
    http://localhost:8080/invocations
