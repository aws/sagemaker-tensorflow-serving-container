# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import signal
from contextlib import contextmanager

import importlib.util
import json
import logging
import os
import re
import subprocess
import time
from collections import namedtuple

import falcon
import requests

from multi_model_utils import MultiModelException

INFERENCE_SCRIPT_PATH = '/opt/ml/model/code/inference.py'
PYTHON_PROCESSING_ENABLED = os.path.exists(INFERENCE_SCRIPT_PATH)
SAGEMAKER_MULTI_MODEL_ENABLED = os.environ.get('SAGEMAKER_MULTI_MODEL', 'false').lower() == "true"
MODEL_CONFIG_FILE_PATH = '/sagemaker/model-config.cfg'
TFS_GRPC_PORT = os.environ.get('TFS_GRPC_PORT')
TFS_REST_PORT = os.environ.get('TFS_REST_PORT')
SAGEMAKER_TFS_PORT_RANGE = os.environ.get('SAGEMAKER_SAFE_PORT_RANGE')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'
CUSTOM_ATTRIBUTES_HEADER = 'X-Amzn-SageMaker-Custom-Attributes'

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_port, '
                     'custom_attributes, request_content_type, accept_header, content_length')


def default_handler(data, context):
    """A default inference request handler that directly send post request to TFS rest port with
    un-processed data and return un-processed response

    :param data: input data
    :param context: context instance that contains tfs_rest_uri
    :return: inference response from TFS model server
    """
    response = requests.post(context.rest_uri, data=data)
    return response.content, context.accept_header


class PythonServiceResource(object):

    def __init__(self):
        if SAGEMAKER_MULTI_MODEL_ENABLED:
            self._model_tfs_rest_port = {}
            self._model_tfs_grpc_port = {}
            self._model_tfs_pid = {}
            self._tfs_ports = self._parse_sagemaker_port_range(SAGEMAKER_TFS_PORT_RANGE)
        else:
            self._tfs_grpc_port = TFS_GRPC_PORT
            self._tfs_rest_port = TFS_REST_PORT

        self._tfs_default_model_name = os.environ.get('TFS_DEFAULT_MODEL_NAME', "None")
        if PYTHON_PROCESSING_ENABLED:
            self._handler, self._input_handler, self._output_handler = self._import_handlers()
            self._handlers = self._make_handler(self._handler,
                                                self._input_handler,
                                                self._output_handler)
        else:
            self._handlers = default_handler

        # TODO: error handling of TFS subprocess

    def on_post(self, req, res, model_name=None):
        log.info(req.uri)
        if model_name or "invocations" in req.uri:
            self._handle_invocation_post(req, res, model_name)
        else:
            data = json.loads(req.stream.read()
                              .decode('utf-8'))
            self._handle_load_model_post(res, data)

    def _parse_sagemaker_port_range(self, port_range):
        lower, upper = port_range.split('-')
        lower = int(lower)
        upper = int(upper)
        rest_port = lower
        grpc_port = (lower + upper) // 2
        tfs_ports = {
            "current_rest_port": rest_port,
            "current_grpc_port": grpc_port,
            "rest_port_limit": grpc_port - 1,
            "grpc_port_limit": upper
        }
        return tfs_ports

    def _ports_available(self):
        if self._tfs_ports["current_rest_port"] > self._tfs_ports["rest_port_limit"]:
            return False
        if self._tfs_ports["current_grpc_port"] > self._tfs_ports["grpc_port_limit"]:
            return False
        return True

    def _create_tfs_config_individual_model(self, model_name, base_path):
        config = 'model_config_list: {\n'
        config += '  config: {\n'
        config += '    name: "{}",\n'.format(model_name)
        config += '    base_path: "{}",\n'.format(base_path)
        config += '    model_platform: "tensorflow"\n'
        config += '  }\n'
        config += '}\n'
        return config

    def _create_tfs_command(self, grpc_port, rest_port, tfs_config_path, tfs_batching_args=""):
        cmd = "tensorflow_model_server " \
              "--port={} " \
              "--rest_api_port={} " \
              "--model_config_file={} " \
              "--max_num_load_retries=0 {}" \
            .format(grpc_port, rest_port, tfs_config_path,
                    tfs_batching_args)
        return cmd

    def _handle_load_model_post(self, res, data):  # noqa: C901
        model_name = data['model_name']
        base_path = data['url']

        # model is already loaded
        if model_name in self._model_tfs_pid:
            res.status = falcon.HTTP_409
            res.body = json.dumps({
                'error': 'Model {} is already loaded.'.format(model_name)
            })

        # check if there are available ports
        if not self._ports_available():
            res.status = falcon.HTTP_507
            res.body = json.dumps({
                'error': 'Memory exhausted: no available ports to load the model.'
            })

        self._model_tfs_rest_port[model_name] = self._tfs_ports['current_rest_port']
        self._model_tfs_grpc_port[model_name] = self._tfs_ports['current_grpc_port']

        # validate model files are in the specified base_path
        if self.validate_model_dir(base_path):
            try:
                tfs_config = self._create_tfs_config_individual_model(model_name, base_path)
                tfs_config_file = '/sagemaker/tfs-config/{}/model-config.cfg'.format(model_name)
                log.info('tensorflow serving model config: \n%s\n', tfs_config)
                os.makedirs(os.path.dirname(tfs_config_file))
                with open(tfs_config_file, 'w') as f:
                    f.write(tfs_config)

                # TODO: get_tfs_batching_args()
                cmd = self._create_tfs_command(self._tfs_ports['current_grpc_port'],
                                               self._tfs_ports['current_rest_port'],
                                               tfs_config_file,
                                               )
                p = subprocess.Popen(cmd.split())
                time.sleep(1)

                log.info('started tensorflow serving (pid: %d)', p.pid)
                # update model name <-> tfs pid map
                self._model_tfs_pid[model_name] = p

                # increment next available rest/grpc port
                self._tfs_ports["current_rest_port"] += 1
                self._tfs_ports["current_grpc_port"] += 1
                res.status = falcon.HTTP_200
                res.body = json.dumps({
                    'success':
                        'Successfully loaded model {}, '
                        'listening on rest port {} '
                        'and grpc port {}.'.format(model_name,
                                                   self._model_tfs_rest_port,
                                                   self._model_tfs_grpc_port,)
                })
            except MultiModelException as multi_model_exception:
                if multi_model_exception.code == 409:
                    res.status = falcon.HTTP_409
                    res.body = multi_model_exception.msg
                elif multi_model_exception.code == 408:
                    res.status = falcon.HTTP_408
                    res.body = multi_model_exception.msg
                else:
                    raise MultiModelException(falcon.HTTP_500, multi_model_exception.msg)
            except FileExistsError as e:
                res.status = falcon.HTTP_409
                res.body = json.dumps({
                    'error': 'Model {} is already loaded. {}'.format(model_name, str(e))
                })
            except OSError as os_error:
                if os_error.errno == 12:
                    raise MultiModelException(falcon.HTTP_507,
                                              'Memory exhausted: '
                                              'not enough memory to start TFS instance')
                else:
                    raise MultiModelException(falcon.HTTP_500, os_error.strerror)
        else:
            res.status = falcon.HTTP_404
            res.body = json.dumps({
                'error':
                    'Could not find valid base path {} for servable {}'.format(base_path,
                                                                               model_name)
            })

    def _handle_invocation_post(self, req, res, model_name=None):
        log.info("SAGEMAKER_MULTI_MODEL_ENABLED: {}".format(str(SAGEMAKER_MULTI_MODEL_ENABLED)))
        if SAGEMAKER_MULTI_MODEL_ENABLED:
            if model_name:
                if model_name not in self._model_tfs_rest_port:
                    res.status = falcon.HTTP_404
                    res.body = json.dumps({
                        'error': "Model {} is not loaded yet.".format(model_name)
                    })
                    return
                else:
                    log.info("model name: {}".format(model_name))
                    rest_port = self._model_tfs_rest_port[model_name]
                    log.info("rest port: {}".format(str(self._model_tfs_rest_port[model_name])))
                    grpc_port = self._model_tfs_grpc_port[model_name]
                    log.info("grpc port: {}".format(str(self._model_tfs_grpc_port[model_name])))
                    data, context = self._parse_request(req, rest_port, grpc_port, model_name)
            else:
                res.status = falcon.HTTP_400
                res.body = json.dumps({
                    'error': 'Invocation request does not contain model name.'
                })
        else:
            data, context = self._parse_request(req, self._tfs_rest_port, self._tfs_grpc_port)

        try:
            res.status = falcon.HTTP_200
            res.body, res.content_type = self._handlers(data, context)
        except Exception as e:  # pylint: disable=broad-except
            log.exception('exception handling request: {}'.format(e))
            res.status = falcon.HTTP_500
            res.body = json.dumps({
                'error': str(e)
            }).encode('utf-8')  # pylint: disable=E1101

    def _import_handlers(self):
        spec = importlib.util.spec_from_file_location('inference', INFERENCE_SCRIPT_PATH)
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        _custom_handler, _custom_input_handler, _custom_output_handler = None, None, None
        if hasattr(inference, 'handler'):
            _custom_handler = inference.handler
        elif hasattr(inference, 'input_handler') and hasattr(inference, 'output_handler'):
            _custom_input_handler = inference.input_handler
            _custom_output_handler = inference.output_handler
        else:
            raise NotImplementedError('Handlers are not implemented correctly in user script.')

        return _custom_handler, _custom_input_handler, _custom_output_handler

    def _make_handler(self, custom_handler, custom_input_handler, custom_output_handler):
        if custom_handler:
            return custom_handler

        def handler(data, context):
            processed_input = custom_input_handler(data, context)
            response = requests.post(context.rest_uri, data=processed_input)
            return custom_output_handler(response, context)

        return handler

    def _parse_request(self, req, rest_port, grpc_port, model_name=None):
        tfs_attributes = self._parse_tfs_custom_attributes(req)
        tfs_uri = self._tfs_uri(rest_port, tfs_attributes, model_name)

        if not model_name:
            model_name = tfs_attributes.get('tfs-model-name')

        context = Context(model_name,
                          tfs_attributes.get('tfs-model-version'),
                          tfs_attributes.get('tfs-method'),
                          tfs_uri,
                          grpc_port,
                          req.get_header(CUSTOM_ATTRIBUTES_HEADER),
                          req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE,
                          req.get_header('Accept') or DEFAULT_ACCEPT_HEADER,
                          req.content_length)

        data = req.stream
        return data, context

    def _parse_tfs_custom_attributes(self, req):
        attributes = {}
        header = req.get_header(CUSTOM_ATTRIBUTES_HEADER)
        if header:
            for attribute in re.findall(r'(tfs-[a-z\-]+=[^,]+)', header):
                k, v = attribute.split('=')
                attributes[k] = v
        return attributes

    def _tfs_uri(self, port, attributes, model_name=None):

        log.info("sagemaker tfs attributes: ")
        log.info(str(attributes))

        tfs_model_name = model_name if model_name \
            else attributes.get('tfs-model-name', self._tfs_default_model_name)
        tfs_model_version = attributes.get('tfs-model-version')
        tfs_method = attributes.get('tfs-method', 'predict')

        uri = 'http://localhost:{}/v1/models/{}'.format(port, tfs_model_name)
        if tfs_model_version:
            uri += '/versions/' + tfs_model_version
        uri += ':' + tfs_method
        return uri

    def on_get(self, req, res, model_name=None):  # pylint: disable=W0613
        if model_name is None:
            models_info = {}
            uri = 'http://localhost:{}/v1/models/{}'
            for model, port in self._model_tfs_rest_port.items():
                try:
                    info = json.loads(requests.get(uri.format(port, model)).content)
                    models_info[model] = info
                except ValueError as e:
                    log.exception('exception handling request: {}'.format(e))
                    res.status = falcon.HTTP_500
                    res.body = json.dumps({
                        'error': str(e)
                    }).encode('utf-8')
            res.status = falcon.HTTP_200
            res.body = json.dumps(models_info)
        else:
            if model_name not in self._model_tfs_rest_port:
                res.status = falcon.HTTP_404
                res.body = json.dumps({
                    'error': 'Model {} is loaded yet.'.format(model_name)
                }).encode('utf-8')
            else:
                port = self._model_tfs_rest_port[model_name]
                uri = 'http://localhost:{}/v1/models/{}'.format(port, model_name)
                try:
                    info = requests.get(uri)
                    res.status = falcon.HTTP_200
                    res.body = json.dumps({
                        'model': info
                    }).encode('utf-8')
                except ValueError as e:
                    log.exception('exception handling request: {}'.format(e))
                    res.status = falcon.HTTP_500
                    res.body = json.dumps({
                        'error': str(e)
                    }).encode('utf-8')

    def on_delete(self, req, res, model_name):  # pylint: disable=W0613
        if model_name not in self._model_tfs_pid:
            res.status = falcon.HTTP_404
            res.body = json.dumps({
                'error': 'Model {} is not loaded yet'.format(model_name)
            })
        else:
            try:
                self._model_tfs_pid[model_name].kill()
                os.remove('/sagemaker/tfs-config/{}/model-config.cfg'.format(model_name))
                os.rmdir('/sagemaker/tfs-config/{}'.format(model_name))
                del self._model_tfs_rest_port[model_name]
                del self._model_tfs_grpc_port[model_name]
                del self._model_tfs_pid[model_name]
                res.status = falcon.HTTP_200
                res.body = json.dumps({
                    'success': 'Successfully unloaded model {}.'.format(model_name)
                })
            except OSError as error:
                res.status = falcon.HTTP_500
                res.body = json.dumps({
                    'error': str(error)
                }).encode('utf-8')

    def validate_model_dir(self, model_path):
        # model base path doesn't exits
        if not os.path.exists(model_path):
            return False
        # model versions doesn't exist
        log.info("model path exists")
        versions = []
        for _, dirs, _ in os.walk(model_path):
            for dirname in dirs:
                log.info("dirname: {}".format(dirname))
                if dirname.isdigit():
                    versions.append(dirname)
        return self.validate_model_versions(versions)

    def validate_model_versions(self, versions):
        log.info(versions)
        if not versions:
            return False
        for v in versions:
            if v.isdigit():
                # TensorFlow model server will succeed with any versions found
                # even if there are directories that's not a valid model version,
                # the loading will succeed.
                return True
        return False

    @contextmanager
    def _timeout(self, seconds):
        def _raise_timeout_error(signum, frame):
            raise Exception(408, 'Timed our after {} seconds'.format(seconds))

        try:
            signal.signal(signal.SIGALRM, _raise_timeout_error)
            signal.alarm(seconds)
            yield
        finally:
            signal.alarm(0)


class PingResource(object):
    def on_get(self, req, res):  # pylint: disable=W0613
        res.status = falcon.HTTP_200


class ServiceResources(object):
    def __init__(self):
        self._enable_python_processing = PYTHON_PROCESSING_ENABLED
        self._enable_model_manager = SAGEMAKER_MULTI_MODEL_ENABLED
        self._python_service_resource = PythonServiceResource()
        self._ping_resource = PingResource()

    def add_routes(self, application):
        application.add_route('/ping', self._ping_resource)
        application.add_route('/invocations', self._python_service_resource)

        if self._enable_model_manager:
            application.add_route('/models', self._python_service_resource)
            application.add_route('/models/{model_name}', self._python_service_resource)
            application.add_route('/models/{model_name}/invoke', self._python_service_resource)


app = falcon.API()
resources = ServiceResources()
resources.add_routes(app)
