# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os
import re
import signal
import subprocess

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ServiceManager(object):
    def __init__(self):
        self._state = 'initializing'
        self._nginx = None
        self._tfs = None
        self._requirements = None
        self._execute_custom_code = False
        self._python_service = None
        self._nginx_ping_requests = 'js_content ping'
        self._nginx_invocation_requests = 'js_content invocations'
        self._tfs_version = os.environ.get('SAGEMAKER_TFS_VERSION', '1.12')
        self._nginx_http_port = os.environ.get('SAGEMAKER_BIND_TO_PORT', '8080')
        self._nginx_loglevel = os.environ.get('SAGEMAKER_TFS_NGINX_LOGLEVEL', 'error')
        self._tfs_default_model_name = os.environ.get('SAGEMAKER_TFS_DEFAULT_MODEL_NAME', None)
        self._python_path = os.environ.get('PYTHONPATH', '/opt/ml/model/lib')

        if 'SAGEMAKER_SAFE_PORT_RANGE' in os.environ:
            port_range = os.environ['SAGEMAKER_SAFE_PORT_RANGE']
            parts = port_range.split('-')
            low = int(parts[0])
            hi = int(parts[1])
            if low + 2 > hi:
                raise ValueError('not enough ports available in SAGEMAKER_SAFE_PORT_RANGE ({})'
                                 .format(port_range))
            self._tfs_grpc_port = str(low)
            self._tfs_rest_port = str(low + 1)
            self._gunicorn_port = str(low + 2)

        else:
            # just use the standard default ports
            self._tfs_grpc_port = '9000'
            self._tfs_rest_port = '8501'
            self._gunicorn_port = '5000'

    def _create_tfs_config(self):
        models = self._find_models()

        if not models:
            raise ValueError('no SavedModel bundles found!')

        if self._tfs_default_model_name is None:
            self._tfs_default_model_name = os.path.basename(models[0])
            log.info('using default model name: {}'.format(self._tfs_default_model_name))

        # config (may) include duplicate 'config' keys, so we can't just dump a dict
        config = 'model_config_list: {\n'
        for m in models:
            config += '  config: {\n'
            config += '    name: "{}",\n'.format(os.path.basename(m))
            config += '    base_path: "{}",\n'.format(m)
            config += '    model_platform: "tensorflow"\n'
            config += '  },\n'
        config += '}\n'

        log.info('tensorflow serving model config: \n%s\n', config)

        with open('/sagemaker/model-config.cfg', 'w') as f:
            f.write(config)

    def _find_models(self):
        base_path = '/opt/ml/model'
        models = []
        for f in self._find_saved_model_files(base_path):
            parts = f.split('/')
            if len(parts) >= 6 and re.match(r'^\d+$', parts[-2]):
                model_path = '/'.join(parts[0:-2])
                if model_path not in models:
                    models.append(model_path)
        return models

    def _find_saved_model_files(self, path):
        for e in os.scandir(path):
            if e.is_dir():
                yield from self._find_saved_model_files(os.path.join(path, e.name))
            else:
                if e.name == 'saved_model.pb':
                    yield os.path.join(path, e.name)

    def _setup_python_service(self):
        requirements_path = '/opt/ml/model/requirements.txt'
        inference_path = '/opt/ml/model/inference.py'

        if os.path.exists(requirements_path):
            cmd = 'pip3 install --install-option="--prefix=${}" -r {}'.\
                format(self._python_path, requirements_path)
            log.info('installing required packages...')
            p = subprocess.Popen(cmd.split())
            self._requirements = p

        if os.path.exists(inference_path):
            self._nginx_ping_requests = 'proxy_pass http://localhost:{}/ping'\
                .format(self._gunicorn_port)
            self._nginx_invocation_requests = 'proxy_pass http://localhost:{}/invocations'\
                .format(self._gunicorn_port)
            self._execute_custom_code = True

    def _create_nginx_config(self):
        template = self._read_nginx_template()
        pattern = re.compile(r'%(\w+)%')
        template_values = {
            'TFS_VERSION': self._tfs_version,
            'TFS_REST_PORT': self._tfs_rest_port,
            'TFS_DEFAULT_MODEL_NAME': self._tfs_default_model_name,
            'NGINX_HTTP_PORT': self._nginx_http_port,
            'NGINX_LOG_LEVEL': self._nginx_loglevel,
            'FORWARD_PING_REQUESTS': self._nginx_ping_requests,
            'FORWARD_INVOCATION_REQUESTS': self._nginx_invocation_requests,
            'GUNICORN_PORT': self._gunicorn_port
        }

        config = pattern.sub(lambda x: template_values[x.group(1)], template)
        log.info('nginx config: \n%s\n', config)

        with open('/sagemaker/nginx.conf', 'w') as f:
            f.write(config)

    def _read_nginx_template(self):
        with open('/sagemaker/nginx.conf.template', 'r') as f:
            template = f.read()
            if not template:
                raise ValueError('failed to read nginx.conf.template')

            return template

    def _start_tfs(self):
        self._log_version('tensorflow_model_server --version', 'tensorflow version info:')
        tfs_config_path = '/sagemaker/model-config.cfg'
        cmd = "tensorflow_model_server --port={} --rest_api_port={} --model_config_file={}".format(
            self._tfs_grpc_port, self._tfs_rest_port, tfs_config_path)
        log.info('tensorflow serving command: {}'.format(cmd))
        p = subprocess.Popen(cmd.split())
        log.info('started tensorflow serving (pid: %d)', p.pid)
        self._tfs = p

    def _start_python_service(self):
        self._log_version('gunicorn --version', 'gunicorn version info:')
        cmd = 'gunicorn -b localhost:{} --chdir /sagemaker ' \
              'python_service:app --reload'.format(self._gunicorn_port)
        log.info('gunicorn command: {}'.format(cmd))
        p = subprocess.Popen(cmd.split())
        log.info('started python service (pid: %d)', p.pid)
        self._python_service = p

    def _start_nginx(self):
        self._log_version('/usr/sbin/nginx -V', 'nginx version info:')
        p = subprocess.Popen('/usr/sbin/nginx -c /sagemaker/nginx.conf'.split())
        log.info('started nginx (pid: %d)', p.pid)
        self._nginx = p

    def _log_version(self, command, message):
        try:
            output = subprocess.check_output(
                command.split(),
                stderr=subprocess.STDOUT).decode('utf-8', 'backslashreplace').strip()
            log.info('{}\n{}'.format(message, output))
        except subprocess.CalledProcessError:
            log.warning('failed to run command: %s', command)

    def _stop(self, *args):  # pylint: disable=W0613
        self._state = 'stopping'
        log.info('stopping services')
        try:
            os.kill(self._nginx.pid, signal.SIGQUIT)
        except OSError:
            pass
        try:
            os.kill(self._tfs.pid, signal.SIGTERM)
        except OSError:
            pass

        self._state = 'stopped'
        log.info('stopped')

    def start(self):
        log.info('starting services')
        self._state = 'starting'
        signal.signal(signal.SIGTERM, self._stop)

        self._create_tfs_config()
        self._setup_python_service()
        self._create_nginx_config()

        self._start_tfs()

        if self._execute_custom_code:
            self._start_python_service()

        self._start_nginx()
        self._state = 'started'

        while True:
            pid, status = os.wait()

            if self._state != 'started':
                break

            if pid == self._nginx.pid:
                log.warning('unexpected nginx exit (status: {}). restarting.'.format(status))
                self._start_nginx()

            elif pid == self._tfs.pid:
                log.warning(
                    'unexpected tensorflow serving exit (status: {}). restarting.'.format(status))
                self._start_tfs()

            elif pid == self._python_service.pid:
                log.warning('unexpected python service exit (status: {}). starting.'.format(status))
                self._start_python_service()

            elif pid == self._requirements.pid:
                log.error('failed to install required packages (status: {}).'.format(status))
                self._state = 'error'

        self._stop()


if __name__ == '__main__':
    ServiceManager().start()
