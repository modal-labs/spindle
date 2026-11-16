import with_proxy_auth

from spindle.benchmarks.client_metrics import install_from_env

with_proxy_auth.install()
install_from_env()
