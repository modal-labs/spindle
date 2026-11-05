"""A registered config's trainer or inference app, deployed from the frontend."""

import json
import os

from spindle.deployments import DeploymentConfig

from .deployment_apps import build_inference_app, build_trainer_app
from .deployment_configs import WORKER_CONFIG_ENV, WORKER_ROLE_ENV, platform_from_env

deployment = DeploymentConfig.from_registry(json.loads(os.environ[WORKER_CONFIG_ENV]))
build = {"trainer": build_trainer_app, "inference": build_inference_app}[
    os.environ[WORKER_ROLE_ENV]
]
app = build(deployment, platform_from_env())[0]
