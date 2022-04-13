import json
import shutil
import signal
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, Popen, run
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, BaseSettings


class Settings(BaseSettings):
    mount_server_pach_config_path: Path = Path("/tmp/config.json")
    mount_server_terminate_timeout_in_seconds: int = 15
    mount_server_log_file_path: Path = Path("/tmp/pachctl-mount-server.log")


class MountServerManager:
    def __init__(
        self,
        pach_config_path: Path,
        log_file_path: Path,
        terminate_timeout_in_seconds: int,
    ):
        self._pach_config_path = pach_config_path
        self._log_file_path = log_file_path
        self._terminate_timeout_in_seconds = terminate_timeout_in_seconds
        self._temp_pach_config_path: Optional[Path] = None
        self._mount_server_process: Optional[Popen] = None

    def is_server_running(self) -> bool:
        return (
            self._mount_server_process is not None
            and self._mount_server_process.poll() is None
        )

    def start_server(self, context: str, mount_dir: str):
        # TODO: consider ensuring context actually exists in the config
        assert not self.is_server_running()

        # Since we need to change the context, copy the config over to a
        # temporary file so as to not affect the original config file.
        self._temp_pach_config_path = NamedTemporaryFile(delete=True)
        shutil.copyfile(self._pach_config_path, self._temp_pach_config_path.name)
        print(self._temp_pach_config_path.name)
        self._temp_pach_config_path.seek(0)

        try:
            # TODO: without using a shell, find a way to get the path to
            # `pachctl` automatically
            run(
                ["/usr/local/bin/pachctl", "config", "set", "active-context", context],
                env={
                    "PACH_CONFIG": self._temp_pach_config_path.name,
                },
                check=True,
            )
        except CalledProcessError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Failed to start the mount server. "
                    f"Ensure '{context}' is a valid context."
                ),
            )

        self._mount_server_process = Popen(
            [
                "/usr/local/bin/pachctl",
                "mount-server",
                "--mount-dir",
                mount_dir,
            ],
            env={
                "PACH_CONFIG": self._temp_pach_config_path.name,
            },
            stdout=self._log_file_path.open("wb"),
            stderr=STDOUT,
        )

    def stop_server(self):
        assert self.is_server_running()
        self._mount_server_process.send_signal(signal.CTRL_C_EVENT)
        self._mount_server_process.wait(timeout=self._terminate_timeout_in_seconds)
        self._mount_server_process = None

        # TODO: determine if `pachctl` only needs to read the config once when a
        # command is executed. If so, we don't need to maintain the temporary
        # config file past the lifetime of the `pachctl` command
        self._temp_pach_config_path.close()
        self._temp_pach_config_path = None


def validate_config(config: bytes):
    """Ensures `config` represents a valid `pachctl` config

    Args:
        config: a serialized `pachctl` config

    Returns: the active context of the config, if the config is valid

    Raises:
        HTTPException: If the config is not valid JSON or does not match the
            expected schema
    """
    # Ensure the config is valid JSON
    try:
        json.loads(config)
    except JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Unable to parse config; invalid JSON"
        )

    # Ensure the config matches the expected schema
    with NamedTemporaryFile() as tf:
        tf.write(config)
        tf.seek(0)

        try:
            # Attempt to pull the active-context from the config.
            # If the config is well-formed, this should succeed.
            result = run(
                ["/usr/local/bin/pachctl", "config", "get", "active-context"],
                env={
                    "PACH_CONFIG": str(tf.name),
                },
                check=True,
                capture_output=True,
            )
            active_context = result.stdout.strip()
            return active_context
        except CalledProcessError:
            raise HTTPException(
                status_code=400,
                detail="Unable to process config. Check for invalid fields",
            )


def get_server_status(manager: MountServerManager) -> Dict:
    current_time = datetime.now()
    running = manager.is_server_running()
    return {"current_time": current_time.isoformat(), "running": running}


class ConfigSpec(BaseModel):
    payload: bytes


class ServerSpec(BaseModel):
    context: str
    mount_dir: str


settings = Settings()
app = FastAPI()
manager = MountServerManager(
    pach_config_path=settings.mount_server_pach_config_path,
    log_file_path=settings.mount_server_log_file_path,
    terminate_timeout_in_seconds=settings.mount_server_terminate_timeout_in_seconds,
)


@app.on_event("startup")
async def startup_event():
    # ensures the config is created if it doesn't already exist
    run(
        ["/usr/local/bin/pachctl", "version"],
        env={
            "PACH_CONFIG": str(settings.mount_server_pach_config_path),
        },
    )


@app.get("/config")
async def get_config():
    return json.loads(settings.mount_server_pach_config_path.read_text())


@app.put("/config")
async def put_config(config_spec: ConfigSpec):
    config: bytes = config_spec.payload

    active_context = validate_config(config)
    settings.mount_server_pach_config_path.write_bytes(config)
    return {"active_context": active_context}


@app.get("/server")
async def server_status():
    return get_server_status(manager)


@app.put("/server")
async def start_server(server_spec: ServerSpec):
    if manager.is_server_running():
        manager.stop_server()
    manager.start_server(context=server_spec.context, mount_dir=server_spec.mount_dir)

    return get_server_status(manager)


@app.delete("/server")
async def stop_server():
    if manager.is_server_running():
        manager.stop_server()

    return get_server_status(manager)
