import asyncio
import os
from typing import Dict, Optional, Iterable
import contextlib


def _return_code(exit_code: int = None,
                 stdout: str = None,
                 stderr: str = None,
                 cmd: Iterable[str] = None,
                 timed_out: bool = None):
    return {"exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "cmd": cmd,
            "timed_out": timed_out}


async def execute_process(cmd,
                    env: Optional[Dict[str, str]] = None,
                    timeout_sec: Optional[float] = None):
    """
    Executes a subprocess asynchronously with optional environment variables and timeout.

    Parameters
    ----------
    cmd : list
        List of strings that will be passed for execution.
    env : dict, optional
        Environment variables to set for the subprocess. Defaults to None.
    timeout_sec : float, optional
        Timeout in seconds for the subprocess execution. Defaults to None.

    Returns
    -------
    dict
    """

    cwd = os.getcwd()
    exec_env = os.environ.copy()
    if env:
        for k, v in env.items():
            exec_env[str(k)] = str(v)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            env=exec_env,
        )

        try:
            stdout_b, stderr_b = (
                await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
                if timeout_sec
                else await proc.communicate()
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return _return_code(stderr=f"Timed out after {timeout_sec}s", timed_out=True, cmd=cmd)
        return _return_code(exit_code=proc.returncode,
                            stdout=stdout_b.decode(),
                            stderr=stderr_b.decode(),
                            cmd=cmd)
    except FileNotFoundError as e:
        return _return_code(stderr=f"Failed to execute via Singularity instance; verify that the Singularity instance is running "
                                  f"and the executable exists inside the container. Error: {e}",
                            cmd=cmd)
