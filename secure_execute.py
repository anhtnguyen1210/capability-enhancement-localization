import os,sys,math,tempfile,subprocess,hashlib
SECURE_CODE_EVAL_TIMEOUT_SECONDS=3
def _secure_humaneval_execution(
    program,
    target,
    timeout_seconds=SECURE_CODE_EVAL_TIMEOUT_SECONDS,
):
    if os.environ.get("LLMCOMP_HUMANEVAL_SANDBOX") != "1":
        raise RuntimeError("HumanEval program classification requires LLMCOMP_HUMANEVAL_SANDBOX=1")
    helper = os.environ.get("LLMCOMP_HUMANEVAL_HELPER")
    if not helper or not os.path.isfile(helper) or not os.access(helper, os.X_OK):
        raise RuntimeError("HumanEval secure helper is missing or not executable")
    memory_bytes = int(os.environ.get("LLMCOMP_HUMANEVAL_PROGRAM_MEMORY_BYTES", 536_870_912))
    pids_limit = int(os.environ.get("LLMCOMP_HUMANEVAL_PROGRAM_PIDS_LIMIT", 32))
    cpu_seconds = max(1, int(math.ceil(timeout_seconds)) + 1)
    file_bytes = int(os.environ.get("LLMCOMP_HUMANEVAL_PROGRAM_FILE_BYTES", 1_048_576))
    try:
        with tempfile.TemporaryDirectory(prefix="humaneval-program-") as tmp:
            completed = subprocess.run(
                [
                    helper, str(memory_bytes), str(pids_limit), str(cpu_seconds),
                    str(file_bytes), "deny-writes", sys.executable, "-I", "-c",
                    program + "\n" + target,
                ],
                cwd=tmp, capture_output=True, timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as exc:
        return {
            "execution_outcome": "timeout",
            "execution_returncode": None,
            "execution_stdout_hash": hashlib.sha256(exc.stdout or b"").hexdigest(),
            "execution_stderr_hash": hashlib.sha256(exc.stderr or b"").hexdigest(),
        }
    except OSError as exc:
        return {
            "execution_outcome": "sandbox_error",
            "execution_returncode": None,
            "execution_stdout_hash": hashlib.sha256(b"").hexdigest(),
            "execution_stderr_hash": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        }
    if completed.returncode == 0:
        outcome = "passed"
    elif completed.returncode == 125:
        # The helper reserves this status for setup/filter/exec failures.
        outcome = "sandbox_error"
    elif completed.returncode < 0:
        outcome = "crash"
    else:
        outcome = "failed"
    return {
        "execution_outcome": outcome,
        "execution_returncode": completed.returncode,
        "execution_stdout_hash": hashlib.sha256(completed.stdout).hexdigest(),
        "execution_stderr_hash": hashlib.sha256(completed.stderr).hexdigest(),
    }
