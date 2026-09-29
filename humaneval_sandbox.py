#!/usr/bin/env python3
"""Run a command in the resource-limited HumanEval Apptainer boundary."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--memory-bytes", type=int, default=536_870_912)
    parser.add_argument("--pids-limit", type=int, default=128)
    parser.add_argument("--cpu-seconds", type=int, default=8)
    parser.add_argument("--file-bytes", type=int, default=1_048_576)
    parser.add_argument("--output", help="optional path for the JSON result")
    parser.add_argument("--passthrough", action="store_true", help="stream child output")
    parser.add_argument("--preserve-returncode", action="store_true")
    parser.add_argument("--nv", action="store_true", help="bind NVIDIA devices and libraries")
    parser.add_argument(
        "--bind", action="append", default=[],
        help="additional Apptainer bind specification (source:destination:mode)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    with tempfile.TemporaryDirectory(prefix="humaneval-sandbox-") as tmp:
        helper = Path(tmp) / "humaneval_seccomp_exec"
        source = Path(__file__).with_name("humaneval_seccomp_exec.c")
        subprocess.run(["cc", "-O2", "-Wall", "-Werror", str(source), "-o", str(helper)], check=True)
        container_command = [
            "apptainer", "exec", "--containall", "--cleanenv", "--no-home",
            "--writable-tmpfs", "--ipc", "--pid",
            "--bind", f"{Path(args.repo).resolve()}:/workspace:ro",
            "--bind", f"{helper}:/sandbox/seccomp_exec:ro", "--cwd", "/tmp",
        ]
        for bind in args.bind:
            container_command.extend(("--bind", bind))
        if args.nv:
            container_command.append("--nv")
        container_command.extend([
            str(Path(args.image).resolve()), "/sandbox/seccomp_exec",
            str(args.memory_bytes), str(args.pids_limit), str(args.cpu_seconds),
            str(args.file_bytes), "-", *command,
        ])
        process = subprocess.Popen(
            container_command,
            stdout=None if args.passthrough else subprocess.PIPE,
            stderr=None if args.passthrough else subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        def child_pids(pid):
            try:
                raw = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                return []
            return [int(value) for value in raw.split()] if raw else []

        def descendants(pid):
            found = []
            for child in child_pids(pid):
                found.append(child)
                found.extend(descendants(child))
            return found

        def payload_pids():
            payloads = []
            for descendant in descendants(process.pid):
                try:
                    command = Path(f"/proc/{descendant}/comm").read_text().strip()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if command == "appinit":
                    payloads.extend(child_pids(descendant))
            return payloads

        def forward_signal(signum, _frame):
            if process.poll() is None:
                targets = payload_pids()
                if not targets:
                    os.killpg(process.pid, signal.SIGTERM)
                    return
                for target in targets:
                    try:
                        os.kill(target, signum)
                    except ProcessLookupError:
                        pass

        previous_handlers = {
            signum: signal.signal(signum, forward_signal)
            for signum in (signal.SIGUSR1, signal.SIGTERM)
        }
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
            outcome = "passed" if process.returncode == 0 else "failed"
            payload = {
                "outcome": outcome,
                "returncode": process.returncode,
                "stdout": stdout or "",
                "stderr": stderr or "",
            }
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            payload = {
                "outcome": "timeout", "returncode": None,
                "stdout": stdout or exc.stdout or "", "stderr": stderr or exc.stderr or "",
            }
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    encoded = json.dumps(payload, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.preserve_returncode:
        return 124 if timed_out else int(payload["returncode"])
    return 0 if payload["outcome"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
