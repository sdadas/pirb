"""Run the official Bielik-One backend and frontend in one Linux container."""

import argparse
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="speakleash/Bielik-One-4B-r1")
    parser.add_argument("--revision")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-port", type=int, default=8013)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--kv-cache-gb", type=float)
    args = parser.parse_args()
    if args.port == args.backend_port:
        parser.error("frontend and backend ports must differ")

    children = []

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        from bielik_one.launch import resolve_package

        # Resolve once so both processes use exactly the same cached snapshot.
        package = str(resolve_package(args.model, args.revision))
        backend = ["bielik-one", "backend", package, "--backend-host", "127.0.0.1",
                   "--port", str(args.backend_port)]
        frontend = ["bielik-one", "frontend", package, "--host", args.host,
                    "--port", str(args.port), "--backend-url", f"http://127.0.0.1:{args.backend_port}"]
        for flag in ("fp8", "eager"):
            if getattr(args, flag):
                backend.append("--" + flag)
        if args.kv_cache_gb is not None:
            backend.extend(["--kv-cache-gb", str(args.kv_cache_gb)])
        if args.max_num_seqs is not None:
            for command in (backend, frontend):
                command.extend(["--max-num-seqs", str(args.max_num_seqs)])
        if args.no_calibration:
            frontend.append("--no-calibration")

        children.append(subprocess.Popen(backend, start_new_session=True))
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if children[0].poll() is not None:
                raise RuntimeError(f"Bielik-One backend exited with code {children[0].returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Bielik-One backend startup timed out")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.backend_port}/health", timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        children.append(subprocess.Popen(frontend, start_new_session=True))
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError("A Bielik-One serving process stopped; shutting down the container")
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == "__main__":
    main()
