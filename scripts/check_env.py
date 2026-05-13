"""
Print environment information needed before training on a new machine.
"""

import importlib
import platform
import shutil
import subprocess
import sys


PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "trl",
    "qwen_vl_utils",
    "PIL",
    "psycopg2",
]


def run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "not found"
    output = (result.stdout or result.stderr).strip()
    return output or f"exit_code={result.returncode}"


def package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}"
    return getattr(module, "__version__", "unknown")


def main():
    print("## Python")
    print(sys.executable)
    print(sys.version.replace("\n", " "))
    print(f"platform: {platform.platform()}")
    print()

    print("## CUDA / GPU")
    print(f"nvcc: {shutil.which('nvcc') or 'not found'}")
    print(run(["nvidia-smi"]))
    print()

    print("## Python packages")
    for package in PACKAGES:
        print(f"{package}: {package_version(package)}")
    print()

    try:
        import torch

        print("## Torch")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device_count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total_gib = props.total_memory / 1024**3
                print(f"gpu[{index}]: {props.name}, {total_gib:.1f} GiB")
    except Exception as exc:
        print(f"torch check failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
