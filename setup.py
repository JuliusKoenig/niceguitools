import sys
from pathlib import Path


def setup_exit(exit_code: int = 0, message: str = None):
    if message is not None:
        print(message)
    raise SystemExit(exit_code)


def get_version():
    with open(Path("src") / "niceguitools" / "__version__", "r") as f:
        return f.read()


def change_version(new_version: str):
    with open(Path("src") / "niceguitools" / "__version__", "w") as f:
        f.write(new_version)


def print_version():
    print(f"Current version: {get_version()}")
    setup_exit(0)


if __name__ == "__main__":
    old_version = None
    if "-bV" in sys.argv or "--build-version" in sys.argv:
        index = sys.argv.index("-bV" if "-bV" in sys.argv else "--build-version") + 1
        if index >= len(sys.argv):
            setup_exit(1, "Missing version number.")
        if sys.argv[index].startswith("-"):
            setup_exit(1, "Missing version number.")
        old_version = get_version()
        build_version = f"{old_version}.{sys.argv[index]}"
        change_version(build_version)
        sys.argv.pop(index)
        sys.argv.pop(index - 1)

    if "-v" in sys.argv or "--version" in sys.argv:
        print_version()

    print(f"Building version {get_version()}...")

    from setuptools import setup

    setup()
    print(f"Successfully built version {get_version()}.")
    if old_version is not None:
        change_version(old_version)
