import pathlib
import sys

sys.path.append(str(pathlib.Path().resolve()))

from src.cli import cli  # noqa: E402

if __name__ == "__main__":
    print(pathlib.Path().resolve())
    cli(prog_name="mednlp")
