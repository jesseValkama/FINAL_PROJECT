import os
from pathlib import Path
from src.train.train_loop import run_loop 
from src.settings import Settings


def main() -> None:
    """
    This is the main file, run everything from here
    Uses both command-line arguments and also a settings.yaml

    Command-line args:
        train: 0 | 1
        test: 0 | 1
        inference: 0 | 1

    """
    settings = Settings()
    project_dir = Path(settings.project_dir)
    assert project_dir.exists, "Enter a valid project directory, no need for main.py"
    os.chdir(project_dir)

    run_loop(settings=settings)


if __name__ == "__main__":
    main()