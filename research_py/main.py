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


def change_file_names() -> None:
    print("Started renaming")
    path = Path("E:/Datasets/omnifall/OOPS/falls")
    assert path.exists()
    files = os.listdir(path)
    os.chdir(path)
    for file in files:
        new_name = file.replace(" ", "")
        new_name = new_name.replace("(", "")
        new_name = new_name.replace(")", "")
        new_name = new_name.replace("!", "")
        new_name = new_name.replace("'", "")
        new_name = new_name.replace(",", "")
        new_name = new_name.replace("&", "")
        new_name = new_name.replace("WhipItWhipIt", "WhipIt_WhipIt")
        if not new_name == file:
            print(f"Renamed {file} to {new_name}")
            os.rename(file, new_name)


if __name__ == "__main__":
    main()