from src.train.train_loop import run_loop 
from src.settings import Settings


def main() -> None:
    """
    This is the main file, run everything from here
    Uses both command-line arguments and also a settings.yaml

    Command-line args:

    """
    settings = Settings()

    run_loop(settings=settings)


if __name__ == "__main__":
    main()