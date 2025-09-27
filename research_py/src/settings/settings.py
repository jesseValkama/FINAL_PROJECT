class Settings:

    def __init__(self):
        self._train = True
        self._test = True

        self._split_format = "cs-staged"
        self._dataset_path = "D:/datasets/omnifall"
        self._labels_path = None
        self._train_path = None
        self._val_path = None
        self._test_path = None

        self._train_batch_size = 4
        self._val_batch_size = 4
        self._test_batch_size = 4

        self._num_workers = 0

        self._fps = 5

    @property
    def train(self) -> bool:
        return self._train
    
    @property
    def test(self) -> bool:
        return self._test
    
    @property
    def split_format(self) -> str:
        return self._split_format
    
    @property
    def dataset_path(self) -> str:
        return self._dataset_path
    
    @property
    def train_batch_size(self) -> int:
        return self._train_batch_size
    
    @property
    def val_batch_size(self) -> int:
        return self._val_batch_size
    
    @property
    def test_batch_size(self) -> int:
        return self._test_batch_size
    
    @property
    def num_workers(self) -> int:
        return self._num_workers
    
    @property
    def fps(self) -> float:
        return self._fps


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")