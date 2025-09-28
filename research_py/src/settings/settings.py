from typing import List


class Settings:

    def __init__(self):
        self._train = True
        self._test = True

        self._split_format = "cs-staged"
        self._dataset_path = "D:/datasets/omnifall"
        self._weights_path = "research_py/weights/ultralytics"

        self._train_batch_size = 4
        self._val_batch_size = 4
        self._test_batch_size = 4

        self._image_size = 480
        self._video_length = 20 # frames

        self._num_workers = 0

        self._fps = 5
        
        #credit for imagenet mean and stdev:
        #  https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self._mean = (0.485, 0.456, 0.406)
        self._standard_deviation = (0.229, 0.224, 0.225)

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
    def weights_path(self) -> str:
        return self._weights_path
    
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
    def image_size(self) -> int:
        return self._image_size
    
    @property
    def video_length(self) -> int:
        return self._video_length

    @property
    def num_workers(self) -> int:
        return self._num_workers
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def mean(self) -> List[float]:
        return self._mean
    
    @property
    def standard_deviation(self) -> List[float]:
        return self._standard_deviation



if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")