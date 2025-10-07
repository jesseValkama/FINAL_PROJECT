from typing import List


class Settings:

    def __init__(self):
        
        """
        1.9 val, 1.63 train with ALL, 0.05 SMOOTH, 0.3 DROPOUT, 0.0005 DECAY, 128 HS
        1.87 val, 1.5 train with ALL, 0.0 SMOOTH, 0.3 DROPOUT, 0.0005 DECAY, 64 HS

        would augmentations even help? is lstm just not good enough?
        
        loss needs to be more dependent on the loss for fall and fallen

        CURRENT ASSUMPTIONS:
            HIGHER BATCH_SIZE SHOULD STABILISE
            PERHAPS SLIGHTLY HIHGER RES FOR MORE PRECISE PREDICTIONS
            LOWER HIDDEN SIZE SEEMS TO PREVENT OVERFITTING (HIGHER DROPOUT TOO?)
        """

        self._train = True
        self._test = True
        self._inference = True

        self._split_format = "cs-staged"
        self._dataset_path = "C:/Datasets/omnifall"
        self._weights_path = "weights"
        self._dataset_labels = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other"]

        self._yolo_model = "yolo11s-pose.pt"
        self._work_model = "work"
        self._test_model = "test"
        self._inference_model = "inference"

        self._train_batch_size = 12
        self._val_batch_size = 12
        self._test_batch_size = 12

        self._image_size = 320
        self._fps = 5 # fps for loading the videos, then later flatten according to video_length 
        self._video_length = 12 # frames

        self._num_workers = 4

        self._lstm_input_size = 17 * 2
        self._lstm_hidden_size = 64 
        self._lstm_num_layers = 2
        self._lstm_bias = True
        self._lstm_dropout_prob = 0.5
        self._lstm_bidirectional = True # enables bi-lstm

        self._min_epochs = 20
        self._max_epochs = 150
        self._early_stop_tries = 12
        self._validation_interval = 6

        self._learning_rate = 0.002
        self._weight_decay = 0.0005
        self._label_smoothing = 0.00
        self._cls_weights_factor = 1
        self._cls_ignore_thresh = 10

        self._amp = False # TODO: implement
        self._async_transfers = False # TODO: implement

        self._train_dev = "cuda:0"
        
        # credit for imagenet mean and stdev:
        #   https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self._mean = (0.485, 0.456, 0.406)
        self._standard_deviation = (0.229, 0.224, 0.225)

    @property
    def train(self) -> bool:
        return self._train
    
    @property
    def test(self) -> bool:
        return self._test
    
    @property
    def inference(self) -> bool:
        return self._inference
    
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
    def yolo_model(self) -> str:
        return self._yolo_model

    @property
    def dataset_labels(self) -> List[str]:
        return self._dataset_labels
    
    @property
    def work_model(self) -> str:
        return self._work_model
    
    @property
    def test_model(self) -> str:
        return self._test_model
    
    @property
    def inference_model(self) -> str:
        return self._inference_model
    
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
    
    @property
    def lstm_input_size(self) -> int:
        return self._lstm_input_size
    
    @property
    def lstm_hidden_size(self) -> int:
        return self._lstm_hidden_size

    @property
    def lstm_num_layers(self) -> int:
        return self._lstm_num_layers

    @property
    def lstm_bias(self) -> bool:
        return self._lstm_bias

    @property
    def lstm_dropout_prob(self) -> float:
        return self._lstm_dropout_prob

    @property
    def lstm_bidirectional(self) -> bool:
        return self._lstm_bidirectional

    @property
    def min_epochs(self) -> int:
        return self._min_epochs
    
    @property
    def early_stop_tries(self) -> int:
        return self._early_stop_tries
    
    @property
    def max_epochs(self) -> int:
        return self._max_epochs
    
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property
    def weight_decay(self) -> float:
        return self._weight_decay
    
    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
    
    @property
    def cls_weight_factor(self) -> int:
        return self._cls_weights_factor
    
    @property
    def cls_ignore_thresh(self) -> int:
        return self._cls_ignore_thresh
    
    @property
    def validation_interval(self) -> int:
        return self._validation_interval
    
    @property
    def amp(self) -> bool:
        return self._amp
    
    @property
    def async_transfers(self) -> bool:
        return self._async_transfers
    
    @property
    def train_dev(self) -> str:
        return self._train_dev


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")