from torch import nn
from typing import List


class Settings:

    def __init__(self):
        
        """
        TODO:
        implement sat
        calculate FLOPs in test
        implement inference with grad-cam

        test label smoothing:
            best: 0.0

        test different lstm sizes
            best: 64 is && 32 hs
        """
        self._project_dir = "D:/self-studies/bachelors_final_project/research_py" # required to be hard coded since vs code changes the dir -> if i debug the dir is different from if i run from the terminal
        self._dataset = "omnifall"
        self._train = True
        self._test = True
        self._inference = False

        self._split_format = "cs-staged"
        self._ucf101_path = "C:/Datasets"
        self._omnifall_path = "C:/Datasets/omnifall"
        self._weights_path = "weights"
        self._dataset_labels = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other"]
        # applied after the weighting based on sample sizes
        self._label_weights = [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self._work_model = "work"
        self._test_model = "experiment1"
        self._inference_model = "experiment1"

        self._train_batch_size = 20
        self._val_batch_size = 20
        self._test_batch_size = 20
        self._image_size = 224
        self._video_length = 12 # frames

        self._criterion = "sce"
        self._self_adaptive_training = False
        self._TRADES = False
        self._sce_alpha = 1
        self._sce_beta = 0.2
        self._rnn_type = nn.LSTM # DO NOT INIT HERE
        self._frozen_layers = 3
        self._lstm_input_size = 64
        self._lstm_hidden_size = 32
        self._lstm_num_layers = 1
        self._lstm_bias = True
        self._lstm_dropout_prob = 0.0
        self._lstm_bidirectional = False

        self._min_epochs = 20
        self._max_epochs = 4
        self._early_stop_tries = 8
        self._validation_interval = 2
        self._warmup_length = 4

        self._learning_rate = 0.001
        self._weight_decay = 0.0005
        self._label_smoothing = 0.0
        self._cls_weights_factor = 0.4
        self._cls_ignore_thresh = 10

        self._num_workers = 4
        self._amp = False # TODO: currently hardcoded
        self._async_transfers = True
        self._train_dev = "cuda:0"
        
        # credit for imagenet mean and stdev:
        #   https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self._mean = (0.485, 0.456, 0.406)
        self._standard_deviation = (0.229, 0.224, 0.225)

    @property
    def project_dir(self) -> str:
        return self._project_dir

    @property
    def train(self) -> bool:
        return self._train
    
    @train.setter
    def train(self, train) -> None:
        self._train = train
    
    @property
    def test(self) -> bool:
        return self._test
    
    @test.setter
    def test(self, test) -> None:
        self._test = test
    
    @property
    def inference(self) -> bool:
        return self._inference
    
    @inference.setter
    def inference(self, inference) -> None:
        self._inference = inference
    
    @property
    def split_format(self) -> str:
        return self._split_format
    
    @property
    def dataset(self) -> str:
        return self._dataset
    
    @dataset.setter
    def dataset(self, dataset) -> None:
        self._dataset = dataset

    @property
    def dataset_path(self) -> str:
        match self.dataset:
            case "omnifall":
                return self._omnifall_path
            case "ucf101":
                return self._ucf101_path
            case _:
                raise RuntimeError("Provide a valid dataset")
    
    @property
    def weights_path(self) -> str:
        return self._weights_path
    
    @property
    def dataset_labels(self) -> List[str]:
        return self._dataset_labels
    
    @property
    def label_weights(self) -> List[float]:
        return self._label_weights
    
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
    def mean(self) -> List[float]:
        return self._mean
    
    @property
    def standard_deviation(self) -> List[float]:
        return self._standard_deviation

    @property
    def rnn_type(self) -> nn.Module:
        return self._rnn_type
    
    @property
    def frozen_layers(self) -> int:
        return self._frozen_layers

    @property
    def criterion(self) -> str:
        return self._criterion
    
    @property
    def sce_alpha(self) -> float:
        return self._sce_alpha
    
    @property
    def sce_beta(self) -> float:
        return self._sce_beta

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
    def cls_weight_factor(self) -> float:
        return self._cls_weights_factor
    
    @property
    def cls_ignore_thresh(self) -> int:
        return self._cls_ignore_thresh
    
    @property
    def validation_interval(self) -> int:
        return self._validation_interval
    
    @property
    def warmup_length(self) -> int:
        return self._warmup_length

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