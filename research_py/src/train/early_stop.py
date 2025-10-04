class EarlyStop:

    def __init__(self, patience: int, tries: int) -> None:
        self._counter = 0
        self._patience = patience
        self._tries = tries

    def __call__(self, epoch: int, improvement: bool) -> bool:
        """
        Method to stop training

        Args:
            epoch: the current epoch
            improvement: whether the model improved last validation or not

        Returns:
            bool: Whether to stop training
        """
        if not improvement:
            self._counter += 1
            counter = self._counter if self._counter < self._tries else self._tries
            print(f"The model did not improve ({counter} / {self._tries})")

        if epoch < self._patience:
            return False
        elif epoch == self._patience:
            print("Early stopping activated")
        
        if self._counter == self._tries:
            print("Early stopping stopped training")
            return True
        

if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")