from tinygrad.tensor import Tensor
from tinygrad import dtypes
from model.tokenizer import Tokenizer


class NextTokenPredictionDataset:
    def __init__(self, input_file: str, context_size: int, tokenizer: Tokenizer) -> None:
        self.context_size = context_size

        # load data in memory
        data = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                data.extend(tokenizer.encode(line, end_of_string=True))

        self.data = Tensor(data, dtype=dtypes.long)

    def __len__(self) -> int:
        return len(self.data) - (self.context_size + 1)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx : idx + self.context_size], self.data[idx + 1 : idx + self.context_size + 1]