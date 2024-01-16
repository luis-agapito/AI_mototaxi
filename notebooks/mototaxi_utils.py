import math
import matplotlib.pyplot as plt
import warnings

from torch._utils import _accumulate
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm, Generator
from typing import (List, Optional, Sequence, TypeVar, Union)

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

#Based on Pytorch 2.1.2 torch.utils.data.Subset()
class CustomSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            #return [self.dataset[self.indices[idx]] for idx in indices]
            return [[self.dataset[self.indices[idx]], self.indices[idx]] for idx in indices]
            #return [self.dataset[self.indices[idx]] for idx in indices], [self.indices[idx] for idx in indices]

    def __len__(self):
        return len(self.indices)

#Based on Pytorch 2.1.2 torch.utils.data.random_splits()
def custom_random_split(dataset: Dataset[T], lengths: Sequence[Union[int, float]],
                        generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    return [CustomSubset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def plot_history(history, num_epochs=None):
    if num_epochs is None:
        #tensorflow
        train_accuracy = [0.] + history['accuracy']
        train_loss = history['loss']
        val_accuracy = [0.] + history['val_accuracy']
        val_loss = history['val_loss']
        num_epochs = len(history['accuracy'])
    else:
        #pytorch
        train_accuracy = [0.] + history['train']['accuracy']
        train_loss = [0.] + history['train']['loss']
        val_accuracy = [0.] + history['val']['accuracy']
        val_loss = [0.] + history['val']['loss']

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_accuracy, '-o', label='training set')
    plt.plot(val_accuracy, '-o', label='validation set')
    plt.ylim([0, 1.])
    plt.xlim([0, num_epochs])
    plt.legend()
    plt.ylabel('Accuracy')
    plt.setp(plt.gca(), xticklabels=[])

    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_epochs+1), train_loss, '-o', label='training set')
    plt.plot(range(1, num_epochs+1), val_loss, '-o', label='validation set')
    plt.ylim([0, 1.])
    plt.xlim([0, num_epochs])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

