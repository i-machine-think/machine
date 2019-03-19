import torch
import torchtext


def get_standard_iter(data, batch_size=64, device=None):
    """
    Helper function to get the batch iter from a torchtext dataset
    Args:
        data (torchtext Dataset)
        batch_size (int, optional)
        device (torch.device, optional): if need to force data
                                        to be run on specific device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False)
