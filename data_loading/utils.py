import torch
from torch.nn.utils.rnn import pad_sequence

# this is like `transform` in StressDataset, as a preprocessing step
def pad_x_tensors(batch):
        """
        Args:
        - batch: a list of tuples of length batch_size (determined by Dataloader), 
                where each tuple is (x_tensor, y_tensor) (determined by Dataset.__getitem__())
        """
        # Extract x_tensor and y_tensor from each sample in the batch
        x_tensors = [sample[0] for sample in batch]
        y_tensors = torch.tensor([sample[1] for sample in batch])
        x_tensors_seq_len = torch.tensor([len(t) for t in x_tensors])

        # Pad sequences to make them of equal length
        padded_x_tensors = pad_sequence(x_tensors, batch_first=True, padding_value=0)

        return  padded_x_tensors, y_tensors, x_tensors_seq_len