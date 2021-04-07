import torch
import h5py

from config import TrainConfig


class Text2EmojiDataSet(torch.utils.data.Dataset):
    def __init__(self, size: int = None):
        self.h5dest = h5py.File(TrainConfig.H5DSET_PATH, 'r')['training']
        self.size = size

    def __len__(self):
        return self.size if self.size is not None else self.h5dest['word'].shape[0]

    def __getitem__(self, i):
        '''
        returns:
            (mask, word_embedding, sentence_embedding)
        '''
        return (
            self.h5dest['body'][i],
            self.h5dest['mask'][i],
            self.h5dest['word'][i],
            self.h5dest['sentence'][i],
        )


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.h5dest = h5py.File(TrainConfig.H5DSET_TEST_PATH, 'r')['training']

    def __len__(self):
        return self.h5dest['word'].shape[0]

    def __getitem__(self, i):
        '''
        returns:
            (body, mask, word_embedding, sentence_embedding)
        '''
        return (
            self.h5dest['body'][i],
            self.h5dest['mask'][i],
            self.h5dest['word'][i],
            self.h5dest['sentence'][i],
        )
