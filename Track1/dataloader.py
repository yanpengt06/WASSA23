from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

"""
    dataloader.py: set Dataloader Class here
"""


def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('./data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('./data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec_dir = './data/%s/person_vect.pkl' % (dataset_name)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec


def get_SST2_loaders(dataset_name='IEMOCAP', batch_size=8, num_workers=0, pin_memory=False, args=None):
    print('building datasets..')
    trainset = SST2(dataset_name, 'train')
    devset = SST2(dataset_name, 'validation')
    testset = SST2(dataset_name, 'test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_WASSASingle_loaders(batch_size=8, num_workers=0, pin_memory=False, args=None, tokenizer=None):
    print('building datasets..')
    trainset = WASSASingle(split='tr', tokenizer=tokenizer)
    devset = WASSASingle(split='dev', tokenizer=tokenizer)
    testset = WASSASingle(split='te', tokenizer=tokenizer)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_WASSAContext_loaders(batch_size=8, num_workers=0, pin_memory=False, args=None, tokenizer=None):
    print('building datasets..')
    trainset = WASSAContext(split='train', tokenizer=tokenizer, args=args)
    devset = WASSAContext(split='dev', tokenizer=tokenizer, args=args)
    testset = WASSAContext(split='test', tokenizer=tokenizer, args=args)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(dataset_name='IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args=None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    trainset = IEMOCAPDataset(dataset_name, 'train', speaker_vocab, label_vocab, args)
    devset = IEMOCAPDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    train_sampler = get_train_valid_sampler(trainset)
    valid_sampler = get_train_valid_sampler(devset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(dataset_name, 'test', speaker_vocab, label_vocab, args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_SST2_loaders(dataset_name="sst2", num_workers=4)
    for batch in train_loader:
        break
    print(batch)
