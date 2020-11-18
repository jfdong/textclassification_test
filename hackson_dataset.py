import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.datasets.text_classification import URLS
from torchtext.datasets import text_classification
from torchtext.data.utils import get_tokenizer
import pdb
#import text_pipeline
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab


def _create_data_with_sp_transform(data_path):
    data = []
    labels = []
    import pandas as pd

    toxic = pd.read_csv(data_path)
    tokenizer = get_tokenizer("basic_english")

    # build vocab
    vocab = {}
    id = 0
    for i in range(len(toxic.tweet)):
        tokens = set(tokenizer(toxic.tweet[i]))
        for token in tokens:
            if token not in vocab.keys():
                vocab[token] = id
                id += 1
    
    for i in range(len(toxic.tweet)):
        #pdb.set_trace()
        label = int(toxic.label[i])
        tokens = set(tokenizer(toxic.tweet[i]))
        token_ids = list([vocab[token] for token in tokens])
    
        data.append((label, torch.tensor(token_ids)))
        labels.append(label)
    return data, set(labels)


def setup_datasets(dataset_name, root='.data', vocab_size=20000, include_unk=False):

    train_csv_path = './.data/hackson/train.csv'
    test_csv_path = './.data/hackson/test_withsomelabels.csv'
       
    train_data, train_labels = _create_data_with_sp_transform(train_csv_path)
    test_data, test_labels = _create_data_with_sp_transform(test_csv_path)
    #pdb.set_trace()

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (text_classification.TextClassificationDataset(None, train_data, train_labels),
            text_classification.TextClassificationDataset(None, test_data, test_labels))
