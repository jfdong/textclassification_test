import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.datasets.text_classification import URLS
from torchtext.datasets import text_classification
from torchtext.data.utils import get_tokenizer
import pdb
#import text_pipeline

def _create_data_with_sp_transform(data_path):

    data = []
    labels = []
    #spm_path = pretrained_sp_model['text_unigram_15000']
    #text_pipeline = sentencepiece_processor(download_from_url(spm_path))
    import pandas as pd

    toxic = pd.read_csv(data_path)
    tokenizer = get_tokenizer("basic_english")

    print(toxic.tweet[0:10])
    for i in range(len(toxic.tweet)):
        #pdb.set_trace()
           
            #token_ids = text_pipeline(corpus)
        label = int(toxic.label[i])
        token_ids = tokenizer(toxic.tweet[i])
    
        data.append((label, torch.tensor(token_ids)))
        labels.append(label)
    return data, set(labels)


def setup_datasets(dataset_name, root='.data', vocab_size=20000, include_unk=False):

    train_csv_path = './.data/hackson/train.csv'
    test_csv_path = './.data/hackson/test.csv'
    train_data, train_labels = _create_data_with_sp_transform(train_csv_path)
    test_data, test_labels = _create_data_with_sp_transform(test_csv_path)

    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (text_classification.TextClassificationDataset(None, train_data, train_labels),
            text_classification.TextClassificationDataset(None, test_data, test_labels))
