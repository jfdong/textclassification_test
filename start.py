from torchtext import data

id_field = data.LabelField()
label = data.LabelField()
tweet = data.Field()

train_fields = [('id', id_field), ('label', label), ('tweet', tweet)]
test_fields = [('id', id_field), ('tweet', tweet)]

train_data, test_data = data.TabularDataset.splits(
    path = './data',
    train = 'train.csv',
    test = 'test.csv',
    format = 'csv',
    fields = train_fields
    )



for d in train_data[0:9]:
    print(d.__dict__.keys())
    print(d.__dict__.values())

for d in test_data[0:9]:
    print(d.__dict__.keys())
    print(d.__dict__.values())

#tweet.build_vocab(train_data)

#train_iterator = data.BucketIterator.splits(
#    (train_data), batch_size=2, device='cpu')


