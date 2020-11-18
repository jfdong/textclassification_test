import pandas as pd

toxic = pd.read_csv('./data/train.csv')

print(toxic.tweet[0:100])
