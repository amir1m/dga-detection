import pandas as pd

dataset_all = pd.read_csv('data/dataset_all.csv')

dataset_all.head()

dataset_all['class'].value_counts()

matsnu = dataset_all.loc[dataset_all['class'] == 'cryptolocker']
matsnu.head()

m = matsnu['domain']

m = m.reset_index().drop(['index'], axis=1)
m.columns = ['host']

ramdo = dataset_all.loc[dataset_all['class'] == 'ramdo']
ramdo.head()

goz = pd.read_csv('data/goz.txt', header=None, sep='\t')
goz.head()
g = goz.copy()

mg = m.append(g, ignore_index=True)
mg

suppobox = pd.read_csv('data/suppobox.txt', header=None, sep='\t')
suppobox.head()

predictor.predict('subjectdirect.ru	')

predictor.predict
