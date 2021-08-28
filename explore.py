import pandas as pd

dataset_all = pd.read_csv('data/dataset_all.csv')

dataset_all.head()

dataset_all['class'].value_counts()

matsnu = dataset_all.loc[dataset_all['class'] == 'matsnu']
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
##############

dataset_medium = pd.read_csv('data/umudga/csv/dataset_medium.csv')
dataset_medium.shape
dataset_medium.label.value_counts()

dataset_big_legit = pd.read_csv('data/umudga/csv/dataset_big_legit.csv')
dataset_big_legit.shape
dataset_big_legit.label.value_counts()

#Select uncommon rows

common_df = dataset_big_legit.merge(dataset_medium, on=['domain'])
common_df.shape

common_df.head()

#common_df = common_df.drop(['label_y'], axis=1)
# common_df.columns = ['domain', 'label']
# common_df.head()

diff_df = dataset_big_legit[~dataset_big_legit.domain.isin(common_df.domain)]

diff_df.head()

diff_df.shape

diff_df.label.value_counts()
new_legit = diff_df.sample(n=175000)

new_legit.shape
new_legit.label.value_counts()

new_legit.to_csv('data/umudga/csv/new_legit.csv', index=False)

r_legit = pd.read_csv('data/umudga/csv/new_legit.csv')
r_legit.label.value_counts()

r_legit.head(n=100)


test_df = dataset_medium.copy()

test_df.label.value_counts()

new_test_df = test_df.append(new_legit)

new_test_df.label.value_counts()
