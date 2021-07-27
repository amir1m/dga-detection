SEED = 42
import numpy as np
np.random.seed(SEED)
import pandas as pd
from sklearn.model_selection import train_test_split

import ktrain
from ktrain import text


dga_df  = pd.read_csv("data/legit-dga_domains.csv")

dga_df.head()
dga_df['class'].value_counts()
dga_df['subclass'].value_counts()


dga_df_pp = dga_df.copy()
dga_df_pp['true_label'] = dga_df_pp['class'].replace("legit", 0)
dga_df_pp['true_label'] = dga_df_pp['true_label'].replace("dga", 1)

dga_df_pp

dga_df_pp = dga_df_pp.drop(['domain', 'class', 'subclass'], axis=1)

dga_df_pp
dga_df_pp['true_label'].value_counts()

X = dga_df_pp['host']

# get the labels
labels = dga_df_pp.true_label

# Split the data into train/test datasets
X_train, X_test, y_train, y_test = train_test_split(X,
                    labels,
                    test_size=0.30,
                    random_state=SEED)

# Count of label 0 and 1 in the training data set
print("Rows in X_train %d : " % len(X_train))
type(X_train.values.tolist())


y_train.value_counts()

# DistilBERT
model_name = 'distilbert-base-uncased'
t = text.Transformer(model_name, class_names=labels.unique(),
                     maxlen=500)

# Pre-process train and test dataset
train = t.preprocess_train(X_train.tolist(), y_train.to_list())

val = t.preprocess_test(X_test.tolist(), y_test.to_list())

# Tarin
model = t.get_classifier()
learner = ktrain.get_learner(model,
                       train_data=train,
                       val_data=val,
                       batch_size=8)

#learner.fit_onecycle(3e-5, 3)

predictor = ktrain.load_predictor('model/predictor_distillbert')

predictor.predict('14mefho2608f31tn33aa1uo0rl.com')

predictor.predict('aadjsbkprnuu1111.ru')
