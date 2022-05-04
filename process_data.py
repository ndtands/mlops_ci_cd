import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import category_encoders as ce
from feature_engine.discretisation import EqualFrequencyDiscretiser

#LOGGING
import logging
import time
import sys
# LOGGING SETUP
logger = logging.getLogger('Process data')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(filename)s: %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
logging.Formatter.converter = time.gmtime
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info('Step1. Load dataset')
defaultTest = pd.read_csv("test.csv")                # Importing the dataframe 'test.csv' from the appropriate folder
defaultTrain = pd.read_csv("train.csv")              # Importing the dataframe 'train.csv' from the appropriate folder


defaultTest.drop(['RISK_MM'], axis= 1, inplace=True)
defaultTrain.drop(['RISK_MM'], axis= 1, inplace=True)

assert len(defaultTest['RainTomorrow'].unique()) == 2, 'Test set have outline target'
assert len(defaultTrain['RainTomorrow'].unique()) == 2, 'Train set have outline target'

assert defaultTest['RainTomorrow'].isnull().sum() == 0, 'Test set have missing value'
assert defaultTrain['RainTomorrow'].isnull().sum() == 0, 'Train set have missing value'

numerical_test = [var for var in defaultTest.columns if defaultTest[var].dtype!='O']
numerical_train = [var for var in defaultTrain.columns if defaultTrain[var].dtype!='O']
assert len(numerical_test) == len(numerical_train), 'numerical missmatch train and test'

categorical_test = [col for col in defaultTest.columns if defaultTest[col].dtypes == 'O' and col != 'RainTomorrow']
categorical_train = [col for col in defaultTrain.columns if defaultTrain[col].dtypes == 'O' and  col != 'RainTomorrow']
assert len(categorical_test) == len(categorical_train), 'categorical missmatch train and test'

logger.info('Step2. Process dataset')

for col in numerical_test:
    col_median = defaultTrain[col].median()
    defaultTest[col].fillna(col_median, inplace=True)

for col in numerical_train:
    col_median = defaultTrain[col].median()
    defaultTrain[col].fillna(col_median, inplace=True)
print('\t - Fill missing value for numerical done.')

for var in categorical_test:
    defaultTest[var].fillna(defaultTrain[var].mode()[0], inplace=True)

for var in categorical_train:
    defaultTrain[var].fillna(defaultTrain[var].mode()[0], inplace=True)
print('\t - Fill missing value for categorical done.')

assert sum(defaultTest[numerical_test].isnull().sum()) + sum(defaultTrain[numerical_train].isnull().sum()) == 0, 'There is miss the value (Nummerical)'
assert sum(defaultTest[categorical_test].isnull().sum()) + sum(defaultTrain[categorical_train].isnull().sum()) == 0, 'There is miss the value (Categorical)'

def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [defaultTrain, defaultTest]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)


logger.info('Step3. Transform dataset')
x_train = defaultTrain.drop(['RainTomorrow'], axis = 1)
x_test = defaultTest.drop(['RainTomorrow'], axis = 1)
y_train = defaultTrain['RainTomorrow']
y_test = defaultTest['RainTomorrow']
le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
print('\t - Encoder Target')

encoder = ce.BinaryEncoder(cols=['RainToday'])
X_train_target = encoder.fit_transform(x_train)
X_test_target = encoder.transform(x_test)
print('\t - Encoder RainToday')

col_categorical = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
X_train_categorical = pd.concat([X_train_target[['RainToday_0', 'RainToday_1']],
                      pd.get_dummies(x_train[col_categorical])], axis= 1 )

X_test_categorical = pd.concat([X_test_target[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(x_test[col_categorical])], axis= 1 )
print('\t - Dummies cateforical')
disc = EqualFrequencyDiscretiser(q=10, variables = numerical_train)
disc.fit(x_train[numerical_train])
X_train_numerical = disc.transform(x_train[numerical_train])
X_test_numerical = disc.transform(x_test[numerical_test])
print('\t - Discretiser numerical')

train = pd.concat([X_train_numerical, X_train_categorical], axis=1)
test = pd.concat([X_test_numerical, X_test_categorical], axis=1)

logger.info('Step4. Filter dataset')
duplicated_feat = []
for i in range(0, len(x_train.columns)):
    col_1 = train.columns[i]
    for col_2 in train.columns[i + 1:]:
        if train[col_2].equals(train[col_1]):
            duplicated_feat.append(col_2)
train.drop(labels=duplicated_feat, axis=1, inplace=True)
test.drop(labels=duplicated_feat, axis=1, inplace=True)
print(f'\t - Filter duplicate feature. Have {len(duplicated_feat)} feature duplicated')

quasi_constant_feat = []
for feature in train.columns:
    predominant = (train[feature].value_counts() / np.float(
        len(train))).sort_values(ascending=False).values[0]
    if predominant > 0.998:
        quasi_constant_feat.append(feature)
train.drop(labels=quasi_constant_feat, axis=1, inplace=True)
test.drop(labels=quasi_constant_feat, axis=1, inplace=True)
print(f'\t - Filter quasi constant feature. Have {len(duplicated_feat)} feature quasi constant')

train.insert(loc=115,column='target', value = y_train)
test.insert(loc=115,column='target', value = y_test)

train.to_csv('train_processed_transformed.csv', encoding='utf-8')
test.to_csv('test_processed_transformed.csv', encoding='utf-8')
logger.info('Done transform and process dataset')