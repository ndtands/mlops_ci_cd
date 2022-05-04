import warnings
warnings.filterwarnings('ignore')
#LOGGING
import logging
import time
import sys
# LOGGING SETUP
logger = logging.getLogger('Train')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(filename)s: %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
logging.Formatter.converter = time.gmtime
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from utils import *
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

logger.info('Step1. Load dataset and split dataset')
train = pd.read_csv('train_processed_transformed.csv')
train, dev = train_test_split(train,test_size= 0.3, stratify=train['target'])
test = pd.read_csv('test_processed_transformed.csv')
x_train = train.drop(['target'], axis = 1)
x_test =  test.drop(['target'], axis = 1)
y_train = train['target']
y_test = test['target']
x_dev = dev.drop(['target'], axis = 1)
y_dev = dev['target']

logger.info('Step2. Select feature')
# xác định roc-auc cho mỗi đặc trưng
roc_values = []
for feature in x_train.columns:
    # train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(x_train[feature].values.reshape(-1, 1), y_train)
    # obtain the predictions
    y_scored = clf.predict_proba(x_dev[feature].values.reshape(-1, 1))
    # calculate and store the roc-auc
    roc_values.append(roc_auc_score(y_dev, y_scored[:, 1]))
roc_values = pd.Series(roc_values)
roc_values.index = x_train.columns
selected_features_roc = roc_values[roc_values > 0.51].index
print(f'\t - ROC have {len(selected_features_roc)} feature')

sel_ = SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=10))
sel_.fit(x_train, y_train)
selected_feat_random = x_train.columns[(sel_.get_support())]
print(f'\t - Randomforest have {len(selected_feat_random)} feature')


logger.info('Step3. Experiment with some model')
logger.info('A. Logistic')
model = LogisticRegression()
logistic_all, logistic_roc, logistic_random, model = print_results(model, x_train, x_dev, x_test, y_train, y_dev, y_test, selected_features_roc, selected_feat_random)
probsLR = model.predict_proba(x_test[selected_feat_random])[:, 1]
fprLR, tprLR, thresholdsLR = metrics.roc_curve(y_test, probsLR)


logger.info('B. DecisionTre')
model = tree.DecisionTreeClassifier()
DT_all, DT_roc, DT_random, model = print_results(model, x_train, x_dev, x_test, y_train, y_dev, y_test, selected_features_roc, selected_feat_random)
probsDT = model.predict_proba(x_test[selected_feat_random])[:, 1]
fprDT, tprDT, thresholdsDT = metrics.roc_curve(y_test, probsDT)


logger.info('C. Adaboost')
model = AdaBoostClassifier()
Ada_all, Ada_roc, Ada_random, model = print_results(model, x_train, x_dev, x_test, y_train, y_dev, y_test, selected_features_roc, selected_feat_random)
probsAda = model.predict_proba(x_test[selected_feat_random])[:, 1]
fprAda, tprAda, thresholdsAda = metrics.roc_curve(y_test, probsAda)



logger.info('D. Random forest')
model = RandomForestClassifier()
Rf_all, Rf_roc, Rf_random, model = print_results(model, x_train, x_dev, x_test, y_train, y_dev, y_test, selected_features_roc, selected_feat_random)
probsRf = model.predict_proba(x_test[selected_feat_random])[:, 1]
fprRf, tprRf, thresholdsRf = metrics.roc_curve(y_test, probsRf)


logger.info('E. Gradient boosting')
model = GradientBoostingClassifier()
Gb_all, Gb_roc, Gb_random, model = print_results(model, x_train, x_dev, x_test, y_train, y_dev, y_test, selected_features_roc, selected_feat_random)
probsGb = model.predict_proba(x_test[selected_feat_random])[:, 1]
fprGb, tprGb, thresholdsGb = metrics.roc_curve(y_test, probsGb)

#Plot ROC
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(fprLR, tprLR, label = "LogReg")
axes.plot(fprDT, tprDT, label = "DT")
axes.plot(fprAda, tprAda, label = "Ada")
axes.plot(fprRf, tprRf, label = "Rf")
axes.plot(fprGb, tprGb, label = "Gb")
axes.set_xlabel("False positive rate")
axes.set_ylabel("True positive rate")
axes.set_title("ROC Curve for Logistic regression, Decession Tree, Adaboost, Gradient boost, Random forest")
axes.grid(which = 'major', c='#cccccc', linestyle='--', alpha=0.5)
axes.legend(shadow=True)
plt.savefig('ROC.png', dpi=120)
logger.info('Step4. Save ROC figure Done')

# - - - - - - - GENERATE METRICS FILE
with open("metrics.json", 'w') as outfile:
        json.dump(
        	{ "logistic_all_feauture"                   : logistic_all,
        	  "logistic_roc_feauture"                   : logistic_roc,
        	  "logistic_randomforest_feauture"          : logistic_random,
        	  "DecesionTree_all_feauture"                   : DT_all,
        	  "DecesionTree_roc_feauture"                   : DT_roc,
        	  "DecesionTree_randomforest_feauture"          : DT_random,
              "Adaboost_all_feauture"                   : Ada_all,
        	  "Adaboost_roc_feauture"                   : Ada_roc,
        	  "Adaboost_randomforest_feauture"          : Ada_random,
        	  "RandomForest_all_feauture"                   : Rf_all,
        	  "RandomForest_roc_feauture"                   : Rf_roc,
        	  "RandomForest_randomforest_feauture"          : Rf_random,
              "GradientBoost_all_feauture"                   : Gb_all,
        	  "GradientBoost_roc_feauture"                   : Gb_roc,
        	  "GradientBoost_randomforest_feauture"          : Gb_random,
              }, 
        	  outfile
        	)
logger.info('Step5. Save metrics Done')