import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

import matplotlib.pyplot as plt

train = pd.read_csv("/Users/advait/PycharmProjects/dropout_kaggle/Students_final.csv")

n_stim_arr = []
f1_arr = []
auc_arr = []

rf = RandomForestClassifier(n_estimators=35, max_depth=15, random_state=10)
dropout = train.pop("Target")
x_train, x_test, y_train, y_test = train_test_split(train,dropout,test_size=0.2, random_state=10)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
    # print(x_test)
    # print(y_test,y_pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
f1 = f1_score(y_test,y_pred)
print("F1 score: ",f1_score(y_test,y_pred))
# print(roc_auc)
print("Mean square error: ",mean_squared_error(y_test,y_pred))
print("roc auc score: ",roc_auc_score(y_test,y_pred))
feature_importances = rf.feature_importances_
individual_trees = rf.estimators_
# print(individual_trees)
print(feature_importances)
plt.figure(figsize=(7, 5))
plt.plot(false_positive_rate, true_positive_rate, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()