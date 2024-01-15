import boost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,f1_score,roc_auc_score

train = pd.read_csv("/Users/advait/PycharmProjects/dropout_kaggle/Students_final.csv")

dropout = train.pop("Target")
x_train, x_test, y_train, y_test = train_test_split(train,dropout,test_size=0.2, random_state=10)
clf = xgb.XGBClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(y_test)
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))