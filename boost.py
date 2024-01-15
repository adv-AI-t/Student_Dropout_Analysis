from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,f1_score,roc_auc_score
import pandas as pd
import xgboost as xgb

train = pd.read_csv("/Users/advait/PycharmProjects/dropout_kaggle/Students_final.csv")
dropout = pd.read_csv("/Users/advait/PycharmProjects/dropout_kaggle/dropout.csv")
# print(train)
# dropout = train.pop("Target")
# print(train)
x_train, x_test, y_train, y_test = train_test_split(train,dropout,test_size=0.2, random_state=10)
model = xgb.XGBClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(y_test)
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))