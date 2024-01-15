from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, roc_curve, auc
import matplotlib as plt

train = pd.read_csv("/Users/advait/PycharmProjects/dropout_kaggle/Students_final.csv")

svm_model = svm.SVC(kernel="linear")
dropout = train.pop("Target")
x_train, x_test, y_train, y_test = train_test_split(train,dropout,test_size=0.2, random_state=10)
svm_model.fit(x_train,y_train)
y_pred = svm_model.predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
# plt.figure(figsize=(8, 5))
# plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()