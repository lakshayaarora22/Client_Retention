import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("tel_churn.csv")
df = df.drop('Unnamed: 0', axis=1)
x = df.drop('Churn', axis=1)
y = df['Churn']
sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2)
sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(x, y)
xr_train1, xr_test1, yr_train1, yr_test1 = train_test_split(X_resampled1, y_resampled1, test_size=0.2)
model_rf_smote = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_rf_smote.fit(xr_train1, yr_train1)
yr_predict1 = model_rf_smote.predict(xr_test1)
model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)
print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))
print(metrics.confusion_matrix(yr_test1, yr_predict1))
filename = 'model.sav'
pickle.dump(model_rf_smote, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
model_score_r1 = load_model.score(xr_test1, yr_test1)
print(model_score_r1)
