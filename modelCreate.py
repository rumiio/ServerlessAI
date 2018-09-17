from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import warnings
import boto3

s3 = boto3.resource('s3')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# get csv file from s3 bucket
#s3.Object('aiml-rumi', 'diabetes.data.csv').download_file('diabetes.data.csv')
# bucket = s3.Bucket('aiml-rumi')
# obj = bucket.Object('diabetes.data.csv')

#with open('diabetes.data.csv', 'wb') as data:
#    obj.download_fileobj(data)

# load data
dataset = loadtxt('diabetes.data.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.externals import joblib
joblib.dump(model, 'pima-indians-diabetes.pkl') 

model = joblib.load('pima-indians-diabetes.pkl') 

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



