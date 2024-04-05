import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import joblib



# SVM kernel type
KERNEL_TYPE_LINEAR = 'linear'
KERNEL_TYPE_LINEAR = 'rbf'



def getDataset(csvfile):
    data = pd.read_csv(csvfile)
    print('dataset from ', csvfile)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    print(X, y)
    return X, y

def splitDataset(X, y, test_ratio=0.2, seed=0, norm=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    # apply normalization (optional)
    if norm:
        X_train = normalize(X_train, norm="l2")
        X_test  = normalize(X_test, norm="l2")
    return X_train, X_test, y_train, y_test



import argparse
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-d", "--dataset", help=".csv dataset file's location", default='../datasets/banana.csv')
parser.add_argument("-s", "--seed", help="seed number for randomize the training dataset", default=0)

# Read config file (for camera source, model etc)
args = parser.parse_args()
file_dataset = args.dataset
seed = int(args.seed)



# Get dataset
X, y = getDataset(file_dataset)

# Split data & Preprocess
X_train, X_test, y_train, y_test = splitDataset(X, y, test_ratio=0.1, seed=seed, norm=True)

# Train SVM classifier
print('SVM training...')
model = svm.SVC(kernel=KERNEL_TYPE_LINEAR, C=1000)
model.fit(X_train, y_train)

# Predict with test data on the resulted model
y_pred = model.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Output the model to be used later
file_model = 'svm_model.joblib'
print('save model to ', file_model)
joblib.dump(model, file_model)

