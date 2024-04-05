import pandas as pd
from sklearn import svm
from sklearn.preprocessing import normalize
import joblib



def getInputFeatures(csvfile):
    data = pd.read_csv(csvfile)
    print('dataset from ', csvfile)
    X = data
    print(X)
    return X



import argparse
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-m", "--model", help=".joblib model's location", default='model.joblib')
parser.add_argument("-t", "--t", help="test feature to test model (.csv file)", default='test.csv')

# Read config file (for camera source, model etc)
args = parser.parse_args()
file_input = args.input
modelpath = args.model



# Get dataset
X = getInputFeatures(file_input)

# Normaiization
X_norm = normalize(X, norm="l2")

# Load SVM classifier
print('Load the SVM model from ', modelpath)
model = joblib.load(modelpath)

# Apply input to model
y_predict = model.predict(X_norm)
print("predict output={}", y_predict)
