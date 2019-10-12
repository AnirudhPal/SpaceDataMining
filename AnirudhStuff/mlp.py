# Import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

## Config Vars
VERBOSE = True
RANDOM_STATE = 47
FRAC = 0.2
remFields = ['VER', 'EDATE', 'BIRD', 'STIMEQ', 'STIMEL', 'LATQ', 'EW',
'LONQ', 'ADIAG', 'ACOMMENT', 'SPIN', 'ORBIT']

# Helper Funcs
# Load Data Set
def loadCSV(filename):
    # Get Data Frame
    data_frame = pd.read_csv(filename)

    # Print Data Set
    if VERBOSE:
        print('loadCSV():')
        print(data_frame.head())
        print('...')
        print(data_frame.tail())
        print('')

    # Return Data Frame
    return data_frame

# Store Data Set
def storeCSV(filename, data_frame):
    # Store to File
    data_frame.to_csv(filename, index = False)

# Remove Irrelevant Data and Parse
def preProcess(data_frame):
    # Remove Useless Fields
    for i in remFields:
        data_frame = data_frame.drop([i], axis=1)

    # Split Date
    if 'ADATE' not in remFields:
        data_frame['AYEAR'] = pd.DatetimeIndex(data_frame['ADATE']).year
        data_frame['AMONTH'] = pd.DatetimeIndex(data_frame['ADATE']).month
        data_frame['ADAY'] = pd.DatetimeIndex(data_frame['ADATE']).day
        data_frame = data_frame.drop(['ADATE'], axis=1)

    # Change Lat
    if 'LAT' and 'NS' not in remFields:
        # Iterate
        for sat in data_frame.index:
            # If South
            if(data_frame.loc[sat, 'NS'] == 'S'):
                data_frame.loc[sat, 'LAT'] = - data_frame.loc[sat, 'LAT']
        data_frame = data_frame.drop(['NS'], axis=1)

    # Change NaN
    if 'STIMEU' not in remFields:
        # Iterate
        for sat in data_frame.index:
            # If South
            if(data_frame.loc[sat, 'STIMEU'] == 9999):
                data_frame.loc[sat, 'STIMEU'] = np.NaN;

    # Drop NaN
    data_frame = data_frame.dropna()

    # Print Data Set
    if VERBOSE:
        print('preProcess():')
        print(data_frame.head())
        print('...')
        print(data_frame.tail())
        print('')

    return data_frame

# Run MLP
def mlp(data_frame):
    # Get Labels
    anomalies = data_frame['ATYPE']

    # Strip Anomaly
    data = data_frame.drop(['ATYPE'], axis=1)

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(data, anomalies, test_size = FRAC, random_state = RANDOM_STATE)

    # Create a Classifier
    clf = MLPClassifier(hidden_layer_sizes = (1000,1000,1000), max_iter = 500, alpha=0.0001,
                        solver = 'sgd', verbose = 10,  random_state = RANDOM_STATE, tol = 0.000000001)

    # Train Classifier
    clf.fit(x_train, y_train)

    # Test Classifier
    y_pred = clf.predict(x_test)


    print(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

# Main
def main():
    # Import Dataset (Change This to File Name)
    data_set = loadCSV(r'C:\Users\mailb\Documents\GitHub\SpaceDataMining\AnirudhStuff\Data.csv')

    # Pre Process Data
    data_set_processed = preProcess(data_set)

    # Store in File
    storeCSV('DataClean.csv', data_set_processed)

    # Run MLP
    mlp(data_set_processed)

main()
