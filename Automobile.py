'''
author: Karteek Pradyumna Bulusu
Highly imbalanced and huge dimensional data with a moto to analyze the data and determine the probability of
deals that are captured. [Deals captured being the class]
'''

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score

# Read the CSV file into dataframe
inputFile = pd.read_csv('postPreprocessing.csv')
dataFrame = pd.DataFrame(inputFile)
notNulldataFrame = dataFrame.dropna()  # Null values are ignored for analysis.

# Creating matrix holding features in separate matrix and label in different.
inputMatrix = np.matrix(notNulldataFrame)
featuresMatrix = inputMatrix[:, :30]  # Features
targetMatrix = inputMatrix[:, 31]  # Target
warnings.filterwarnings('ignore')

print("Dimension of the data post data manipulations in MS Excel. [Excluding Class attribute]")
print(featuresMatrix.shape)
global NewDimension

# featureImportanceAndSelection() takes all attributes and target labels into it.
# Calculates feature importance and removes all irrelevant attributes.
# The resultant data in matrix form is passed to detectOutlier().
def featureImportanceAndSelection(Data, Target):
    print("\nFeature Selection process taking place.")
    classifier = ExtraTreesClassifier()
    classifier = classifier.fit(featuresMatrix, targetMatrix)
    classifier.feature_importances_
    #classifier = RandomForestClassifier()
    #classifier = classifier.fit(Data, Target)
    global NewDatMatrix
    global NewDimension
    model = SelectFromModel(classifier, prefit=True)
    NewDatMatrix = model.transform(featuresMatrix)
    print("Dimension of the dataframe after Feature Selection. [Excluding Class attribute]")
    print(NewDatMatrix.shape)
    NewDimension = NewDatMatrix.shape
    return NewDatMatrix

featureImportanceAndSelection(featuresMatrix, targetMatrix)

global aRow
aRow = []

# ifoutlier() receives 'value', Quartile 1 and Quartile 3 value for each feature from detectoutlier() method.
# ifOutlier calculates inner and outer bound for given Q1 and Q3 (This is Inter-Quartile Range (IQR).
# returns True if the entry exists in the range (Outlier).
# returns False if it doesn't exist in the range (Not an outlier).
def ifoutlier(value, Q1, Q3):
    lowerBound = Q1 - 3*(Q3 - Q1)
    upperBound = Q3 + 3*(Q3 - Q1)
    return value < lowerBound or value > upperBound

# detectoutlier() takes one feature at a time from the data matrix.
# Calculates Quartile 1 and Quartile 3 for every feature.
# Passes the value along with Q1 and Q3 to ifoutlier() to check if value is outlier.
# If the value IS OUTLIER, it is replaced with 'Z' and added to list aRow which is later removed from the data frame.
# If the value IS NOT outlier, value is added as such to the list aRow.
def detectoutlier(featureList):
    q1 = np.percentile(featureList, 25)
    q3 = np.percentile(featureList, 75)
    count = 0
    for val in featureList:
        # True if outlier exists.
        if ifoutlier(val, q1, q3):   # Traverse to ifoutlier() method.
            count += 1
            aRow.append('Z')  # outlier entry replaced with 'Z' and appended to list.
        else:
            aRow.append(val)  # Value is appended if it is not an Outlier.
    return aRow

print("\nDetecting outliers in every column of the data")
print("Using Inter-Quartile Range technique to identify outliers ")

# For each column in the data frame, the entries are passed to detectoutlier() function.
for n in range(0, NewDimension[1]):
    listColumnEntry = NewDatMatrix[:, n].tolist()
    detectoutlier(listColumnEntry)

# aRow is single dimensional list. It is reshaped into the dimension of the FeaturesMatrix
aRowMatrix = np.matrix(aRow)   # List converted to matrix
aRowMatrix = np.reshape(aRowMatrix, (NewDimension[1], NewDimension[0]))   # Reshaped to the form of actual data.
aRowMatrix = np.transpose(aRowMatrix)   # Transposed since it was initially in the form of (column, row)


# 'CompleteData' ignores all the rows containing 'Z' which are outliers.
# Removing all the outliers since it would tamper the analysis results.
TotalData = np.concatenate((aRowMatrix, targetMatrix), axis=1)
CompleteData = TotalData[np.all(TotalData != 'Z', axis=1)]
print('\nDataFrame is combined with the Class')
print("\nOutliers are removed from the data.\n")
print("Dimension of the data post outlier removal is\n", CompleteData.shape)

# findROC() takes the Actual Target values and predicted target values obtained from BaggingClassifierPhase() function.
# Calculates the area under curve using roc_auc_score function.
def findROC(target, predicted):
    TargetValues = []
    # Changing the format of the entry from '0.0' and '1.0' to 0 and 1 since roc_auc_score accept only binary entries.
    for i in target:
        if i == '0.0':
            TargetValues.append(0)
        else:
            TargetValues.append(1)

    PredictValues = []
    for i in predicted:
        if i == '0.0':
            PredictValues.append(0)
        else:
            PredictValues.append(1)

    AreaUnderCurve = roc_auc_score(TargetValues, PredictValues)
    print("\nArear under curve: ", AreaUnderCurve)
    return AreaUnderCurve

# BaggingClassifierPhase() takes the clean data, creates a Bagging Classifier over the dataset.
# BaggingClassifier is an ensemble method which creates subsets and fits for each subset using fit() function.
# predict() function will predict label for each of the fit models and average the prediction.
# f1_score: is based on the precision and recall of the model. Tells the accuracy of the classifier.
# Confusion matrix: Displays the True positive, True negative, False positive, False negative values.
# Probability: probability of occurrence of Flag = 1 is calculated using predict_proba() method.
def BaggingClassifierPhase(fullData):
    Features = fullData[:, 0:NewDimension[1]-1].astype(float)
    Target = fullData[:, NewDimension[1]]
    Bagmodel = BaggingClassifier()
    #Bagmodel = RandomForestClassifier()
    Bagmodel.fit(Features, Target)
    print("Bagging Classifier is built and making predictions and calculating score of the model.")
    TargetMat = np.ravel(Target)  # Converts into 1D numpy.array
    predicted = Bagmodel.predict(Features)
    print("\nf1_score:\n", metrics.f1_score(Target, predicted, pos_label='1.0', average='weighted'))
    print("\nConfusion matrix:\n    0.0\t  1.0\n", metrics.confusion_matrix(TargetMat, predicted))
    print("Calculating the score of the model for 10-folds.")
    # Similar scores are obtained for every fold which shows that the model is well built and is NOT overfitting.
    print("\nScore of the model for each of the 10 folds:\n", cross_val_score(Bagmodel, Features, TargetMat, cv=10))
    findROC(TargetMat, predicted)
    probability = Bagmodel.predict_proba(Features)
    listProb = probability[:, 1].tolist()  # List of probability values for Flag = 0 for every observation.
    sumProb = 0
    for z in listProb:
        sumProb += z
    AverageProbability = sumProb/len(listProb)
    print("\nAverage probability of deals being Captured is: ", AverageProbability)
    return AverageProbability

BaggingClassifierPhase(CompleteData)
