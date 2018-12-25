import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from statistics import mean
from statistics import median
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import signal

def lowPassFilter(dataFrame):
    array = dataFrame.values
    X = array[:, 0:178]
    b, a = signal.butter(3,0.01, 'highpass')
    output_signal = signal.filtfilt(b,a,X[0])
    print(output_signal)
    plt.plot(X[0])
    plt.show()
    plt.plot(output_signal)
    plt.show()


def normalization(X_input):
    scaler = MinMaxScaler()
    np_scaled = scaler.fit(X_input)
    MinMaxScaler(copy=True,feature_range =(-1,1))
    X_input = scaler.transform(X_input)
    return X_input

def standardization(X_input):
    scaler = StandardScaler()
    StandardScaler(copy=True, with_mean=True, with_std=True)
    np_scaled = scaler.fit(X_input)
    normalized = scaler.transform(X_input)
    return normalized


def featureSelection(dataFrame):
    array = dataFrame.values
    features = dataFrame.columns.values
    X = array[:,0:8]
    Y = array[:, 8]
    X = standardization(X)
    model = ExtraTreesClassifier()
    model.fit(X,Y)
    feature_weights = model.feature_importances_
    feature_index = 0
    selected_features = []
    for feature_weight in feature_weights:
        if feature_weight > 0.008:
            selected_features.append(features[feature_index])
        feature_index +=1
    return selected_features


def RunRandomForestClassifier(data,selected_features):
    print("Running RandomForest Classifier")
    X = data[selected_features]
    Y = data['y']
    X = standardization(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators = 600,max_depth =20)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
    precision, recall, fscore, support = score(y_test, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

def RunKNeighborsClassifier(data,selected_features):
    print("Running KNeighbors Classifier")
    X = data[selected_features]
    Y = data['y']
    X = standardization(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("KNN Accuracy:", metrics.accuracy_score(y_test, y_pred))
    precision, recall, fscore, support = score(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

def RunSVMClassifier(data, selected_features):
    print("running SVM Classifier")
    X = data[selected_features]
    X = standardization(X)
    Y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))

    precision, recall, fscore, support = score(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

def RunNaiveBayes(data, selected_features):

    X = data[selected_features]
    X = standardization(X)
    print(X)
    Y = data['y']
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred))

    precision, recall, fscore, support = score(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


def plotRoc(data, selected_features):
    X = data[selected_features]
    X = standardization(X)
    Y = data['y']
    Y = label_binarize(Y,classes =[1,2,3,4,5])
    n_class = Y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 10))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='blue',
             lw=lw, label='ROC curve for Class 1 (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='green',
             lw=lw, label='ROC curve for Class 2 (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='red',
             lw=lw, label='ROC curve for Class 3 (area = %0.2f)' % roc_auc[2])
    plt.plot(fpr[3], tpr[3], color='yellow',
             lw=lw, label='ROC curve for Class 4 (area = %0.2f)' % roc_auc[3])
    plt.plot(fpr[4], tpr[4], color='darkorange',
             lw=lw, label='ROC curve for Class 5 (area = %0.2f)' % roc_auc[4])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for KNN')
    plt.legend(loc="lower right")
    plt.show()

def featureExtraction(dataFrame):
    array = dataFrame.values
    X = array[:, 0:178]
    Y = array[:, 178]
    buffer = []
    for i in range(0,len(array)):
        X_mean = mean(X[i])
        X_median = median(X[i])
        X_minimum = min(X[i])
        X_maximum = max(X[i])
        X_skew = skew(X[i])
        X_kurt = kurtosis(X[i])
        X_q1 = np.percentile(X[i], 25)
        X_q3 = np.percentile(X[i], 75)
        buffer.append([X_mean,X_median,X_minimum,X_maximum,X_skew,X_kurt,X_q1,X_q3,Y[i]])
    df = pd.DataFrame(buffer, columns=['Mean', 'Median','Minimum','Maximum','Skew','Kurtosis','Q1','Q3','y'], dtype=float)
    return df

def combineClass(dataFrame):
    print("Combining classes")
    Y = dataFrame['y']
    for index in range(0,len(Y)):
        if Y[index] == 3.0:
            Y[index] = 2.0
        elif Y[index] == 4.0:
            Y[index] = 3.0
        elif Y[index] == 5.0:
            Y[index] = 3.0
    return Y

def knn_combined_performance(dataFrame, selected_features):
    X = dataFrame[selected_features]
    Y = combineClass(dataFrame)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = KNeighborsClassifier(n_neighbors = 17)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("KNN Accuracy:", metrics.accuracy_score(y_test, y_pred))
    precision, recall, fscore, support = score(y_test, y_pred)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    Y = label_binarize(Y,classes =[1,2,3])
    n_class = Y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 12))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='blue',
             lw=lw, label='ROC curve for Class 1 (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='green',
             lw=lw, label='ROC curve for Class 2 (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='red',
             lw=lw, label='ROC curve for Class 3 (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for Combined Class KNN')
    plt.legend(loc="lower right")
    plt.show()



def main():
    cwd = os.getcwd()
    filename = cwd + '\data\data.csv'
    dataFrame = pd.read_csv(filename)
    dataFrame = dataFrame.set_index('index')
    headers = list(dataFrame.columns.values)
    headers.remove('y')

    #Preprocessing step
    #lowPassFilter(dataFrame)
    #Performs Feature extraction
    extracted_dataframe = featureExtraction(dataFrame)
    #Performs Features Selection
    selected_features = featureSelection(extracted_dataframe)
    #plotRoc(extracted_dataframe, selected_features)

    #Classification and Validation Step
    #RunNaiveBayes(extracted_dataframe, selected_features)
    #RunRandomForestClassifier(extracted_dataframe,selected_features )
    #RunSVMClassifier(extracted_dataframe,selected_features )
    #RunKNeighborsClassifier(extracted_dataframe,selected_features )
    #knn_combined_performance(extracted_dataframe,selected_features)


if __name__ == "__main__":
    main()