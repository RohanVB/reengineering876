import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from statistics import mean
from statistics import median
from statistics import stdev
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

def combineDataframes(dataFrame1,dataFrame2):
    print("Combining data frames")
    frames = [dataFrame1,dataFrame2]
    combined_df = pd.concat(frames)
    return combined_df

def naiveBayes(dataFrame):
    X = dataFrame[['Ch1 mean', 'Ch1 median', 'Ch1 stdev', 'Ch2 mean', 'Ch2 median', 'Ch2 stdev', 'Ch3 mean', 'Ch3 median', 'Ch3 stdev',
         'Ch4 mean', 'Ch4 median', 'Ch4 stdev', 'Ch5 mean', 'Ch5 median', 'Ch5 stdev', 'Ch6 mean', 'Ch6 median', 'Ch6 stdev',
         'Ch7 mean', 'Ch7 median', 'Ch7 stdev', 'Ch8 mean', 'Ch8 median', 'Ch8 stdev']]
    Y = dataFrame['Action Type']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_pred, y_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def svm_linear(dataFrame):
    X = dataFrame[['Ch1 mean', 'Ch1 median', 'Ch1 stdev', 'Ch2 mean', 'Ch2 median', 'Ch2 stdev', 'Ch3 mean', 'Ch3 median', 'Ch3 stdev',
         'Ch4 mean', 'Ch4 median', 'Ch4 stdev', 'Ch5 mean', 'Ch5 median', 'Ch5 stdev', 'Ch6 mean', 'Ch6 median', 'Ch6 stdev',
         'Ch7 mean', 'Ch7 median', 'Ch7 stdev', 'Ch8 mean', 'Ch8 median', 'Ch8 stdev']]
    Y = dataFrame['Action Type']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    svml = LinearSVC()
    svml.fit(X_train, y_train)
    y_pred = svml.predict(X_test)
    print("Linear SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))

def svm_kernel(dataFrame):
    X = dataFrame[['Ch1 mean', 'Ch1 median', 'Ch1 stdev', 'Ch2 mean', 'Ch2 median', 'Ch2 stdev', 'Ch3 mean', 'Ch3 median', 'Ch3 stdev',
         'Ch4 mean', 'Ch4 median', 'Ch4 stdev', 'Ch5 mean', 'Ch5 median', 'Ch5 stdev', 'Ch6 mean', 'Ch6 median', 'Ch6 stdev',
         'Ch7 mean', 'Ch7 median', 'Ch7 stdev', 'Ch8 mean', 'Ch8 median', 'Ch8 stdev']]
    Y = dataFrame['Action Type']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    svm = SVC( kernel= 'poly', gamma='scale')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_pred, y_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
def k_nearest(dataFrame):
    X = dataFrame[['Ch1 mean', 'Ch1 median', 'Ch1 stdev', 'Ch2 mean', 'Ch2 median', 'Ch2 stdev', 'Ch3 mean', 'Ch3 median', 'Ch3 stdev',
         'Ch4 mean', 'Ch4 median', 'Ch4 stdev', 'Ch5 mean', 'Ch5 median', 'Ch5 stdev', 'Ch6 mean', 'Ch6 median', 'Ch6 stdev',
         'Ch7 mean', 'Ch7 median', 'Ch7 stdev', 'Ch8 mean', 'Ch8 median', 'Ch8 stdev']]
    Y = dataFrame['Action Type']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("K Nearest Neighbour Accuracy:", metrics.accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_pred, y_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
def rollingWindowFeatureExtraction(dataFrame,action_type):
    print("Running Feature Extraction")
    window_size = 300
    ch1median =[]
    ch1mean = []
    ch1stdev = []
    ch2median =[]
    ch2mean = []
    ch2stdev = []
    ch3median =[]
    ch3mean = []
    ch3stdev = []
    ch4median =[]
    ch4mean = []
    ch4stdev = []
    ch5median =[]
    ch5mean = []
    ch5stdev = []
    ch6median =[]
    ch6mean = []
    ch6stdev = []
    ch7median =[]
    ch7mean = []
    ch7stdev = []
    ch8median =[]
    ch8mean = []
    ch8stdev = []
    d = {}
    label = action_type*(np.ones(2000))           # crate label list
    for i in range(0,180000,90):
        ch1mean.append(mean(dataFrame[i:i+window_size][0]))
        ch1median.append(median(dataFrame[i:i + window_size][0]))
        ch1stdev.append(stdev(dataFrame[i:i +window_size][0]))

        ch2mean.append(mean(dataFrame[i:i+window_size][1]))
        ch2median.append(median(dataFrame[i:i + window_size][1]))
        ch2stdev.append(stdev(dataFrame[i:i + window_size][1]))

        ch3mean.append(mean(dataFrame[i:i+window_size][2]))
        ch3median.append(median(dataFrame[i:i + window_size][2]))
        ch3stdev.append(stdev(dataFrame[i:i + window_size][2]))

        ch4mean.append(mean(dataFrame[i:i+window_size][3]))
        ch4median.append(median(dataFrame[i:i + window_size][3]))
        ch4stdev.append(stdev(dataFrame[i:i + window_size][3]))

        ch5mean.append(mean(dataFrame[i:i+window_size][4]))
        ch5median.append(median(dataFrame[i:i + window_size][4]))
        ch5stdev.append(stdev(dataFrame[i:i + window_size][4]))

        ch6mean.append(mean(dataFrame[i:i+window_size][5]))
        ch6median.append(median(dataFrame[i:i + window_size][5]))
        ch6stdev.append(stdev(dataFrame[i:i + window_size][5]))

        ch7mean.append(mean(dataFrame[i:i+window_size][6]))
        ch7median.append(median(dataFrame[i:i + window_size][6]))
        ch7stdev.append(stdev(dataFrame[i:i + window_size][6]))

        ch8mean.append(mean(dataFrame[i:i+window_size][7]))
        ch8median.append(median(dataFrame[i:i + window_size][7]))
        ch8stdev.append(stdev(dataFrame[i:i + window_size][7]))

    d = {'Ch1 mean': ch1mean, 'Ch1 median': ch1median,'Ch1 stdev': ch1stdev, 'Ch2 mean': ch2mean, 'Ch2 median': ch2median, 'Ch2 stdev': ch2stdev, 'Ch3 mean': ch3mean, 'Ch3 median': ch3median, 'Ch3 stdev': ch3stdev,
         'Ch4 mean': ch4mean, 'Ch4 median': ch4median, 'Ch4 stdev': ch4stdev, 'Ch5 mean': ch5mean, 'Ch5 median': ch5median, 'Ch5 stdev': ch5stdev, 'Ch6 mean': ch6mean, 'Ch6 median': ch6median, 'Ch6 stdev': ch6stdev,
         'Ch7 mean': ch7mean, 'Ch7 median': ch7median, 'Ch7 stdev': ch7stdev, 'Ch8 mean': ch8mean, 'Ch8 median': ch8median, 'Ch8 stdev': ch8stdev, 'Action Type': label}
    dataFrame = pd.DataFrame(d)
    return dataFrame

def normalizer(DataFrame):
    transformer = Normalizer().fit(DataFrame)  # fit does nothing.
    x = transformer.transform(DataFrame)
    return x

def standardization_maxabs(DataFrame):
    max_abs_scaler = MaxAbsScaler().fit(DataFrame)
    X_test_maxabs = max_abs_scaler.transform(DataFrame)
    return(X_test_maxabs)

def standardization_zscore(DataFrame):
    scaler = StandardScaler().fit(DataFrame)
    X_test_z = scaler.transform(DataFrame)
    return(X_test_z)

def feature_selection(DataFrame):
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(DataFrame)
    # pac.transform(DataFrame)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

def main():
    cwd = os.getcwd()
    filename_aggressive = cwd + '\\data\\Aggressive.txt'
    filename_normal = cwd + '\\data\\Normal.txt'
    dataFrame_agg0 = pd.read_csv(filename_aggressive,sep='\t',header=None,nrows=180000)
    dataFrame_norm0 = pd.read_csv(filename_normal, sep='\t', header=None, nrows=180000)

    # action_type 1 is for aggressive, action_type 0 is for normal
    action_type = 1
#    stand_agg = standardization_maxabs(dataFrame_agg0)
    dataFrame_agg1 = rollingWindowFeatureExtraction(dataFrame_agg0,action_type)

    # plt(dataFrame_agg1)
    # plt(dataFrame_agg0)

    action_type =0
 #   stand_nor = standardization_maxabs(dataFrame_norm0)
    dataFrame_norm1 = rollingWindowFeatureExtraction(dataFrame_norm0, action_type)

    Combined_df = combineDataframes(dataFrame_agg1,dataFrame_norm1)
    # naiveBayes(Combined_df)

    # print(Combined_df)
    # print(Combined_df.shape)
    # feature_selection(Combined_df)
    #
    naiveBayes(Combined_df)
    k_nearest(Combined_df)
    # svm_linear(Combined_df)
    svm_kernel(Combined_df)


#    print(Combined_df)
#    x = normalizer(Combined_df)
#    stand_Combined_df = standardization_zscore(Combined_df)
#    print(stand_Combined_df)

#    svm_linear(Combined_df）


#      print(x)
#     x = standardization_zscore(Combined_df)


if __name__ == "__main__":
    main()
