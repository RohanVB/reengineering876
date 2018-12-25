import numpy as np
import pandas as pd
from sklearn import preprocessing as prepro
from sklearn.decomposition import PCA
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class ClinicalDataSet:
    fileName = './ClinicalDemogData_COFL_bak.xlsx'
    nonfallerOrg = None
    fallerOrg = None
    completeDataset = None
    normalizedDataset = None
    PCADataset = None

    def readDateSet(self):
        self.nonfallerOrg = pd.read_excel(self.fileName)
        self.fallerOrg = pd.read_excel(self.fileName, sheet_name='Fallers', sort=True)

    def connetDataset(self):
        self.completeDataset = pd.concat([self.fallerOrg, self.nonfallerOrg])

    '''
    finish the functions below
    1. drop columns date, yr almost
    2. change the content in column "#" to 1(faller) or 0(nonfaller)

    '''

    def preprocessFaller(self):
        self.fallerOrg.drop('Date', axis=1, inplace=True)
        self.fallerOrg.drop('yr almost', axis=1, inplace=True)
        self.fallerOrg['#'].values[:] = 1  # 1 means faller
        self.fallerOrg['#'] = self.fallerOrg['#'].astype(int)  # change type object to int
        cols = self.fallerOrg.columns.values.tolist()
        self.VerigyUncheckedNan(self.fallerOrg, cols)
        self.replaceNanWithMean(self.fallerOrg, cols)

    def preprocessNonFaller(self):
        self.nonfallerOrg.drop('Date', axis=1, inplace=True)
        self.nonfallerOrg.drop('yr almost', axis=1, inplace=True)
        self.nonfallerOrg['#'].values[:] = 0  # 0 means nonfaller
        self.nonfallerOrg['#'] = self.nonfallerOrg['#'].astype(int)  # change type object to int
        cols = self.nonfallerOrg.columns.values.tolist()
        self.VerigyUncheckedNan(self.nonfallerOrg, cols)
        self.replaceNanWithMean(self.nonfallerOrg, cols)

    def replaceNanWithMean(self, dataFrame, column):
        for col in column:
            mean = 0
            data = dataFrame[col]
            dscrb = data.describe()
            if not dscrb.__contains__('mean'):  # this column is object type
                tmp = data.astype(float).describe()
                mean = tmp['mean']
                dataFrame[col] = data.astype(float, inplace=True)  # Generate None file
            else:
                mean = dscrb['mean']
            data.fillna(mean, inplace=True)

    '''
    check for strs containing space which cannot be found be dropna()
    '''

    def VerigyUncheckedNan(self, dataFrame, columns):
        for column in columns:
            i = 0
            tmp = dataFrame[column]
            for v in tmp:
                if v is None or type(v) is str:
                    tmp[i] = None
                i = i + 1
        return

    def countMissingData(self, dataFrame):
        countMatrix = dataFrame.isnull()  # find "N/A" data

        for col in fallOrg.columns:
            i = 0
            while (i < len(fallOrg.index)):
                print(i)
                print(col)
                tmp = fallOrg[col].values[i]
                print(tmp)
                if type(tmp) is np.float64 or type(tmp) is np.int64 or type(tmp) is int or type(tmp) is float:
                    i = i + 1
                    continue
                else:
                    # print('True')
                    countMatrix[col].values[i] = True
                    i = i + 1
        return countMatrix.sum().sort_values()

    def imputeMissingDataWithMean(self, dataFrame, name):
        zeroIndex = []
        tmp = dataFrame[name]
        for v in tmp:
            if v is 0:
                tmp[i] = 0
            i = i + 1
        return

    def PCAFeatureExtraction(self, componentsNum):
        print('--------------PCAFeatureExtraction-----------')
        X = self.normalizedDataset
        print('--------------before PCADataset----------- %s', X.shape)
        pca = PCA(n_components=componentsNum)
        self.PCADataset = pca.fit_transform(X)
        print('--------------PCADataset----------- %s', self.PCADataset.shape)
        print('explained_variance_:  %s', pca.explained_variance_)
        print('explained_variance_ratio_:   %s', pca.explained_variance_ratio_)
        print('sum:  ', np.array(pca.explained_variance_ratio_).sum())


def init():
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)




