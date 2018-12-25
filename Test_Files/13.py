from ClinicalDataSet import ClinicalDataSet
from GaitDataset import GaitDataset

import numpy as np
import pandas as pd
from sklearn import preprocessing as prepro
from sklearn.decomposition import PCA
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
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

if __name__ == "__main__":
    dataController = ClinicalDataSet()
    dataController.readDateSet()  # get data

    # preprocess nonfaller
    dataController.preprocessFaller()
    dataController.preprocessNonFaller()
    # dataController.nonfallerOrg.drop(axis=0,index=[10, 36, 37,38,42,43])
    # dataController.fallerOrg.drop(axis=0, index=[10, 25])
    dataController.connetDataset()

    print('dataController.faller.shape-------%s', dataController.fallerOrg.shape)

    print('dataController.nonfallerOrg.shape-------%s', dataController.nonfallerOrg.shape)

    # normalize
    Xdata = dataController.completeDataset.values[:, 1:]  # get Xdata
    # dataController.normalizedDataset = prepro.scale(Xdata, axis=0, with_std=True, with_mean=True)
    dataController.normalizedDataset = prepro.normalize(Xdata, norm='l1', axis=0)
    dataController.PCAFeatureExtraction(5)
    print('dataController.PCADataset.shape-------%s', dataController.PCADataset.shape)

    dm = GaitDataset()
    dm.initVars()

    # dm.fallerParams.del
    print('dataManager.nonfallerParams.shape-------%s', len(dm.nonfallerParams))
    print('dataManager.fallerParams.shape-------%s', len(dm.fallerParams))

    dm.rearrageVars()
    dm.generateDataFrame()
    print('dataframe shape--------', dm.dataFrame.shape)
    extract_settings = MinimalFCParameters()
    dm.X = extract_features(dm.dataFrame, column_id='id', column_sort='time', default_fc_parameters=extract_settings)
    print('X.shape------', dm.X.shape)
    y = np.array([1, 1, 0, 0])
    dm.X = dm.X[['ap_acc__maximum', 'ap_acc__minimum',
                 'ml_acc__maximum', 'ml_acc__minimum',
                 'pitch_v__maximum', 'pitch_v__minimum',
                 'roll_v__maximum', 'roll_v__minimum',
                 'v_acc__maximum', 'v_acc__minimum',
                 'yaw_v__maximum', 'yaw_v__minimum']]
    # dm.X = dm.X[['ap_acc__standard_deviation', 'ap_acc__variance',
    #              'ml_acc__standard_deviation', 'ml_acc__variance',
    #              'pitch_v__standard_deviation', 'pitch_v__variance',
    #              'roll_v__standard_deviation', 'roll_v__variance',
    #              'v_acc__standard_deviation', 'v_acc__variance',
    #              'yaw_v__standard_deviation', 'yaw_v__variance']]

    # dm.X = prepro.scale(dm.X, axis=0, with_std=True, with_mean=True)
    dm.X = prepro.normalize(dm.X, norm='l1', axis=0)
    # print('dm.X.head()', dm.X[1:10])
    # training modle
    X = pd.concat([pd.DataFrame(dataController.PCADataset), pd.DataFrame(dm.X)], axis=1)
    # X = prepro.scale(X, axis=0, with_std=True, with_mean=True)
    Y = dataController.completeDataset.values[:, 0]
    print('X.shape------', X.shape)
    print('Y.shape------', Y.shape)

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    print('--------------X_train %s-----------', X_train.shape)
    print('------------- X_validation %s-----------', X_validation.shape)
    print('--------------Y_train %s-----------', X_train.shape)
    print('------------- Y_validation %s-----------', X_validation.shape)

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    results = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        model.fit(X_train, Y_train)

        # precision_rst = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='precision')
        # recall_rst = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='recall')
        accuracy_rst = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append({'name': name, 'accuracy': accuracy_rst})  # , 'precision': precision_rst, 'recall' :recall_rst})
    # print('---results---- %s'. results[0])
    for result in results:
        print("%s: %f (%f)" % (result['name'], result['accuracy'].mean(), result['accuracy'].std()))
        # print("%s: %f (%f)" % (result['name'], result['precision'].mean(), result['precision'].std()))
        # print("%s: %f (%f)" % (result['name'], result['recall'].mean(), result['recall'].std()))

    nb = KNeighborsClassifier()
    nb.fit(X_train, Y_train)
    predictions = nb.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    y_score = nb.predict(X_validation)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], thresholds = roc_curve(Y_validation, y_score)
    roc_auc[0] = auc(fpr[0], tpr[0])

    print('fpr:', fpr)
    print('tpr:', tpr)
    # fpr["micro"], tpr["micro"], _ = roc_curve(Y_validation.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2

    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic figure')
    plt.legend(loc="lower right")
    plt.show()
