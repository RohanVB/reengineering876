from os import listdir
import numpy as np
import pandas as pd
from sklearn import preprocessing as prepro
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters,EfficientFCParameters
from sklearn import model_selection
from sklearn.decomposition import PCA


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



class GaitDataset:
    # load data from dataset
    filePath = "./dataset_bak/"
    # dataset = np.fromfile(fileName, dtype='int16', count=-1, sep='')


    dataFrame = None
    fallerParams = []
    nonfallerParams = []
    id = []
    #after scale
    v_acc = []
    ml_acc = []
    ap_acc = []
    yaw_v = []
    pitch_v = []
    roll_v = []
    time = []
    Y = []
    X = []
    X_filtered = []
    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []


    # before scale
    faller_id = []
    faller_v_acc = []
    faller_ml_acc = []
    faller_ap_acc = []
    faller_yaw_v = []
    faller_pitch_v = []
    faller_roll_v = []

    nonfaller_id = []
    nonfaller_v_acc = []
    nonfaller_ml_acc = []
    nonfaller_ap_acc = []
    nonfaller_yaw_v = []
    nonfaller_pitch_v = []
    nonfaller_roll_v = []

    def initVars(self):
        files = listdir(self.filePath)
        files.sort()
        # del files[0]  # .DS_Store file useless
        print('len(files)=', len(files))

        for index, file in enumerate(files):

            # file_id =  file.split('_')
            # file_id = int(file_id[len(file_id)-2:])
            fname = self.filePath + file
            if file.endswith('dat'):  # dat file
                dataset = np.fromfile(fname, dtype='int16', count=-1, sep='')
                dataset = np.reshape(dataset, (6, -1))

                # print('dataset.shape------ %s', dataset.shape)

                if file.startswith('c'):  # nonfaller
                    self.nonfaller_v_acc.append(dataset[0])
                    self.nonfaller_ml_acc.append(dataset[1])
                    self.nonfaller_ap_acc.append(dataset[2])
                    self.nonfaller_yaw_v.append(dataset[3])
                    self.nonfaller_pitch_v.append(dataset[4])
                    self.nonfaller_roll_v.append(dataset[5])
                else:
                    self.faller_v_acc.append(dataset[0])
                    self.faller_ml_acc.append(dataset[1])
                    self.faller_ap_acc.append(dataset[2])
                    self.faller_yaw_v.append(dataset[3])
                    self.faller_pitch_v.append(dataset[4])
                    self.faller_roll_v.append(dataset[5])

            else: #head
                with open(fname) as f:
                    content = f.readlines()
                content = [x.rstrip('\n') for x in content]
                idx = content[0].split('_')[0]
                idx = int(idx[len(idx) - 2:])

                length = content[0].split(' ')[3] #samples in each signal

                # v_acc data
                v_acc_gain_and_baseline = content[1].split(' ')[2] #47916.5958(-53949)/g
                v_acc_gain = float(v_acc_gain_and_baseline.split('(')[0]) #47916.5958
                end = v_acc_gain_and_baseline.split('(')[1].find(')')
                v_acc_base = int(v_acc_gain_and_baseline.split('(')[1][:end]) #-53949

                # ml_acc data
                ml_acc_gain_and_baseline = content[2].split(' ')[2]  # 47916.5958(-53949)/g
                ml_acc_gain = float(ml_acc_gain_and_baseline.split('(')[0])  # 47916.5958
                mlend = ml_acc_gain_and_baseline.split('(')[1].find(')')
                ml_acc_base = int(ml_acc_gain_and_baseline.split('(')[1][:mlend])  # -53949

                # app_acc data
                app_acc_gain_and_baseline = content[3].split(' ')[2]  # 47916.5958(-53949)/g
                app_acc_gain = float(app_acc_gain_and_baseline.split('(')[0])  # 47916.5958
                append = app_acc_gain_and_baseline.split('(')[1].find(')')
                app_acc_base = int(app_acc_gain_and_baseline.split('(')[1][:append])  # -53949

                # yaw_v data
                yaw_v_gain_and_baseline = content[4].split(' ')[2]  # 47916.5958(-53949)/g
                yaw_v_gain = float(yaw_v_gain_and_baseline.split('(')[0])  # 47916.5958
                yaw_vend = yaw_v_gain_and_baseline.split('(')[1].find(')')
                yaw_v_base = int(yaw_v_gain_and_baseline.split('(')[1][:yaw_vend])  # -53949

                # pitch_v data
                pitch_v_gain_and_baseline = content[5].split(' ')[2]  # 47916.5958(-53949)/g
                pitch_v_gain = float(pitch_v_gain_and_baseline.split('(')[0])  # 47916.5958
                pitch_vend = pitch_v_gain_and_baseline.split('(')[1].find(')')
                pitch_v_base = int(pitch_v_gain_and_baseline.split('(')[1][:pitch_vend])  # -53949

                # roll_v data
                roll_v_gain_and_baseline = content[6].split(' ')[2]  # 47916.5958(-53949)/g
                roll_v_gain = float(roll_v_gain_and_baseline.split('(')[0])  # 47916.5958
                roll_vend = roll_v_gain_and_baseline.split('(')[1].find(')')
                roll_v_base = int(roll_v_gain_and_baseline.split('(')[1][:roll_vend])  # -53949
                newitem = {'idx': idx, 'len':length, 'v_acc':[v_acc_gain, v_acc_base], 'ml_acc':[ml_acc_gain, ml_acc_base],
                                            'app_acc': [app_acc_gain, app_acc_base], 'yaw_v':[yaw_v_gain, yaw_v_base],
                                            'pitch_v':[pitch_v_gain,pitch_v_base], 'roll_v': [roll_v_gain,roll_v_base]}
                if file.startswith('c'):  # nonfaller
                    self.nonfallerParams.append(newitem)
                else:
                    self.fallerParams.append(newitem)
        print('len(self.fallerParams)',len(self.fallerParams))
        print('len(self.faller_ap_acc)', len(self.faller_ap_acc))
        print('len(self.nonfallerParams)', len(self.nonfallerParams))
        print('len(self.nonfaller_ap_acc)', len(self.nonfaller_ap_acc))

    # scale
    def rearrageVars(self):
        for index, head in enumerate(self.fallerParams):
            self.faller_v_acc[index] = self.faller_v_acc[index] - head['v_acc'][1] #subtract baseline
            self.faller_ml_acc[index]= self.faller_ml_acc[index] - head['ml_acc'][1]
            self.faller_ap_acc[index] = self.faller_ap_acc[index] - head['app_acc'][1]
            self.faller_yaw_v[index] = self.faller_yaw_v[index] - head['yaw_v'][1]
            self.faller_pitch_v[index] = self.faller_pitch_v[index] - head['pitch_v'][1]
            self.faller_roll_v[index]= self.faller_roll_v[index]- head['roll_v'][1]

            self.faller_v_acc[index] = self.faller_v_acc[index] / head['v_acc'][0]  # divide gain
            self.faller_ml_acc[index] = self.faller_ml_acc[index] / head['ml_acc'][0]
            self.faller_ap_acc[index] = self.faller_ap_acc[index] / head['app_acc'][0]
            self.faller_yaw_v[index] = self.faller_yaw_v[index] / head['yaw_v'][0]
            self.faller_pitch_v[index] = self.faller_pitch_v[index] / head['pitch_v'][0]
            self.faller_roll_v[index] = self.faller_roll_v[index] / head['roll_v'][0]

            # generate complete data
            length = int(head['len'])
            self.id  = self.id + np.array([index+1]*length).tolist()
            self.Y.append(1)
            self.time = self.time + list(range(0,length))
            self.v_acc = self.v_acc + self.faller_v_acc[index].tolist()
            self.ml_acc = self.ml_acc + self.faller_ml_acc[index].tolist()
            self.ap_acc = self.ap_acc + self.faller_ap_acc[index].tolist()
            self.yaw_v = self.yaw_v + self.faller_yaw_v[index].tolist()
            self.pitch_v = self.pitch_v + self.faller_pitch_v[index].tolist()
            self.roll_v = self.roll_v + self.faller_roll_v[index].tolist()


        for index, head in enumerate(self.nonfallerParams):
            # print('index=', index)
            # print('self.nonfaller_v_acc[index]=', self.nonfaller_v_acc[index])
            # print('head[\'v_acc\'][1]=', head['v_acc'][1])
            self.nonfaller_v_acc[index] = self.nonfaller_v_acc[index] - head['v_acc'][1]  # subtract baseline
            self.nonfaller_ml_acc[index] = self.nonfaller_ml_acc[index] - head['ml_acc'][1]
            self.nonfaller_ap_acc[index] = self.nonfaller_ap_acc[index] - head['app_acc'][1]
            self.nonfaller_yaw_v[index] = self.nonfaller_yaw_v[index] - head['yaw_v'][1]
            self.nonfaller_pitch_v[index] = self.nonfaller_pitch_v[index] - head['pitch_v'][1]
            self.nonfaller_roll_v[index] = self.nonfaller_roll_v[index] - head['roll_v'][1]

            self.nonfaller_v_acc[index] = self.nonfaller_v_acc[index] / head['v_acc'][0]  # divide gain
            self.nonfaller_ml_acc[index] = self.nonfaller_ml_acc[index] / head['ml_acc'][0]
            self.nonfaller_ap_acc[index] = self.nonfaller_ap_acc[index] / head['app_acc'][0]
            self.nonfaller_yaw_v[index] = self.nonfaller_yaw_v[index] / head['yaw_v'][0]
            self.nonfaller_pitch_v[index] = self.nonfaller_pitch_v[index] / head['pitch_v'][0]
            self.nonfaller_roll_v[index] = self.nonfaller_roll_v[index] / head['roll_v'][0]

            # generate complete data
            newI = index + len(self.fallerParams) + 1
            length = int(head['len'])
            self.id = self.id + np.array([newI] * length).tolist()
            self.Y.append(0)
            self.time = self.time + list(range(0, length))

            self.v_acc = self.v_acc + self.nonfaller_v_acc[index].tolist()
            self.ml_acc = self.ml_acc + self.nonfaller_ml_acc[index].tolist()
            self.ap_acc = self.ap_acc + self.nonfaller_ap_acc[index].tolist()
            self.yaw_v = self.yaw_v + self.nonfaller_yaw_v[index].tolist()
            self.pitch_v = self.pitch_v + self.nonfaller_pitch_v[index].tolist()
            self.roll_v = self.roll_v + self.nonfaller_roll_v[index].tolist()

        return

    def generateDataFrame(self):
        dic = {'id':self.id, 'time':self.time, 'v_acc': self.v_acc, 'ml_acc':self.ml_acc, 'ap_acc': self.ap_acc, 'yaw_v':self.yaw_v, 'pitch_v':self.pitch_v, 'roll_v':self.roll_v}
        self.dataFrame  = pd.DataFrame(dic)
        return

    def normalize(self):
        self.X = prepro.scale(self.X, axis=0, with_std=True, with_mean=True)
        self.X_filtered = prepro.scale(self.X, axis=0, with_std=True, with_mean=True)

    def train(self):
        validation_size = 0.20
        seed = 7
        print('Y----',self.Y)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(self.X, self.Y, test_size=validation_size,
                                                                                        random_state=seed)

        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))

        results = []
        filtered_results = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            model.fit(self.X_train, self.Y_train)

            accuracy_rst = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring='accuracy')
            results.append(
                {'name': name, 'accuracy': accuracy_rst})  # , 'precision': precision_rst, 'recall' :recall_rst})
        for result in results:
            print("%s: %f (%f)" % (result['name'], result['accuracy'].mean(), result['accuracy'].std()))


