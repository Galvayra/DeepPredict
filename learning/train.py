import numpy as np
import matplotlib.pyplot as plt
import DeepPredict.arguments as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
from .neuralNet import MyNeuralNetwork
import time


class MyTrain(MyNeuralNetwork):
    def __init__(self, vector_list):
        super().__init__()
        self.vector_list = vector_list

    def training(self):
        def __show_shape():
            def __count_mortality(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [1]:
                        _count += 1

                return _count

            x_train_np = np.array([np.array(j) for j in x_train])
            x_test_np = np.array([np.array(j) for j in x_test])
            y_train_np = np.array([np.array(j) for j in y_train])
            y_test_np = np.array([np.array(j) for j in y_test])

            print("\n\n\n\n=====================================\n")
            print("dims - ", len(x_train[0]))
            print("learning count -", len(y_train), "\t mortality count -", __count_mortality(y_train))
            print("test     count -", len(y_test), "\t mortality count -", __count_mortality(y_test), "\n")

            print(np.shape(x_train_np), np.shape(y_train_np))
            print(np.shape(x_test_np), np.shape(y_test_np))

        def __train_svm():
            def __add_score(**kwargs):
                for _k, _v in kwargs.items():
                    self.score[_k] += _v

            model = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            probas_ = model.predict_proba(x_test)

            _precision = precision_score(y_test, y_pred)
            _recall = recall_score(y_test, y_pred)
            _accuracy = accuracy_score(y_test, y_pred)
            _f1 = f1_score(y_test, y_pred)
            _svm_fpr, _svm_tpr, _ = roc_curve(y_test, probas_[:, 1])
            _svm_fpr *= 100
            _svm_tpr *= 100

            _auc = auc(_svm_fpr, _svm_tpr) / 100
            __add_score(**{"P": _precision, "R": _recall, "F1": _f1, "Acc": _accuracy, "AUC": _auc})

            if op.DO_SHOW:
                print('\n\n')
                print(k_fold + 1, "fold", 'SVM')
                print('Precision : %.1f' % (_precision * 100))
                print('Recall    : %.1f' % (_recall * 100))
                print('F1-Score  : %.1f' % (_f1 * 100))
                print('Accuracy  : %.1f' % (_accuracy * 100))
                print('AUC       : %.1f' % _auc)
                plot.plot(_svm_fpr, _svm_tpr, alpha=0.3, label='ROC %d (AUC = %0.1f)' % (k_fold+1, _auc))

        def __init_plt(_title):
            _fig = plt.figure(figsize=(10, 6))
            _fig.suptitle("ROC CURVE", fontsize=16)
            _plot = plt.subplot2grid((2, 2), (0, 0))
            _plot.set_title(_title)

            _plot.set_ylabel("Sensitivity")
            _plot.set_xlabel("100 - Specificity")

            return _plot

        def __show_plt():
            plot.legend(loc="lower right")
            plt.show()

        def __show_score(_method):
            print("\n\n============ " + _method + " ============\n")
            print("Total precision - %.1f" % ((self.score["P"] / op.NUM_FOLDS) * 100))
            print("Total recall    - %.1f" % ((self.score["R"] / op.NUM_FOLDS) * 100))
            print("Total F1-Score  - %.1f" % ((self.score["F1"] / op.NUM_FOLDS) * 100))
            print("Total accuracy  - %.1f" % ((self.score["Acc"] / op.NUM_FOLDS) * 100))
            print("Total auc       - %.1f" % (self.score["AUC"] / op.NUM_FOLDS))
            print("\n\n======================================\n")

        start_time = time.time()
        plot = None

        if op.DO_SHOW:
            if op.DO_SVM:
                plot = __init_plt("Support Vector Machine")
            else:
                plot = __init_plt("Feed Forward Neural Network")

        for k_fold in range(op.NUM_FOLDS):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            if op.DO_SHOW:
                __show_shape()

            if op.DO_SVM:
                __train_svm()
            else:
                self.feed_forward_nn(k_fold, x_train, y_train, x_test, y_test, plot)

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time))
        print("\n\n")

        if op.DO_SVM:
            __show_score("Support Vector Machine")
        else:
            __show_score("Feed Forward NN")

        if op.DO_SHOW:
            __show_plt()

    def vector2txt(self, _file_name):
        def __vector2txt():
            def __write_vector(_w_file):
                for dimension, v in enumerate(x):
                    if v != 0:
                        _w_file.write(str(dimension + 1) + ":" + str(v) + token)
                _w_file.write("\n")

            with open("make/" + train_file_name + "_" + str(k_fold + 1) + ".txt", 'w') as train_file:
                for x, y in zip(x_train, y_train):
                    if y[0] == 1:
                        train_file.write(str(1) + token)
                    else:
                        train_file.write(str(-1) + token)
                    __write_vector(train_file)

            with open("make/" + test_file_name + "_" + str(k_fold + 1) + ".txt", 'w') as test_file:
                for x, y in zip(x_test, y_test):
                    if y[0] == 1:
                        test_file.write(str(1) + token)
                    else:
                        test_file.write(str(-1) + token)
                    __write_vector(test_file)

        token = " "
        train_file_name = "train_" + _file_name
        test_file_name = "test_" + _file_name

        for k_fold in range(op.NUM_FOLDS):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            __vector2txt()


class MyPredict(MyNeuralNetwork):
    def __init__(self, vector_list):
        super().__init__()
        self.vector_list = vector_list

    def __show_score(self, _method):
        print("\n\n============ " + _method + " ============\n")
        print("Total precision - %.1f" % ((self.score["P"] / op.NUM_FOLDS) * 100))
        print("Total recall    - %.1f" % ((self.score["R"] / op.NUM_FOLDS) * 100))
        print("Total F1-Score  - %.1f" % ((self.score["F1"] / op.NUM_FOLDS) * 100))
        print("Total accuracy  - %.1f" % ((self.score["Acc"] / op.NUM_FOLDS) * 100))
        print("Total auc       - %.1f" % (self.score["AUC"] / op.NUM_FOLDS))
        print("\n\n======================================\n")

    def predict(self):
        def __show_plt():
            plot.legend(loc="lower right")
            plt.show()

        plot = None

        if op.DO_SHOW:
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            plot = plt.subplot2grid((2, 2), (0, 0))
            plot.set_title("Feed Forward Neural Network")

            plot.set_ylabel("Sensitivity")
            plot.set_xlabel("100 - Specificity")

        for k_fold in range(op.NUM_FOLDS):
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_test = self.vector_list[k_fold]["y_test"]

            self.load_feed_forward_nn(k_fold, x_test, y_test, plot)

        self.__show_score("Feed Forward NN")

        if op.DO_SHOW:
            __show_plt()
