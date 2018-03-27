import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import DeepPredict.arguments as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import NAME_Y, NAME_X, NAME_HYPO, NAME_PREDICT


class MyTest:
    def __init__(self, vector_list):
        self.vector_list = vector_list

    def test(self, tensor_load=""):
        def __show_score__(_key):
            p_score = float()
            r_score = float()
            f_score = float()
            a_score = float()
            auc_score = float()

            for p in precision[_key]:
                p_score += p
            for r in recall[_key]:
                r_score += r
            for f in f1[_key]:
                f_score += f
            for a in accuracy[_key]:
                a_score += a
            for u in roc_auc[_key]:
                auc_score += u

            print("\n\n============" + _key + "============\n")
            print("Total precision - %.2f" % ((p_score / op.NUM_FOLDS) * 100))
            print("Total recall -  %.2f" % ((r_score / op.NUM_FOLDS) * 100))
            print("Total F1-Score -  %.2f" % ((f_score / op.NUM_FOLDS) * 100))
            print("Total accuracy -  %.2f" % ((a_score / op.NUM_FOLDS) * 100))
            print("Total auc -  %.2f" % ((auc_score / op.NUM_FOLDS) * 100))
            print("\n\n======================================\n")

        def __append_score__(_score_list, _score):
            _score_list.append(_score)
            
        def __load__():
            if op.NUM_HIDDEN_LAYER < 10:
                _hidden_ = "_h_0" + str(op.NUM_HIDDEN_LAYER)
            else:
                _hidden_ = "_h_" + str(op.NUM_HIDDEN_LAYER)

            _epoch_ = "_ep_" + str(op.EPOCH) + "_"
            _tensor_load = tensor_load + _hidden_ + _epoch_ + str(k_fold + 1) + "/"

            sess = tf.Session()
            saver = tf.train.import_meta_graph(_tensor_load + 'model-' + str(op.EPOCH) + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(_tensor_load))
            graph = tf.get_default_graph()
            tf_x = graph.get_tensor_by_name(NAME_X + ":0")
            tf_y = graph.get_tensor_by_name(NAME_Y + ":0")
            hypothesis = graph.get_tensor_by_name(NAME_HYPO + ":0")
            predict = graph.get_tensor_by_name(NAME_PREDICT + ":0")

            h, p = sess.run([hypothesis, predict], feed_dict={tf_x: x_test, tf_y: y_test})

            _precision = precision_score(y_test, p)
            _recall = recall_score(y_test, p)
            _f1 = f1_score(y_test, p)
            _accuracy = accuracy_score(y_test, p)

            print('Precision : %.2f' % (_precision * 100))
            print('Recall : %.2f' % (_recall * 100))
            print('F1-Score : %.2f' % (_f1 * 100))
            print('Accuracy : %.2f' % (_accuracy * 100))

            __append_score__(accuracy["logistic_regression"], _accuracy)
            __append_score__(precision["logistic_regression"], _precision)
            __append_score__(recall["logistic_regression"], _recall)
            __append_score__(f1["logistic_regression"], _f1)
            logistic_fpr, logistic_tpr, _ = roc_curve(y_test, h)
            __append_score__(roc_auc["logistic_regression"], auc(logistic_fpr, logistic_tpr))

        accuracy = {"logistic_regression": list(), "svm": list()}
        precision = {"logistic_regression": list(), "svm": list()}
        recall = {"logistic_regression": list(), "svm": list()}
        roc_auc = {"logistic_regression": list(), "svm": list()}
        f1 = {"logistic_regression": list(), "svm": list()}

        for k_fold in range(op.NUM_FOLDS):
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_test = self.vector_list[k_fold]["y_test"]

            __load__()

        __show_score__("logistic_regression", precision, recall, f1, accuracy, roc_auc)
