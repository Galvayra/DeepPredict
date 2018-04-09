import tensorflow as tf
import matplotlib.pyplot as plt
import DeepPredict.arguments as op
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import NAME_Y, NAME_X, NAME_HYPO, NAME_PREDICT, TENSOR_PATH


class MyTest:
    def __init__(self, vector_list):
        self.vector_list = vector_list

    def predict(self):
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

        def __show_plt__():
            logistic_plot.legend(loc="lower right")
            plt.show()

        def __load__():
            if op.NUM_HIDDEN_LAYER < 10:
                _hidden_ = "h_0" + str(op.NUM_HIDDEN_LAYER)
            else:
                _hidden_ = "h_" + str(op.NUM_HIDDEN_LAYER)

            _epoch_ = "_ep_" + str(op.EPOCH) + "_"

            if op.USE_ID:
                _tensor_load = TENSOR_PATH + op.USE_ID.split('#')[0] + "_"
            else:
                _tensor_load = TENSOR_PATH

            _tensor_load += _hidden_ + _epoch_ + str(k_fold + 1) + "/"

            sess = tf.Session()
            saver = tf.train.import_meta_graph(_tensor_load + 'model-' + str(op.EPOCH) + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(_tensor_load))

            print("\n\n\nRead Neural Network -", _tensor_load, "\n")
            graph = tf.get_default_graph()
            tf_x = graph.get_tensor_by_name(NAME_X + ":0")
            tf_y = graph.get_tensor_by_name(NAME_Y + ":0")
            hypothesis = graph.get_tensor_by_name(NAME_HYPO + ":0")
            predict = graph.get_tensor_by_name(NAME_PREDICT + ":0")

            h, p = sess.run([hypothesis, predict], feed_dict={tf_x: x_test, tf_y: y_test})

            logistic_fpr, logistic_tpr, _ = roc_curve(y_test, h)

            _precision = precision_score(y_test, p)
            _recall = recall_score(y_test, p)
            _f1 = f1_score(y_test, p)
            _accuracy = accuracy_score(y_test, p)
            _auc = auc(logistic_fpr, logistic_tpr)

            if op.DO_SHOW:
                print('Precision : %.2f' % (_precision * 100))
                print('Recall    : %.2f' % (_recall * 100))
                print('F1-Score  : %.2f' % (_f1 * 100))
                print('Accuracy  : %.2f' % (_accuracy * 100))
                print('AUC       : %.2f' % (_auc * 100))
                logistic_plot.plot(logistic_fpr, logistic_tpr, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (k_fold+1,
                                                                                                               _auc))

            __append_score__(accuracy["logistic_regression"], _accuracy)
            __append_score__(precision["logistic_regression"], _precision)
            __append_score__(recall["logistic_regression"], _recall)
            __append_score__(f1["logistic_regression"], _f1)
            __append_score__(roc_auc["logistic_regression"], _auc)

        accuracy = {"logistic_regression": list(), "svm": list()}
        precision = {"logistic_regression": list(), "svm": list()}
        recall = {"logistic_regression": list(), "svm": list()}
        roc_auc = {"logistic_regression": list(), "svm": list()}
        f1 = {"logistic_regression": list(), "svm": list()}

        if op.DO_SHOW:
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            logistic_plot = plt.subplot2grid((2, 2), (0, 0))
            logistic_plot.set_title("Feed Forward Neural Network")

            logistic_plot.set_ylabel("TPR (sensitivity)")
            logistic_plot.set_xlabel("1 - specificity")

        for k_fold in range(op.NUM_FOLDS):
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_test = self.vector_list[k_fold]["y_test"]

            __load__()

        __show_score__("logistic_regression")

        if op.DO_SHOW:
            __show_plt__()
