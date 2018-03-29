import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import DeepPredict.arguments as op
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
import time
import os
import shutil


class MyTrain:
    def __init__(self, vector_list):
        self.vector_list = vector_list

    def training(self, tensor_save=""):
        def __show_shape__():
            def __count_mortality__(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [1]:
                        _count += 1

                return _count

            print("\n\n\n\n=====================================\n")
            print("dims - ", len(x_train[0]))
            print("learning count -", len(y_train), "\t mortality count -", __count_mortality__(y_train))
            print("test     count -", len(y_test), "\t mortality count -", __count_mortality__(y_test), "\n")

            print(np.shape(x_train_np), np.shape(y_train_np))
            print(np.shape(x_test_np), np.shape(y_test_np))

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
            print("Total recall    -  %.2f" % ((r_score / op.NUM_FOLDS) * 100))
            print("Total F1-Score  -  %.2f" % ((f_score / op.NUM_FOLDS) * 100))
            print("Total accuracy  -  %.2f" % ((a_score / op.NUM_FOLDS) * 100))
            print("Total auc       -  %.2f" % ((auc_score / op.NUM_FOLDS) * 100))
            print("\n\n======================================\n")
            
        def __append_score__(_score_list, _score):
            _score_list.append(_score)

        def __train_svm__():
            model = SVC(kernel=SVM_KERNEL, C=1.0, random_state=None, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            probas_ = model.predict_proba(x_test)

            _precision = precision_score(y_test, y_pred)
            _recall = recall_score(y_test, y_pred)
            _accuracy = accuracy_score(y_test, y_pred)
            _f1 = f1_score(y_test, y_pred)
            _svm_fpr, _svm_tpr, _ = roc_curve(y_test, probas_[:, 1])
            _auc = auc(_svm_fpr, _svm_tpr)

            __append_score__(accuracy["svm"], _accuracy)
            __append_score__(precision["svm"], _precision)
            __append_score__(recall["svm"], _recall)
            __append_score__(f1["svm"], _f1)
            __append_score__(roc_auc["svm"], _auc)

            if op.DO_SHOW:
                print('\n\n')
                print(k_fold + 1, "fold", 'SVM')
                print('Precision : %.2f' % (_precision * 100))
                print('Recall    : %.2f' % (_recall * 100))
                print('F1-Score  : %.2f' % (_f1 * 100))
                print('Accuracy  : %.2f' % (_accuracy * 100))
                print('AUC       : %.2f' % (_auc * 100))
                plot.plot(_svm_fpr, _svm_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % _auc)

        def __logistic_regression__():
            def __init_log_file_name__(_k_fold):
                log_name = "./logs/" + op.USE_ID + "log_"

                if op.NUM_HIDDEN_LAYER < 10:
                    log_name += "h_0" + str(op.NUM_HIDDEN_LAYER)
                else:
                    log_name += "h_" + str(op.NUM_HIDDEN_LAYER)

                log_name += "_ep_" + str(op.EPOCH) + "_k_" + str(_k_fold + 1)

                if op.USE_W2V:
                    log_name += "_w2v"

                return log_name

            def __init_save_dir__():
                if op.NUM_HIDDEN_LAYER < 10:
                    _hidden_ = "_h_0" + str(op.NUM_HIDDEN_LAYER)
                else:
                    _hidden_ = "_h_" + str(op.NUM_HIDDEN_LAYER)

                _epoch_ = "_ep_" + str(op.EPOCH) + "_"
                _save_dir = tensor_save + _hidden_ + _epoch_ + str(k_fold + 1) + "/"

                if os.path.isdir(_save_dir):
                    shutil.rmtree(_save_dir)
                os.mkdir(_save_dir)

                return _save_dir

            save_dir = __init_save_dir__()
            num_input_node = len(x_train[0])

            if NUM_HIDDEN_DIMENSION:
                num_hidden_node = NUM_HIDDEN_DIMENSION
            else:
                num_hidden_node = len(x_train[0])

            tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_input_node], name=NAME_X)
            tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)

            tf_weight = list()
            tf_bias = list()
            tf_layer = [tf_x]

            for i in range(op.NUM_HIDDEN_LAYER):
                num_hidden_node = int(num_input_node / RATIO_HIDDEN)

                tf_weight.append(tf.get_variable("h_weight_" + str(i + 1), dtype=tf.float32,
                                                 shape=[num_input_node, num_hidden_node],
                                                 initializer=tf.contrib.layers.xavier_initializer()))
                tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]), name="h_bias_" + str(i + 1)))
                tf_layer.append(tf.nn.relu(tf.matmul(tf_layer[i], tf_weight[i]) + tf_bias[i]))

                num_input_node = int(num_input_node / RATIO_HIDDEN)

            tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

            hypothesis = tf.sigmoid(tf.matmul(tf_layer[-1], tf_weight[-1]) + tf_bias[-1], name=NAME_HYPO)

            with tf.name_scope("cost"):
                cost = -tf.reduce_mean(tf_y * tf.log(hypothesis) + (1 - tf_y) * tf.log(1 - hypothesis))
                cost_summ = tf.summary.scalar("cost", cost)

            train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

            # cut off
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT)
            _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf_y), dtype=tf.float32))
            accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

            saver = tf.train.Saver()

            with tf.Session() as sess:
                merged_summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(__init_log_file_name__(k_fold))
                writer.add_graph(sess.graph)  # Show the graph

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # if self.is_closed:
                for step in range(op.EPOCH + 1):
                    summary, cost_val, _ = sess.run([merged_summary, cost, train],
                                                    feed_dict={tf_x: x_train, tf_y: y_train})
                    writer.add_summary(summary, global_step=step)

                    if op.DO_SHOW and step % (op.EPOCH / 10) == 0:
                        print(str(step).rjust(5), cost_val)

                h, p, a = sess.run([hypothesis, predict, _accuracy], feed_dict={tf_x: x_test, tf_y: y_test})

                saver.save(sess, save_dir + "model", global_step=op.EPOCH)

            tf.reset_default_graph()

            _precision = precision_score(y_test, p)
            _recall = recall_score(y_test, p)
            _f1 = f1_score(y_test, p)
            _logistic_fpr, _logistic_tpr, _ = roc_curve(y_test, h)
            _auc = auc(_logistic_fpr, _logistic_tpr)

            if _precision == 0 or _recall == 0:
                print('k-fold : %d, Precision : %.2f, Recall : %.2f' % (k_fold, (_precision * 100), (_recall * 100)))

            __append_score__(accuracy["logistic_regression"], a)
            __append_score__(precision["logistic_regression"], _precision)
            __append_score__(recall["logistic_regression"], _recall)
            __append_score__(f1["logistic_regression"], _f1)
            __append_score__(roc_auc["logistic_regression"], _auc)

            if op.DO_SHOW:
                print('\n\n')
                print(k_fold + 1, "fold", 'logistic regression')
                print('Precision : %.2f' % (_precision * 100))
                print('Recall    : %.2f' % (_recall * 100))
                print('F1-Score  : %.2f' % (_f1 * 100))
                print('Accuracy  : %.2f' % (a * 100))
                print('AUC       : %.2f' % (_auc * 100))
                plot.plot(_logistic_fpr, _logistic_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % _auc)

        def __back_propagation__():
            def __init_log_file_name__(_k_fold):
                log_name = "./logs/" + op.USE_ID + "log_"

                if op.NUM_HIDDEN_LAYER < 10:
                    log_name += "h_0" + str(op.NUM_HIDDEN_LAYER)
                else:
                    log_name += "h_" + str(op.NUM_HIDDEN_LAYER)

                log_name += "_ep_" + str(op.EPOCH) + "_k_" + str(_k_fold + 1)

                if op.USE_W2V:
                    log_name += "_w2v"

                return log_name

            def __init_save_dir__():
                if op.NUM_HIDDEN_LAYER < 10:
                    _hidden_ = "_h_0" + str(op.NUM_HIDDEN_LAYER)
                else:
                    _hidden_ = "_h_" + str(op.NUM_HIDDEN_LAYER)

                _epoch_ = "_ep_" + str(op.EPOCH) + "_"
                _save_dir = tensor_save + _hidden_ + _epoch_ + str(k_fold + 1) + "/"

                if os.path.isdir(_save_dir):
                    shutil.rmtree(_save_dir)
                os.mkdir(_save_dir)

                return _save_dir

            def __sigma__(x):
                return tf.div(tf.constant(1.0),
                              tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

            def __sigmaprime__(x):
                return tf.multiply(__sigma__(x), tf.subtract(tf.constant(1.0), __sigma__(x)))

            save_dir = __init_save_dir__()
            num_input_node = len(x_train[0])

            if NUM_HIDDEN_DIMENSION:
                num_hidden_node = NUM_HIDDEN_DIMENSION
            else:
                num_hidden_node = len(x_train[0])

            tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_input_node], name=NAME_X)
            tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=NAME_Y)

            tf_weight = list()
            tf_bias = list()
            z_list = list()
            a_list = [tf_x]

            z_delta_dict = dict()
            b_delta_dict = dict()
            w_delta_dict = dict()
            a_delta_dict = dict()

            for i in range(op.NUM_HIDDEN_LAYER):
                num_hidden_node = int(num_input_node / RATIO_HIDDEN)

                tf_weight.append(tf.Variable(tf.truncated_normal(shape=[num_input_node, num_hidden_node],
                                                                 name="h_weight_" + str(i + 1),
                                                                 dtype=tf.float32)))
                tf_bias.append(tf.Variable(tf.truncated_normal(shape=[1, num_hidden_node],
                                                               name="h_bias_" + str(i + 1))))

                z_list.append(tf.add(tf.matmul(a_list[i], tf_weight[i]), tf_bias[i]))
                a_list.append(__sigma__(z_list[i]))
                z_delta_dict[i] = list()
                b_delta_dict[i] = list()
                w_delta_dict[i] = list()
                a_delta_dict[i] = list()
                # tf_layer.append(tf.nn.relu(tf.matmul(tf_layer[i], tf_weight[i]) + tf_bias[i]))

                num_input_node = int(num_input_node / RATIO_HIDDEN)

            # tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
            #                                  initializer=tf.contrib.layers.xavier_initializer()))
            # tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

            # diff = tf.subtract(a_list[-1], tf_y)
            #
            # tt = tf.Variable([[1, 2], [3, 2]])
            # model = tf.initialize_all_variables()
            # with tf.Session() as session:
            #     # Array 위치가 축 (x,y,z), 거기에 입력하는 숫자가 바꾸고 싶은 차원
            #     x = tf.transpose(tt)
            #     # x = tf.transpose(x, perm=[0 ,1, 2])
            #     session.run(model)
            #     result = session.run(x)
            #     print(result)

            a_delta_dict[i] = tf.subtract(a_list[-1], tf_y)
            for i in reversed(range(op.NUM_HIDDEN_LAYER)):
                z_delta_dict[i] = tf.multiply(a_delta_dict[i], __sigmaprime__(z_list[i]))
                b_delta_dict[i] = z_delta_dict[i]
                w_delta_dict[i] = tf.matmul(tf.transpose(a_list[i]), z_delta_dict[i])

                if i != 0:
                    a_delta_dict[i-1] = tf.matmul(z_delta_dict[i], tf.transpose(w_delta_dict[i]))

            eta = tf.constant(0.5)

            step_list = list()
            for k in sorted(z_delta_dict.keys()):
                print(k, np.shape(tf_weight[k]))
                # step_list.append(tf.assign_add(tf_weight[k], tf.subtract(tf_weight[k], tf.multiply(eta, w_delta_dict[k]))))
                # step_list.append(tf.assign_add(tf_bias[k], tf.subtract(tf_weight[k], tf.multiply(eta, tf.reduce_mean(b_delta_dict[k], axis=[0])))))


            # step = [
            #     tf.assign(w_1,
            #               tf.subtract(w_1, tf.multiply(eta, d_w_1)))
            #     , tf.assign(b_1,
            #                 tf.subtract(b_1, tf.multiply(eta,
            #                                              tf.reduce_mean(d_b_1, axis=[0]))))
            #     , tf.assign(w_2,
            #                 tf.subtract(w_2, tf.multiply(eta, d_w_2)))
            #     , tf.assign(b_2,
            #                 tf.subtract(b_2, tf.multiply(eta,
            #                                              tf.reduce_mean(d_b_2, axis=[0]))))
            # ]


            # hypothesis = tf.sigmoid(tf.matmul(tf_layer[-1], tf_weight[-1]) + tf_bias[-1], name=NAME_HYPO)
            #
            # with tf.name_scope("cost"):
            #     cost = -tf.reduce_mean(tf_y * tf.log(hypothesis) + (1 - tf_y) * tf.log(1 - hypothesis))
            #     cost_summ = tf.summary.scalar("cost", cost)
            #
            # train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
            #
            # # cut off
            # predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT)
            # _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf_y), dtype=tf.float32))
            # accuracy_summ = tf.summary.scalar("accuracy", _accuracy)
            #
            # saver = tf.train.Saver()
            #
            # with tf.Session() as sess:
            #     merged_summary = tf.summary.merge_all()
            #     writer = tf.summary.FileWriter(__init_log_file_name__(k_fold))
            #     writer.add_graph(sess.graph)  # Show the graph
            #
            #     sess.run(tf.global_variables_initializer())
            #     sess.run(tf.local_variables_initializer())
            #
            #     # if self.is_closed:
            #     for step in range(op.EPOCH + 1):
            #         summary, cost_val, _ = sess.run([merged_summary, cost, train],
            #                                         feed_dict={tf_x: x_train, tf_y: y_train})
            #         writer.add_summary(summary, global_step=step)
            #
            #         if op.DO_SHOW and step % (op.EPOCH / 10) == 0:
            #             print(str(step).rjust(5), cost_val)
            #
            #     h, p, a = sess.run([hypothesis, predict, _accuracy], feed_dict={tf_x: x_test, tf_y: y_test})
            #
            #     saver.save(sess, save_dir + "model", global_step=op.EPOCH)
            #
            # tf.reset_default_graph()
            #
            # _precision = precision_score(y_test, p)
            # _recall = recall_score(y_test, p)
            # _f1 = f1_score(y_test, p)
            # _logistic_fpr, _logistic_tpr, _ = roc_curve(y_test, h)
            # _auc = auc(_logistic_fpr, _logistic_tpr)
            #
            # if _precision == 0 or _recall == 0:
            #     print('k-fold : %d, Precision : %.2f, Recall : %.2f' % (k_fold, (_precision * 100), (_recall * 100)))
            #
            # __append_score__(accuracy["back_propagation"], a)
            # __append_score__(precision["back_propagation"], _precision)
            # __append_score__(recall["back_propagation"], _recall)
            # __append_score__(f1["back_propagation"], _f1)
            # __append_score__(roc_auc["back_propagation"], _auc)
            #
            # if op.DO_SHOW:
            #     print('\n\n')
            #     print(k_fold + 1, "fold", 'back propagation')
            #     print('Precision : %.2f' % (_precision * 100))
            #     print('Recall    : %.2f' % (_recall * 100))
            #     print('F1-Score  : %.2f' % (_f1 * 100))
            #     print('Accuracy  : %.2f' % (a * 100))
            #     print('AUC       : %.2f' % (_auc * 100))
            #     plot.plot(_logistic_fpr, _logistic_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % _auc)

        def __show_plt__():
            plot.legend(loc="lower right")
            plt.show()

        def __init_plt__(_title):
            _fig = plt.figure(figsize=(10, 6))
            _fig.suptitle("ROC CURVE", fontsize=16)
            _plot = plt.subplot2grid((2, 2), (0, 0))
            _plot.set_title(_title)

            _plot.set_ylabel("TPR (sensitivity)")
            _plot.set_xlabel("1 - specificity")

            return _plot

        start_time = time.time()
        accuracy = {"logistic_regression": list(), "back_propagation": list(), "svm": list()}
        precision = {"logistic_regression": list(), "back_propagation": list(), "svm": list()}
        recall = {"logistic_regression": list(), "back_propagation": list(), "svm": list()}
        roc_auc = {"logistic_regression": list(), "back_propagation": list(), "svm": list()}
        f1 = {"logistic_regression": list(), "back_propagation": list(), "svm": list()}

        if op.DO_SHOW:
            if op.DO_SVM:
                plot = __init_plt__("SVM")
            else:
                plot = __init_plt__("Feed Forward Neural Network")

        for k_fold in range(op.NUM_FOLDS):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]
            x_train_np = np.array([np.array(j) for j in x_train])
            x_test_np = np.array([np.array(j) for j in x_test])
            y_train_np = np.array([np.array(j) for j in y_train])
            y_test_np = np.array([np.array(j) for j in y_test])

            if op.DO_SHOW:
                __show_shape__()

            if op.DO_SVM:
                __train_svm__()
            else:
                __back_propagation__()
                # __logistic_regression__()

        print("\n\n processing time     --- %s seconds ---" % (time.time() - start_time))
        print("\n\n")

        if op.DO_SVM:
            __show_score__("svm")
        else:
            __show_score__("logistic_regression")

        if op.DO_SHOW:
            __show_plt__()

    def vector2txt(self, _file_name):
        def __vector2txt__():
            def __write_vector__(_w_file):
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
                    __write_vector__(train_file)

            with open("make/" + test_file_name + "_" + str(k_fold + 1) + ".txt", 'w') as test_file:
                for x, y in zip(x_test, y_test):
                    if y[0] == 1:
                        test_file.write(str(1) + token)
                    else:
                        test_file.write(str(-1) + token)
                    __write_vector__(test_file)

        token = " "
        train_file_name = "train_" + _file_name
        test_file_name = "test_" + _file_name

        for k_fold in range(op.NUM_FOLDS):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            __vector2txt__()
