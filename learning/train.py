import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from DeepPredict.learning.options import *
import time


class MyTrain:
    def __init__(self, vector_list, is_closed):
        self.vector_list = vector_list
        self.num_fold = len(self.vector_list)
        self.epoch = EPOCH
        self.is_closed = is_closed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def training(self, do_show=True):
        def __show_shape__():
            def __count_mortality__(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [1]:
                        _count += 1

                return _count

            print("dims - ", len(x_train[0]))
            print("learning count -", len(y_train), "\t mortality count -", __count_mortality__(y_train))
            print("test     count -", len(y_test), "\t mortality count -", __count_mortality__(y_test), "\n")

            print(np.shape(x_train_np), np.shape(y_train_np))
            print(np.shape(x_test_np), np.shape(y_test_np))

        def __train_svm__():

            model = SVC(kernel='rbf', C=1.0, random_state=None, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            probas_ = model.predict_proba(x_test)

            _precision = precision_score(y_test, y_pred)
            _recall = recall_score(y_test, y_pred)
            _accuracy = accuracy_score(y_test, y_pred)

            print('\n\nSVM')
            print('Precision : %.2f' % (_precision*100))
            print('Recall : %.2f' % (_recall*100))
            print('Accuracy : %.2f' % (_accuracy*100))

            return probas_, _accuracy, _precision, _recall

        def __logistic_regression__():
            num_input_node = len(x_train[0])

            if NUM_HIDDEN_DIMENSION:
                num_hidden_node = NUM_HIDDEN_DIMENSION
            else:
                num_hidden_node = len(x_train[0])

            tf_x = tf.placeholder(dtype=tf.float32, shape=[None, num_input_node])
            tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

            tf_weight = list()
            tf_bias = list()
            tf_layer = [tf_x]

            print("\nepoch -", EPOCH)
            if NUM_HIDDEN_LAYER:
                print("num of  hidden layers  -", NUM_HIDDEN_LAYER)
                print("num of nodes in hidden -", num_hidden_node, "\n\n")

            for i in range(NUM_HIDDEN_LAYER):
                if i == 0:
                    tf_weight.append(tf.get_variable("h_weight_" + str(i+1), dtype=tf.float32,
                                                     shape=[num_input_node, num_hidden_node],
                                                     initializer=tf.contrib.layers.xavier_initializer()))
                else:
                    tf_weight.append(tf.get_variable("h_weight_" + str(i+1), dtype=tf.float32,
                                                     shape=[num_hidden_node, num_hidden_node],
                                                     initializer=tf.contrib.layers.xavier_initializer()))
                tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]), name="h_bias_" + str(i+1)))
                tf_layer.append(tf.nn.relu(tf.matmul(tf_layer[i], tf_weight[i]) + tf_bias[i]))

            tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

            hypothesis = tf.sigmoid(tf.matmul(tf_layer[-1], tf_weight[-1]) + tf_bias[-1])

            with tf.name_scope("cost"):
                cost = -tf.reduce_mean(tf_y * tf.log(hypothesis) + (1 - tf_y) * tf.log(1 - hypothesis))
                cost_summ = tf.summary.scalar("cost", cost)

            train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

            # cut off
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
            _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf_y), dtype=tf.float32))
            accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

            with tf.Session() as sess:
                merged_summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter("./logs/log_0" + str(k_fold + 1))
                writer.add_graph(sess.graph)  # Show the graph

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # if self.is_closed:
                for step in range(EPOCH + 1):
                    summary, cost_val, _ = sess.run([merged_summary, cost, train],
                                                    feed_dict={tf_x: x_train, tf_y: y_train})
                    writer.add_summary(summary, global_step=step)

                    if step % (EPOCH / 10) == 0:
                        print(str(step).rjust(5), cost_val)

                h, p, a = sess.run([hypothesis, predict, _accuracy], feed_dict={tf_x: x_test, tf_y: y_test})

            tf.reset_default_graph()

            _precision = precision_score(y_test, p)
            _recall = recall_score(y_test, p)

            print('\n\nlogistic regression')
            print('Precision : %.2f' % (_precision*100))
            print('Recall : %.2f' % (_recall*100))
            print('Accuracy : %.2f' % (a*100))

            return h, a, _precision, _recall

        def __show_plt__():
            logistic_plot.legend(loc="lower right")
            svm_plot.legend(loc="lower right")
            plt.show()

        def __show_score__():
            print("\n\n======================================\n")
            for k in precision:
                print(k)
                print("Total precision -", precision[k] / self.num_fold)
                print()

            for k in recall:
                print(k)
                print("Total recall -", recall[k] / self.num_fold)
                print()

            for k in accuracy:
                print(k)
                print("Total accuracy-Score -", accuracy[k] / self.num_fold)
                print()

        start_time = time.time()
        accuracy = {"logistic_regression": 0, "svm": 0}
        precision = {"logistic_regression": 0, "svm": 0}
        recall = {"logistic_regression": 0, "svm": 0}

        if do_show:
            fig = plt.figure(figsize=(10, 6))
            fig.suptitle("ROC CURVE", fontsize=16)
            svm_plot = plt.subplot2grid((2, 2), (0, 0))
            logistic_plot = plt.subplot2grid((2, 2), (0, 1))
            svm_plot.set_title("SVM")
            logistic_plot.set_title("Logistic regression")

            svm_plot.set_ylabel("TPR (sensitivity)")
            svm_plot.set_xlabel("1 - specificity")
            logistic_plot.set_ylabel("TPR (sensitivity)")
            logistic_plot.set_xlabel("1 - specificity")

        for k_fold in range(self.num_fold):
            x_train = self.vector_list[k_fold]["x_train"]["merge"]
            x_test = self.vector_list[k_fold]["x_test"]["merge"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            x_train_np = np.array([np.array(j) for j in x_train])
            x_test_np = np.array([np.array(j) for j in x_test])
            y_train_np = np.array([np.array(j) for j in y_train])
            y_test_np = np.array([np.array(j) for j in y_test])

            __show_shape__()

            logistic_probas_, accuracy['logistic_regression'], precision['logistic_regression'], recall['logistic_regression'] = __logistic_regression__()
            svm_probas_, accuracy['svm'], precision['svm'], recall['svm'] = __train_svm__()

            if do_show:
                logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probas_)
                roc_auc = auc(logistic_fpr, logistic_tpr)
                logistic_plot.plot(logistic_fpr, logistic_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % roc_auc)
                svm_fpr, svm_tpr, _ = roc_curve(y_test_np, svm_probas_[:, 1])
                roc_auc = auc(svm_fpr, svm_tpr)
                svm_plot.plot(svm_fpr, svm_tpr, alpha=0.3, label='ROC fold 1 (AUC = %0.2f)' % roc_auc)

        print("processing time     --- %s seconds ---" % (time.time() - start_time))

        if do_show:
            __show_plt__()
