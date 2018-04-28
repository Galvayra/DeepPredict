import tensorflow as tf
import DeepPredict.arguments as op
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from .variables import *
import os
import shutil


class MyNeuralNetwork:
    def __init__(self):
        self.__score = {
            "P": 0.0,
            "R": 0.0,
            "F1": 0.0,
            "Acc": 0.0,
            "AUC": 0.0
        }

    @property
    def score(self):
        return self.__score

    @staticmethod
    def __init_log_file_name(k_fold):
        log_name = "./logs/" + op.SAVE_DIR_NAME + op.USE_ID + "log_"

        if op.NUM_HIDDEN_LAYER < 10:
            log_name += "h_0" + str(op.NUM_HIDDEN_LAYER)
        else:
            log_name += "h_" + str(op.NUM_HIDDEN_LAYER)

        log_name += "_ep_" + str(op.EPOCH) + "_k_" + str(k_fold + 1)

        if op.USE_W2V:
            log_name += "_w2v"

        return log_name

    @staticmethod
    def __init_save_dir(k_fold):
        if op.NUM_HIDDEN_LAYER < 10:
            _hidden_ = "h_0" + str(op.NUM_HIDDEN_LAYER)
        else:
            _hidden_ = "h_" + str(op.NUM_HIDDEN_LAYER)

        _epoch_ = "_ep_" + str(op.EPOCH) + "_"

        _save_dir = TENSOR_PATH + op.SAVE_DIR_NAME
        if not os.path.isdir(_save_dir):
            os.mkdir(_save_dir)

        _save_dir += _hidden_ + _epoch_ + str(k_fold + 1) + "/"
        if os.path.isdir(_save_dir):
            shutil.rmtree(_save_dir)
        os.mkdir(_save_dir)

        return _save_dir

    @staticmethod
    def __load_tensor(k_fold):
        if op.NUM_HIDDEN_LAYER < 10:
            _hidden_ = "h_0" + str(op.NUM_HIDDEN_LAYER)
        else:
            _hidden_ = "h_" + str(op.NUM_HIDDEN_LAYER)

        _epoch_ = "_ep_" + str(op.EPOCH) + "_"
        _save_dir = TENSOR_PATH + op.SAVE_DIR_NAME

        if op.USE_ID:
            tensor_load = _save_dir + op.USE_ID.split('#')[0] + "_"
        else:
            tensor_load = _save_dir

        tensor_load += _hidden_ + _epoch_ + str(k_fold + 1) + "/"

        return tensor_load

    def feed_forward_nn(self, k_fold, x_train, y_train, x_test, y_test, plot):
        save_dir = self.__init_save_dir(k_fold)
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
        keep_prob = tf.placeholder(tf.float32)

        # make hidden layers
        for i in range(op.NUM_HIDDEN_LAYER):
            num_hidden_node = int(num_input_node / RATIO_HIDDEN)

            # append weight
            tf_weight.append(tf.get_variable("h_weight_" + str(i + 1), dtype=tf.float32,
                                             shape=[num_input_node, num_hidden_node],
                                             initializer=tf.contrib.layers.xavier_initializer()))
            # append bias
            tf_bias.append(tf.Variable(tf.random_normal([num_hidden_node]), name="h_bias_" + str(i + 1)))

            # append hidden layer
            hidden_layer = tf.nn.relu(tf.add(tf.matmul(tf_layer[i], tf_weight[i]), tf_bias[i]))
            tf_layer.append(tf.nn.dropout(hidden_layer, keep_prob=keep_prob, name="dropout_" + str(i + 1)))

            num_input_node = int(num_input_node / RATIO_HIDDEN)

        tf_weight.append(tf.get_variable("o_weight", dtype=tf.float32, shape=[num_hidden_node, 1],
                                         initializer=tf.contrib.layers.xavier_initializer()))
        tf_bias.append(tf.Variable(tf.random_normal([1]), name="o_bias"))

        hypothesis = tf.matmul(tf_layer[-1], tf_weight[-1]) + tf_bias[-1]
        hypothesis = tf.sigmoid(hypothesis, name=NAME_HYPO)

        with tf.name_scope("cost"):
            cost = -tf.reduce_mean(tf_y * tf.log(hypothesis) + (1 - tf_y) * tf.log(1 - hypothesis))
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=tf_y))
            cost_summ = tf.summary.scalar("cost", cost)

        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        # cut off
        predict = tf.cast(hypothesis > 0.5, dtype=tf.float32, name=NAME_PREDICT)
        _accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf_y), dtype=tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", _accuracy)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.__init_log_file_name(k_fold))
            writer.add_graph(sess.graph)  # Show the graph

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # if self.is_closed:
            for step in range(op.EPOCH + 1):
                summary, cost_val, _ = sess.run([merged_summary, cost, train_op],
                                                feed_dict={tf_x: x_train, tf_y: y_train, keep_prob: 0.7})
                writer.add_summary(summary, global_step=step)

                if op.DO_SHOW and step % (op.EPOCH / 10) == 0:
                    print(str(step).rjust(5), cost_val)

            h, p, acc = sess.run([hypothesis, predict, _accuracy], feed_dict={tf_x: x_test, tf_y: y_test, keep_prob: 1})

            saver.save(sess, save_dir + "model", global_step=op.EPOCH)

        tf.reset_default_graph()

        _precision = precision_score(y_test, p)
        _recall = recall_score(y_test, p)
        _f1 = f1_score(y_test, p)
        _logistic_fpr, _logistic_tpr, _ = roc_curve(y_test, h)
        _auc = auc(_logistic_fpr, _logistic_tpr)

        if _precision == 0 or _recall == 0:
            print("\n\n------------\nIt's not working")
            print('k-fold : %d, Precision : %.2f, Recall : %.2f' % (k_fold + 1, (_precision * 100), (_recall * 100)))
            print("\n------------")

        self.__score["P"] += _precision
        self.__score["R"] += _recall
        self.__score["F1"] += _f1
        self.__score["Acc"] += acc
        self.__score["AUC"] += _auc

        if op.DO_SHOW:
            print('\n\n')
            print(k_fold + 1, "fold", 'logistic regression')
            print('Precision : %.2f' % (_precision * 100))
            print('Recall    : %.2f' % (_recall * 100))
            print('F1-Score  : %.2f' % (_f1 * 100))
            print('Accuracy  : %.2f' % (acc * 100))
            print('AUC       : %.2f' % (_auc * 100))
            plot.plot(_logistic_fpr, _logistic_tpr, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (k_fold + 1, _auc))

    def load_feed_forward_nn(self, k_fold, x_test, y_test, plot):
        tensor_load = self.__load_tensor(k_fold)

        sess = tf.Session()
        saver = tf.train.import_meta_graph(tensor_load + 'model-' + str(op.EPOCH) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(tensor_load))

        print("\n\n\nRead Neural Network -", tensor_load, "\n")
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
            plot.plot(logistic_fpr, logistic_tpr, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (k_fold+1, _auc))

        self.__score["P"] += _precision
        self.__score["R"] += _recall
        self.__score["F1"] += _f1
        self.__score["Acc"] += _accuracy
        self.__score["AUC"] += _auc
