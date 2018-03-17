import tensorflow as tf
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from DeepPredict.learning.variables import EPOCH


class MyTrain:
    def __init__(self, vector_list, is_closed):
        self.vector_list = vector_list
        self.num_fold = len(self.vector_list)
        self.epoch = EPOCH
        self.is_closed = is_closed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def training(self):
        def __show_shape__():
            def __count_mortality__(_y_data_):
                _count = 0
                for _i in _y_data_:
                    if _i == [float(1)]:
                        _count += 1

                return _count

            print("dims - ", len(x_train[0]))
            print("learning count -", len(y_train), "\t mortality count -", __count_mortality__(y_train))
            print("test     count -", len(y_test), "\t mortality count -", __count_mortality__(y_test), "\n")

            print(np.shape(x_train_np), np.shape(y_train_np))
            print(np.shape(x_test_np), np.shape(y_test_np))

        def __train_svm__():

            model = SVC(kernel='rbf', C=1.0, random_state=0, probability=True)
            model.fit(x_train_np, y_train_np)
            y_pred = model.predict(x_test_np)
            y_score = model.decision_function(x_test_np)
            probas_ = model.predict_proba(x_test_np)

            average = average_precision_score(self.vector_list[k_fold]["y_test"], y_score)

            precision, recall, _ = precision_recall_curve(y_test_np, y_score)

            print('Accuracy: %.2f' % (accuracy_score(y_test_np, y_pred) * 100))
            print('Average precision-recall : %.2f' % average)

            # return accuracy_score(y_test_np, y_pred), average, probas_, y_test_np

        def __logistic_regression__():
            dimension = len(x_train[0])

            X = tf.placeholder(dtype=tf.float32, shape=[None, dimension])
            Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

            # W = tf.Variable(tf.random_normal([dimension, 1]), name="weight")
            W = tf.get_variable("weight", dtype=tf.float32, shape=[dimension, dimension],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([dimension]), name="bias")
            L = tf.nn.relu(tf.matmul(X, W) + b)

            W2 = tf.get_variable("weight2", dtype=tf.float32, shape=[dimension, dimension],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([dimension]), name="bias2")
            L2 = tf.nn.relu(tf.matmul(L, W2) + b2)

            W3 = tf.get_variable("weight3", dtype=tf.float32, shape=[dimension, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([1]), name="bias3")

            hypothesis = tf.sigmoid(tf.matmul(L2, W3) + b3)

            with tf.name_scope("cost") as scope:
                cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

                # if self.is_closed:
                cost_summ = tf.summary.scalar("cost", cost)

            train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

            # cut off
            predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

            # if self.is_closed:
            accuracy_summ = tf.summary.scalar("accuracy", accuracy)

            with tf.Session() as sess:
                merged_summary = tf.summary.merge_all()
                # if self.is_closed:
                writer = tf.summary.FileWriter("./logs/log_0" + str(k_fold + 1))
                writer.add_graph(sess.graph)  # Show the graph

                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # if self.is_closed:
                for step in range(EPOCH + 1):
                    summary, cost_val, _ = sess.run([merged_summary, cost, train],
                                                    feed_dict={X: x_train, Y: y_train})
                    writer.add_summary(summary, global_step=step)

                    if step % (EPOCH / 10) == 0:
                        print(str(step).rjust(5), cost_val)
                # else:
                #     for step in range(EPOCH + 1):
                #         cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
                #
                #         if step % (EPOCH / 10) == 0:
                #             print(str(step).rjust(5), cost_val)

                h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X: x_test, Y: y_test})

            tf.reset_default_graph()

        for k_fold in range(self.num_fold):
            x_train = self.vector_list[k_fold]["x_train"]
            x_test = self.vector_list[k_fold]["x_test"]
            y_train = self.vector_list[k_fold]["y_train"]
            y_test = self.vector_list[k_fold]["y_test"]

            x_train_np = np.array([np.array(j) for j in x_train])
            x_test_np = np.array([np.array(j) for j in x_test])
            y_train_np = np.array([np.array(j) for j in y_train])
            y_test_np = np.array([np.array(j) for j in y_test])

            __show_shape__()
            __train_svm__()
            __logistic_regression__()
