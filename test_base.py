#coding=utf8
import datetime
import gc
import os
import time

import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import learn

from data_helper import *
from model.cnn_model import TextCNN

BasePath = sys.path[0]
# Parameters
tf.flags.DEFINE_float("dev_sample_percentage", 0.10, "Percentage of the training data to use for validation")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")

# Misc Parametersfrom data_hel
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()#
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

topn_list = [10, 20, 30]
emb_size_list = [64, 128, 192, 256, 320]
# topn_list = [30]
# emb_size_list = [128]
for my_embedding_dim in emb_size_list:
    for topn in topn_list:
        # name_str = [l1_reg, l2_reg, t_reg, dropout, step_step]
        save_path = BasePath + "/logging_cnn/" + str(my_embedding_dim) + "cnn_logging_topn" + ".txt"
        if (os.path.exists(save_path)):
            y_predictions_array_file = BasePath + "/logging_cnn/"+str(my_embedding_dim)+"cnn_predictions_array" + str(topn) + ".txt"
            if (os.path.exists(y_predictions_array_file)):
                print("the path exists: ")
                print(save_path)
                print(y_predictions_array_file)
                continue
            else:
                f_logging = open(BasePath + "/logging_cnn/" + str(my_embedding_dim) + "cnn_logging_topn" + ".txt", "w+")
                print("make file with")
        else:
            f_logging = open(
                BasePath + "/logging_cnn/" + str(my_embedding_dim) + "cnn_logging_topn" + ".txt", "a")

        f_logging = open(BasePath + "/logging_cnn/" + str(my_embedding_dim) + "cnn_logging_topn" + ".txt", 'a')
        one_hot_vocab_path = BasePath + "/data/one_hot_vocab_" + str(topn) + ".txt"
        with open(one_hot_vocab_path, "rb") as f:
            one_hot_vocab = json.loads(f.read())

        print("the one_hot vocab is : ")
        print(one_hot_vocab)
        f_logging.write("the one_hot vocab is "+ str(topn) +": " + "\n")
        f_logging.write(str(one_hot_vocab) + "\n")

        train_dev_x_file_path = BasePath + "/data/all_train_dev_data_x.txt"
        train_dev_y_file_path = BasePath + "/data/all_train_dev_data_y_topn" + str(topn) + ".txt"
        x_text, y = get_dev_train_data(train_dev_x_file_path, train_dev_y_file_path)
        print(x_text[10])
        print(y[10])

        # Build vocabulary
        min_frequence = 10
        average_document_length = 1000
        print("average_document_length is : " + str(average_document_length))
        vocab_path = BasePath + "/vocab"
        if os.path.exists(vocab_path):
            print("the vocab exists")
            vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        else:
            vocab_processor = learn.preprocessing.VocabularyProcessor\
                                (average_document_length, min_frequency=min_frequence)

        x = np.array(list(vocab_processor.fit_transform(x_text)))

        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        del x
        del x_text
        del y
        gc.collect()
        print("finished free memory")
        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = False
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.45
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=my_embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )

                cnn.add_placeholders()
                cnn.add_embedding()
                cnn.add_conv_pool()
                cnn.add_dropout()
                cnn.add_output()
                cnn.add_loss()
                cnn.add_loss()
                cnn.add_accuracy()

                # Define Training procedure
                global_step = tf.Variable(0, name = "global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)


                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                ISOTIMEFORMAT ='%Y-%m-%d-%X'
                timestamp = time.strftime( ISOTIMEFORMAT, time.localtime())
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))
                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                # TODO: this step can add precision recall f1 score summaries

                # Train summaries
                train_summary_op = tf.summary.merge([loss_summary,  grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables())
                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # train_step
                def train_step(x_batch, y_batch):
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }

                    _, step, summaries, loss, y_predictions = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.predictions],
                        feed_dict)

                    assert len(y_predictions) ==len(y_batch), "the len of data does not match"
                    print("-----------------------------------------------------------------------------------")
                    f_logging.write("-------------------------------------------------------------------------------\n")

                    precision = metrics.precision_score(np.array(y_batch), y_predictions, average='samples')
                    recall = metrics.recall_score(np.array(y_batch), y_predictions, average='samples')
                    f1_score = metrics.f1_score(np.array(y_batch), y_predictions, average='samples')

                    precision_macro = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
                    recall_macro = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
                    f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')

                    precision_micro = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
                    recall_micro = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
                    f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')
                    hamming_loss = metrics.hamming_loss(np.array(y_batch), y_predictions)
                    jaccard = metrics.jaccard_similarity_score(np.array(y_batch), y_predictions)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                          .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
                    print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                          .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
                    print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                          .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
                    print("{}: step {}, jaccard similarity score is: {}"
                          .format(time_str, step, jaccard))
                    # logging
                    f_logging.write("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}\n"
                          .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
                    f_logging.write("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}\n"
                          .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
                    f_logging.write("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}\n"
                          .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
                    train_summary_writer.add_summary(summaries, step)

                    return [precision, recall, f1_score,
                            precision_macro, recall_macro, f1_score_macro,
                            precision_micro, recall_micro, f1_score_micro,
                            hamming_loss, jaccard]

                def dev_step(x_batch, y_batch, writer = None):
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }

                    step, summaries, loss, y_predictions = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.predictions],
                    feed_dict)

                    print(y_predictions[0:10])
                    print(np.array(y_batch[0:10]))

                    assert len(y_predictions) == len(y_batch), "the len of data does not match"
                    print("------------------------------------- DEV STEP ----------------------------------------------")
                    f_logging.write("-------------------------------------------------------------------------------\n")

                    precision = metrics.precision_score(np.array(y_batch), y_predictions, average='samples')
                    recall = metrics.recall_score(np.array(y_batch), y_predictions, average='samples')
                    f1_score = metrics.f1_score(np.array(y_batch), y_predictions, average='samples')

                    precision_macro = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
                    recall_macro = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
                    f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')

                    precision_micro = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
                    recall_micro = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
                    f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')
                    hamming_loss = metrics.hamming_loss(np.array(y_batch), y_predictions)
                    jaccard = metrics.jaccard_similarity_score(np.array(y_batch), y_predictions)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                          .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
                    print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                          .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
                    print("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}"
                          .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
                    print("{}: step {}, jaccard similarity score is: {}"
                          .format(time_str, step, jaccard))
                    # logging
                    f_logging.write("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}\n"
                          .format(time_str, step, loss, precision, recall, f1_score, hamming_loss))
                    f_logging.write("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}\n"
                          .format(time_str, step, loss, precision_macro, recall_macro, f1_score_macro, hamming_loss))
                    f_logging.write("{}: step {}, loss {:g}, precision {:g}, recall {:g}, f1_score {:g}, hamming_loss {:g}\n"
                          .format(time_str, step, loss, precision_micro, recall_micro, f1_score_micro, hamming_loss))
                    dev_summary_writer.add_summary(summaries, step)

                    return y_predictions, np.array(y_batch), [precision, recall, f1_score,
                            precision_macro, recall_macro, f1_score_macro,
                            precision_micro, recall_micro, f1_score_micro,
                            hamming_loss, jaccard]

                evaluate_every = 50
                checkpoint_every = 50

                train_result_list = list()
                dev_result_list = list()
                y_predictions_list = list()
                batches = batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_merge_result = train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    train_result_list.append(train_merge_result)

                    if current_step % evaluate_every == 0:
                        print("Evaluation: ")
                        f_logging.write("Evaluation: \n")
                        y_predictions, y_batch, dev_merge_result = \
                            dev_step(x_dev, y_dev, writer = dev_summary_writer)
                        dev_result_list.append(dev_merge_result)
                        if current_step > 5500:
                            y_predictions_list.append(y_predictions)
                    if current_step == 6001:
                        break

            result_file_path = BasePath + "/logging_cnn/" + str(my_embedding_dim) + "cnn_result" + str(topn) + ".json"
            encode_json = {"train_result_list" : train_result_list,
                           "dev_result_list" : dev_result_list}
            with open(result_file_path, "w+") as f:
                json.dump(encode_json, f, ensure_ascii = False)

            np.savetxt(BasePath + "/logging_cnn/"+str(my_embedding_dim)+"cnn_predictions_array" + str(topn) + ".txt", np.array(y_predictions_list,dtype="float32").reshape((1,-1)))
            np.savetxt(BasePath + "/logging_cnn/"+str(my_embedding_dim)+"cnn_batch_array" + str(topn) + ".txt", np.array(y_dev, dtype="float32").reshape((1,-1)))





