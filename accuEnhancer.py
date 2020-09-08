import numpy as np
import os
from keras.layers import Dense, Activation, Flatten, Dropout, Concatenate, BatchNormalization, Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from keras import Input, Model
import h5py
import tensorflow as tf
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import random
from tensorflow.python import pywrap_tensorflow
import argparse
threads = 30
config = tf.ConfigProto(intra_op_parallelism_threads=threads,
                         inter_op_parallelism_threads=threads,
                         allow_soft_placement=True)


parser = argparse.ArgumentParser(
    description="""Take a raw sequence and a labels bed like file and encode and store
    both as numpy arrays. Split up into traiing, test and validation samples.""")
parser.add_argument('--in_file',
                    help='Six column file. [<chr> <start> <end> <comma separated IDs> <raw sequence> <epi mark>].',
                    required=True)
parser.add_argument('--out_name',
                    help='Output file destination name, suffix',
                    required=True)
parser.add_argument('--epochs',
                    help='Number of epochs',
                    type=int,
                    required=True)
parser.add_argument('--trainable',
                    help='pretrined weight retrainable ("T" or "F")',
                    default="T")
parser.add_argument('--test_file',
                    help='using in-file testing or specified test file',
                    default="")
parser.add_argument('--deephaem_model_path',
                    help='using deephaem pre-trained model weights',
                    default="")
parser.add_argument('--train_from_ckpt',
                    help='load from self-trained checkpoint, int number',
                    default=None)
args = parser.parse_args()

train_file = args.in_file
test_file = args.test_file
trainable = args.trainable
deephaem_model_path = args.deephaem_model_path
train_from_ckpt = args.train_from_ckpt

""" Defining initial parameters """
out_name = args.out_name
num_class = 1


def recall_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_keras(y_true, y_pred):
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def recall_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    true_positive_count = K.sum(K.cast(K.equal(tf.add(y_true, y_pred), 2.), K.floatx()))
    label_positive_count = K.sum(y_true)
    recall = true_positive_count / label_positive_count
    return recall


def accuracy_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    correct_predictions = K.sum(K.cast(K.equal(y_true, y_pred), K.floatx()))
    all_instances = K.sum(K.cast(K.greater_equal(y_true, 0.), K.floatx()))
    accuracy = correct_predictions / all_instances
    return accuracy


def precision_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    true_positive_count = K.sum(K.cast(K.equal(tf.add(y_true, y_pred), 2.), K.floatx()))
    pred_positive_count = K.sum(y_pred)
    precision = true_positive_count / pred_positive_count
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def multi_category_focal_loss2_fixed(y_true, y_pred):
    epsilon = 1.e-7
    gamma=2.
    alpha_v = .7
    alpha = tf.constant([alpha_v], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)

    return loss

# ----- Start to Load data! -----
# Get the sets of seqs and labels from hdf5 formated file
print("Start to read from h5 files to Numpy array:")
h5f = h5py.File(train_file, 'r')
tr_total_len = len(h5f['training_seqs'])
epi_feature_count = len(h5f['training_epi_marks'][0])

dna_train = h5f['training_seqs'][0:tr_total_len]
y_train = h5f['training_labels'][0:tr_total_len]
epi_train = h5f['training_epi_marks'][0:tr_total_len].reshape([tr_total_len, epi_feature_count, 1])
h5f.close()

print("Finished reading training....")

print('\n\ntraining_data')
print(np.shape(dna_train))
training_cases = np.shape(dna_train)[0]

# make indices for shuffeling the data
training_index = np.asarray(range(training_cases))


# ----- Define the model structure -----
# build DNA module
inputs = Input(shape=(200, 4))
# Define first five layers
x = Convolution1D(filters=300, kernel_size=20, name="conv_l1", padding='SAME')(inputs)
x = MaxPooling1D(pool_size=4, strides=4, padding="same")(x)

x = Convolution1D(filters=600, kernel_size=10, name="conv_l2", padding='SAME')(x)
x = MaxPooling1D(pool_size=5, strides=5, padding="same")(x)

x = Convolution1D(filters=900, kernel_size=8, name="conv_l3", padding='SAME')(x)
x = MaxPooling1D(pool_size=5, strides=5, padding="same")(x)

x = Convolution1D(filters=900, kernel_size=4, name="conv_l4", padding='SAME')(x)
x = MaxPooling1D(pool_size=5, strides=5, padding="same")(x)

x = Convolution1D(filters=900, kernel_size=8, name="conv_l5", padding='SAME')(x)
x = MaxPooling1D(pool_size=2, strides=2, padding="same")(x)

x = Convolution1D(filters=100, kernel_size=20, name="conv_l6", padding='SAME')(x)
x = MaxPooling1D(pool_size=10, strides=2, padding="same")(x)

x = Convolution1D(filters=64, kernel_size=10, name="conv_l7", padding='SAME')(x)
x = MaxPooling1D(pool_size=5, strides=1, padding="same")(x)

x = Convolution1D(filters=32, kernel_size=10, name="conv_l8", padding='SAME')(x)
x = MaxPooling1D(pool_size=5, strides=1, padding="same")(x)

# build EPI module
epi_marks = Input(shape=(epi_feature_count, 1))

x1 = Convolution1D(filters=32, kernel_size=10, name="conv_epi1", padding='SAME')(epi_marks)
x1 = BatchNormalization()(x1)
x1 = Convolution1D(filters=16, kernel_size=5, name="conv_epi2", padding='SAME')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling1D(pool_size=5, strides=2, padding="same")(x1)

x1 = Convolution1D(filters=16, kernel_size=5, name="conv_epi3", padding='SAME')(x1)
x1 = BatchNormalization()(x1)
x1 = Convolution1D(filters=16, kernel_size=5, name="conv_epi4", padding='SAME')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling1D(pool_size=5, strides=2, padding="same")(x1)

# Pool together
x = Flatten()(x)
x1 = Flatten()(x1)
x = Concatenate()([x, x1])

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(32)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(num_class)(x)
pred = Activation('sigmoid')(x)

# Another way to define your optimizer
adam = Adam(lr=0.001)
# We add metrics to get more results you want to see
model = Model(inputs=[inputs, epi_marks], outputs=pred)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[accuracy_m, recall_m, precision_m, f1_m, recall_keras, precision_keras, f1_keras])


if trainable == "T":
    trainable = True
    if train_from_ckpt is not None:
        # 從 self-trained checkpoint中讀出資料
        train_from_ckpt = int(train_from_ckpt)
        model.load_weights('reports/data/H1/trained_%d.h5' %train_from_ckpt)
        print('Checkpoint %d weights are loaded.' %train_from_ckpt)
    else:
        print('Train from the beginning!')
else:
    trainable = False
    if deephaem_model_path != "":
        # 從deephaem checkpoint中讀出資料
        #reader = pywrap_tensorflow.NewCheckpointReader(deephaem_model_path)
        try:
            reader = np.load(deephaem_model_path)
            conv1_w = reader['arr_0'] #reader.get_tensor("Hidden_Conv1/weights")
            conv1_b = reader['arr_1'] #reader.get_tensor("Hidden_Conv1/Variable")
            conv2_w = reader['arr_2'] # reader.get_tensor("Hidden_Conv2/weights")
            conv2_b = reader['arr_3'] # reader.get_tensor("Hidden_Conv2/Variable")
            conv3_w = reader['arr_4'] # reader.get_tensor("Hidden_Conv3/weights")
            conv3_b = reader['arr_5'] # reader.get_tensor("Hidden_Conv3/Variable")
            conv4_w = reader['arr_6'] # reader.get_tensor("Hidden_Conv4/weights")
            conv4_b = reader['arr_7'] # reader.get_tensor("Hidden_Conv4/Variable")
            conv5_w = reader['arr_8'] # reader.get_tensor("Hidden_Conv5/weights")
            model.get_layer("conv_l1").set_weights([conv1_w, conv1_b])
            model.get_layer("conv_l1").trainable = trainable
            model.get_layer("conv_l2").set_weights([conv2_w, conv2_b])
            model.get_layer("conv_l2").trainable = trainable
            model.get_layer("conv_l3").set_weights([conv3_w, conv3_b])
            model.get_layer("conv_l3").trainable = trainable
            model.get_layer("conv_l4").set_weights([conv4_w, conv4_b])
            model.get_layer("conv_l4").trainable = trainable
            model.get_layer("conv_l5").set_weights([conv5_w, conv5_b])
            model.get_layer("conv_l5").trainable = trainable
            print("Deephaem model loaded!")
        except:
            assert False, 'Deephaem model loading failed! Terminate.'


# ----- Start training!! -----
if not os.path.exists('reports/%s' % out_name):
    os.makedirs('reports/%s' % out_name)
if train_from_ckpt is not None:
    output_fh = open('reports/%s/epoch_testing.out' % out_name, 'a+')
    output_fh.write("#epoch\taccuracy\trecall\tprecision\tf1_score\n")
    for i in range(train_from_ckpt+1, args.epochs):
        print('---- Training %d iteration------------' % i)
        model.fit([dna_train, epi_train], y_train, epochs=2, batch_size=4000, class_weight={0: 1, 1: 2}, validation_split=0.05)
        # Save current network structure and weights
        model.save("reports/%s/trained_%s.h5" % (out_name, str(i)))
        print('\nSave model to path: reports/%s/trained_%s.h5' % (out_name, str(i)))
        # Evaluate the model with the metrics we defined earlier
    output_fh.close()
else:
    output_fh = open('reports/%s/epoch_testing.out' % out_name, 'w')
    output_fh.write("#epoch\taccuracy\trecall\tprecision\tf1_score\n")
    for i in range(args.epochs):
        print('---- Training %d iteration------------' % i)
        model.fit([dna_train, epi_train], y_train, epochs=2, batch_size=4000, class_weight={0: 1, 1: 2}, validation_split=0.05)
        # Save current network structure and weights
        model.save("reports/%s/trained_%s.h5" % (out_name, str(i)))
        print('\nSave model to path: reports/%s/trained_%s.h5' % (out_name, str(i)))
        # Evaluate the model with the metrics we defined earlier
    output_fh.close()

print('\n---- Testing ------------')
print('\nOutput checkpoint testing result------------')


if len(test_file) > 0:
    print("Start to read from h5 files to Numpy array:")
    h5f = h5py.File(test_file, 'r')
else:
    h5f = h5py.File(train_file, 'r')

te_total_len = len(h5f['test_seqs'])
dna_test = h5f['test_seqs'][0:te_total_len]
y_test = h5f['test_labels'][0:te_total_len]
epi_test = h5f['test_epi_marks'][0:te_total_len].reshape([te_total_len, epi_feature_count, 1])

print('test_data')
print(dna_test.shape)
print('\n\n')
test_cases = np.shape(dna_train)[0]

predicts = model.predict([dna_test, epi_test])
true_positives = 0
predicted_positives = 0
truth_positives = 0
correct_predictions = 0
output_predict_fh = open('reports/%s/predict.out' % out_name, 'w')
for x, y in zip(predicts, y_test):
    output_predict_fh.write("%s\t%s\n" % (x[0], y[0]))
    if x[0] > 0.5 and y[0] > 0.5:
        true_positives += 1
        correct_predictions += 1
    if x[0] > 0.5:
        predicted_positives += 1
    if y[0] > 0.5:
        truth_positives += 1
    if x[0] < 0.5 and y[0] < 0.5:
        correct_predictions += 1
recall = true_positives / truth_positives
precision = true_positives / predicted_positives
f1 = 2 * recall * precision / (recall + precision)
accuracy = correct_predictions / len(predicts)
print('\ntest loss: ----')
print('\ntest accuracy: ', accuracy)
print('\ntest recall: ', recall)
print('\ntest precision: ', precision)
print('\ntest f1_score: ', f1)
output_predict_fh.write("PREDICTED_RESULT\taccuracy\trecall\tprecision\tf1\n")
output_predict_fh.write("PREDICTED_RESULT\t%f\t%f\t%f\t%f\n" % (accuracy, recall, precision, f1))
output_predict_fh.close()
