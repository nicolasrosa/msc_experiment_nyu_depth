# =========== #
#  Libraries  #
# =========== #
import warnings
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Conv2D, Input, MaxPooling2D as Pool, BatchNormalization as BN, UpSampling2D, ZeroPadding2D, Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50, preprocess_input

from scipy import misc
from skimage.transform import resize

# from keras.applications import VGG16

warnings.filterwarnings("ignore")
showImages = True
saveModel = False


# =========== #
#  Functions  #
# =========== #
def model_1():
    input_layer = Input(shape=(224, 224, 1))
    conv_1_a = Conv2D(8, 3, activation="relu", padding="same")(input_layer)
    conv_1_b = Conv2D(8, 3, activation="relu", padding="same")(conv_1_a)
    pool_1 = Pool((2, 2))(conv_1_b)

    conv_2_a = Conv2D(16, 3, activation="relu", padding="same")(pool_1)
    conv_2_b = Conv2D(16, 3, activation="relu", padding="same")(conv_2_a)
    pool_2 = Pool((2, 2))(conv_2_b)

    bn = BN()(pool_2)

    conv_3_a = Conv2D(32, 3, activation="relu", padding="same")(bn)
    conv_3_b = Conv2D(32, 3, activation="relu", padding="same")(conv_3_a)
    conv_3_c = Conv2D(32, 3, activation="relu", padding="same")(conv_3_b)
    pool_3 = Pool((2, 2))(conv_3_c)

    up_1 = UpSampling2D((2, 2))(pool_3)
    conv_4_a = Conv2D(16, 3, activation="relu", padding="same")(up_1)
    conv_4_b = Conv2D(16, 3, activation="relu", padding="same")(conv_4_a)

    up_2 = UpSampling2D((2, 2))(conv_4_a)
    conv_5_a = Conv2D(8, 3, activation="relu", padding="same")(up_2)
    conv_5_b = Conv2D(8, 3, activation="relu", padding="same")(conv_5_a)

    up_3 = UpSampling2D((2, 2))(conv_5_b)
    conv_out = Conv2D(1, 3, activation="sigmoid", padding="same")(up_3)

    model = Model(inputs=input_layer, outputs=conv_out)

    return model


def model_2():
    input_layer = Input(shape=(224, 224, 3))
    conv = Conv2D(4, 3, activation="relu", padding="same")(input_layer)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(16, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(16, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(4, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(4, 3, activation="relu", padding="same")(conv)
    conv_out = Conv2D(1, 3, activation="sigmoid", padding="same")(conv)

    model = Model(inputs=input_layer, outputs=conv_out)

    return model


def model_3():
    input_layer = Input(shape=(224, 224, 3))

    from keras.layers import Conv2DTranspose as DeConv
    resnet = ResNet50(include_top=False, weights="imagenet")
    resnet.trainable = False

    res_features = resnet(input_layer)

    conv = DeConv(1024, padding="valid", activation="relu", kernel_size=3)(res_features)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(512, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(128, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(32, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(8, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(4, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = DeConv(1, padding="valid", activation="sigmoid", kernel_size=5)(conv)

    model = Model(inputs=input_layer, outputs=conv)

    return model


def model_4():
    # ----- Base Model ----- #
    resnet_model = ResNet50(include_top=False, weights="imagenet")
    # resnet_model.summary()

    # print(resnet_model.layers[0].output)
    # print(resnet_model.layers[1].output)
    # print(resnet_model.layers[2].output)
    # print(resnet_model.layers[3].output)

    # Removes previous input and conv1_pad layers
    resnet_model.layers.pop(0)
    # resnet_model.layers.pop(0)
    # resnet_model.layers.pop(0)
    # resnet_model.layers.pop()
    # resnet_model.summary()

    # ----- New Model ----- #
    # Overwrites ResNet layers
    new_input_layer = Input(batch_shape=(None, 224, 224, 1), name='input_1')  # FIXME: O mais correto seria (224, 224, 1)
    new_input_conc = Concatenate()([new_input_layer, new_input_layer, new_input_layer])
    # new_conv1_pad = ZeroPadding2D(padding=(1, 1))(new_input_layer)
    # new_conv_1 = Conv2D(1, 3, activation="relu", padding="same")(new_conv1_pad)
    # new_outputs = resnet_model(new_conv1_pad)

    resnet_output = resnet_model(new_input_conc)

    up_1 = UpSampling2D((2, 2))(resnet_output)
    conv_4_a = Conv2D(16, 3, activation="relu", padding="same")(up_1)
    conv_4_b = Conv2D(16, 3, activation="relu", padding="same")(conv_4_a)

    up_2 = UpSampling2D((2, 2))(conv_4_a)
    conv_5_a = Conv2D(8, 3, activation="relu", padding="same")(up_2)
    conv_5_b = Conv2D(8, 3, activation="relu", padding="same")(conv_5_a)

    up_3 = UpSampling2D((2, 2))(conv_5_b)
    conv_6_a = Conv2D(8, 3, activation="relu", padding="same")(up_3)
    conv_6_b = Conv2D(8, 3, activation="relu", padding="same")(conv_6_a)

    up_4 = UpSampling2D((2, 2))(conv_6_b)
    conv_7_a = Conv2D(8, 3, activation="relu", padding="same")(up_4)
    conv_7_b = Conv2D(8, 3, activation="relu", padding="same")(conv_7_a)

    up_5 = UpSampling2D((2, 2))(conv_7_b)
    new_outputs = Conv2D(1, 3, activation="sigmoid", padding="same")(up_5)

    new_model = Model(new_input_layer, new_outputs)

    # FIXME: Use `get_output_at(node_index)` instead.
    # print(new_model.layers[0].output)
    # print(new_model.layers[1].output)
    # print(new_model.layers[2].output)
    # print([item.get_output_at(0) for item in new_model.layers])
    # input("Continue...")

    return new_model


class NYUDepth:
    def __init__(self):
        self.depth_sparse_filenames = None
        self.depth_gt_filenames = None

        self.X_train = None
        self.X_valid = None
        self.Y_train = None
        self.Y_valid = None

    def read(self):
        """Get filenames for the input/output images"""

        self.depth_sparse_filenames = sorted(glob('nyu_depth/*/*_depth_sparse.png'))
        self.depth_gt_filenames = sorted(glob('nyu_depth/*/*_depth.png'))

    def train_test_split(self):
        # Inputs
        self.X_train = sorted(glob('nyu_depth/training/*_depth_sparse.png'))
        self.X_valid = sorted(glob('nyu_depth/testing/*_depth_sparse.png'))

        # Outputs
        self.Y_train = sorted(glob('nyu_depth/training/*_depth.png'))
        self.Y_valid = sorted(glob('nyu_depth/testing/*_depth.png'))

    def summary(self):
        if (self.depth_sparse_filenames is not None) and (self.depth_gt_filenames is not None):
            print(len(self.depth_sparse_filenames))
            print(len(self.depth_gt_filenames))
        else:
            print(self.X_train)
            print(len(self.X_train))
            print(self.X_valid)
            print(len(self.X_valid))
            print(self.Y_train)
            print(len(self.Y_train))
            print(self.Y_valid)
            print(len(self.Y_valid))


def read_imageX(dsPath):
    depth_sparse = misc.imread(dsPath).astype(np.uint16) / 1000.0
    depth_sparse_resized = resize(depth_sparse, output_shape=(224, 224))  # (480, 640) -> (224, 224)
    depth_sparse_resized_exp = np.expand_dims(np.expand_dims(depth_sparse_resized, -1), 0) # (224, 224) -> Model Input (1, 224, 224, 1)

    # print(depth_sparse.shape)
    # print(depth_sparse_resized.shape)

    # return np.expand_dims(np.expand_dims(depth_sparse_resized, -1), 0)  # (224, 224) -> (1, 224, 224, 1)
    return depth_sparse_resized_exp


def read_imageY(dPath):
    depth = misc.imread(dPath).astype(np.uint16) / 1000.0
    depth_resized = resize(depth, output_shape=(224, 224))  # (480,640) -> (224, 224)
    depth_resized_exp = np.expand_dims(np.expand_dims(depth_resized, -1), 0)  # (224, 224) -> Model Output (1, 224, 224, 1)

    return depth_resized_exp


def imageLoader(depth_sparse_filenames, depth_gt_filenames, batch_size=4):
    assert len(depth_sparse_filenames) == len(depth_gt_filenames)

    numSamples = len(depth_sparse_filenames)

    # This line is just to make the generator infinite, Keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < numSamples:
            limit = min(batch_end, numSamples)

            X_batch = np.concatenate(list(map(read_imageX, depth_sparse_filenames[batch_start:limit])), 0)
            Y_batch = np.concatenate(list(map(read_imageY, depth_gt_filenames[batch_start:limit])), 0)

            # print(X_batch.shape)
            # print(Y_batch.shape)
            # input("Pause")

            yield (X_batch, Y_batch)  # A tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def mat2uint8(mat):
    return cv2.convertScaleAbs(mat * (255 / np.max(mat)))  # Only for Visualization Purpose


class CollectOutputAndTarget(Callback):
    def __init__(self):
        super(CollectOutputAndTarget, self).__init__()
        self.inputs = []  # collect x_input batches
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_x_input = tf.Variable(0., validate_shape=False)
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        x_input = K.eval(self.var_x_input)
        y_true = K.eval(self.var_y_true)
        y_pred = K.eval(self.var_y_pred)

        # print(y_true)
        # print(y_pred)
        # print(mat2uint8(y_true))
        # print(mat2uint8(y_pred))

        if showImages:
            plt.figure(1)
            plt.imshow(x_input[0, :, :, 0])
            plt.draw()

            plt.figure(2)
            plt.imshow(y_true[0, :, :, 0])
            plt.draw()

            plt.figure(3)
            plt.imshow(y_pred[0, :, :, 0])
            plt.draw()

            plt.pause(0.001)


class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# ====== #
#  Main  #
# ====== #
if __name__ == "__main__":
    # ----- Dataset----- #
    dataset = NYUDepth()

    dataset.read()
    # dataset.train_test_split()
    dataset.summary()

    # ----- Model Definition----- #
    model_num = 4
    model_name = ['hourglass', 'block', 'resglass', 'pirate'][model_num - 1]
    if model_num == 1:
        model = model_1()
    elif model_num == 2:
        model = model_2()
    elif model_num == 3:
        model = model_3()
    else:
        model = model_4()

    model.summary()

    input("Press ENTER to start training...")  # TODO: Descomentar

    # ----- Training Configuration ----- #
    # Training Parameters
    lr = 1e-3
    decay = 1e-2
    batch_size = 4
    steps_per_epoch = 30000
    epochs = 10

    model.compile(loss="mse", optimizer=SGD(lr=lr, decay=decay))

    # CallBacks Declaration
    # initialize the variables and the `tf.assign` ops
    cbk = CollectOutputAndTarget()
    fetches = [tf.assign(cbk.var_x_input, model.inputs[0], validate_shape=False),
               tf.assign(cbk.var_y_true, model.targets[0], validate_shape=False),
               tf.assign(cbk.var_y_pred, model.outputs[0], validate_shape=False)]
    model._function_kwargs = {'fetches': fetches}  # use `model._function_kwargs` if using `Model` instead of `Sequential`

    # history = LossHistory()

    # ----- Run Training ----- #
    model.fit_generator(imageLoader(dataset.depth_sparse_filenames, dataset.depth_gt_filenames, batch_size),
                        steps_per_epoch, epochs, callbacks=[cbk])
    # model.fit_generator(imageLoader(dataset.depth_sparse_filenames, dataset.depth_gt_filenames, batch_size), steps_per_epoch, epochs, callbacks=[cbk, history])

    # ----- Results ----- #
    # print(history.losses)

    # ----- Save ----- #
    if saveModel:
        model.save_weights('weights_%s.h5' % model_name)
        model.save('model_%s.h5' % model_name) # FIXME: Não está salvando com o modelo da ResNet

    # out_folder = 'preds_final'
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # weightsPath = 'weights_%s.h5' % model_name
    #
    # lr = 0.1
    #
    # if os.path.exists(weightsPath):
    #     model.load_weights(weightsPath)
    #     lr /= 10
    #
    # for i in range(100):
    #     print(i)
    #
    #     img_gray = np.expand_dims(np.mean(X[0], axis=-1).astype(np.int), axis=-1)
    #     gt_dep = Y[0]*255.
    #     pred_dep = model.predict(X[:1])[0]*255.
    #
    #     print(img_gray.shape, gt_dep.shape, pred_dep.shape)
    #
    #     cv2.imwrite("{}/model_{}_ep_{}.jpg".format(out_folder, model_name, i), np.hstack((img_gray, gt_dep, pred_dep))) # TODO: Descomentar
    #
    #     model.compile(loss="mse", optimizer=SGD(lr=lr, decay=1e-2))
    #     model.fit(X,Y, epochs=10, verbose=1)
    #
    #     lr *= 0.95
    #
    #     model.save_weights('weights_%s.h5' % model_name)
    #     model.save('model_%s.h5' % model_name)

    print("Done.")
