# =========== #
#  Libraries  #
# =========== #
import warnings

import cv2
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import SGD

from scipy import misc
from skimage.transform import resize

from modules.nyudepth import NYUDepth, NYUDepth2
from modules.network import *
from modules.plot import Plot

warnings.filterwarnings("ignore")
showImages = True
saveModel = False


# =========== #
#  Functions  #
# =========== #
# TODO: não é depth_sparse e sim colors
def read_imageX(dsPath):
    depth_sparse = misc.imread(dsPath)
    depth_sparse_resized = resize(depth_sparse, output_shape=(224, 224))  # (480, 640, 3) -> (224, 224, 3)


    depth_sparse_resized_exp = np.expand_dims(depth_sparse_resized, 0)  # (224, 224) -> Model Input (1, 224, 224, 1)

    # print(depth_sparse_resized_exp.shape)
    # input('aki')

    # print(depth_sparse.shape)
    # print(depth_sparse_resized.shape)

    # return np.expand_dims(np.expand_dims(depth_sparse_resized, -1), 0)  # (224, 224) -> (1, 224, 224, 1)
    return depth_sparse_resized_exp


def read_imageY(dPath):
    depth = misc.imread(dPath).astype(np.uint16) / 1000.0
    depth_resized = resize(depth, output_shape=(224, 224))  # (480,640) -> (224, 224)
    depth_resized_exp = np.expand_dims(np.expand_dims(depth_resized, -1),
                                       0)  # (224, 224) -> Model Output (1, 224, 224, 1)

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

        self.train_plotObj = Plot('train', title='Train Predictions')

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
            self.train_plotObj.showTrainResults(x_input[0, :, :, :], y_true[0, :, :, 0], y_pred[0, :, :, 0])


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
    dataset = NYUDepth2()

    # dataset.read()
    dataset.train_test_split()
    dataset.summary()

    # ----- Model Definition----- #
    model_num = 5
    model_name = ['hourglass', 'block', 'resglass', 'pirate', 'cachorro'][model_num - 1]
    if model_num == 1:
        model = model_1()
    elif model_num == 2:
        model = model_2()
    elif model_num == 3:
        model = model_3()
    elif model_num == 4:
        model = model_4()
    else:
        model = model_5()

    model.summary()

    # input("Press ENTER to start training...")  # TODO: Descomentar

    # ----- Training Configuration ----- #
    # Training Parameters
    lr = 1e-3
    decay = 1e-2
    batch_size = 4
    epochs = 200

    model.compile(loss="mse", optimizer=SGD(lr=lr, decay=decay))

    # CallBacks Declaration
    # initialize the variables and the `tf.assign` ops
    cbk = CollectOutputAndTarget()
    fetches = [tf.assign(cbk.var_x_input, model.inputs[0], validate_shape=False),
               tf.assign(cbk.var_y_true, model.targets[0], validate_shape=False),
               tf.assign(cbk.var_y_pred, model.outputs[0], validate_shape=False)]
    model._function_kwargs = {
        'fetches': fetches}  # use `model._function_kwargs` if using `Model` instead of `Sequential`

    # history = LossHistory()

    # ----- Run Training ----- #
    model.fit_generator(imageLoader(dataset.X_train, dataset.Y_train, batch_size),
                        steps_per_epoch=(len(dataset.X_train) // batch_size) + 1,
                        epochs=epochs,
                        validation_data=imageLoader(dataset.X_test, dataset.Y_test),
                        validation_steps=(len(dataset.X_test) // batch_size) + 1,
                        callbacks=[cbk])
    # callbacks=[cbk, history])

    # ----- Results ----- #
    # print(history.losses)

    # ----- Save ----- #
    if saveModel:
        model.save_weights('weights_%s.h5' % model_name)
        model.save('model_%s.h5' % model_name)

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
