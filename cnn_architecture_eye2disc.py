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

from modules.idrid import IDRID
from modules.network import *
from modules.plot import Plot

warnings.filterwarnings("ignore")
showImages = True
saveModel = False


# =========== #
#  Functions  #
# =========== #
def read_imageX(imagePath):
    image = misc.imread(imagePath)
    image_resized = resize(image, output_shape=(224, 224))  # (2848, 4288) -> (224, 224, 3)
    image_resized_exp = np.expand_dims(image_resized, 0)  # (224, 224, 3) -> Model Input (1, 224, 224, 3)

    # print(image_resized_exp.shape)

    # return np.expand_dims(np.expand_dims(image_resized, -1), 0)  # (224, 224) -> (1, 224, 224, 1)
    return image_resized_exp


def read_imageY(gtPath):
    gt = misc.imread(gtPath)
    gt_resized = resize(gt, output_shape=(224, 224))  # (2848, 4288) -> (224, 224)
    gt_resized_exp = np.expand_dims(np.expand_dims(gt_resized, -1), 0)  # (224, 224) -> Model Output (1, 224, 224, 1)

    return gt_resized_exp


def imageLoader(image_filenames, gt_filenames, batch_size=4):
    assert len(image_filenames) == len(gt_filenames)

    numSamples = len(image_filenames)

    # This line is just to make the generator infinite, Keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < numSamples:
            limit = min(batch_end, numSamples)

            X_batch = np.concatenate(list(map(read_imageX, image_filenames[batch_start:limit])), 0)
            Y_batch = np.concatenate(list(map(read_imageY, gt_filenames[batch_start:limit])), 0)

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
    dataset = IDRID()

    dataset.train_test_split()
    # dataset.summary()
    dataset.summary(showFilenames=True)

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
    epochs = 10000

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
    # FIXME: Wrong!!!
    model.fit_generator(generator=imageLoader(dataset.X_train, dataset.Y_train, batch_size),
                        steps_per_epoch=(len(dataset.X_train) // batch_size) + 1,
                        epochs=epochs,
                        validation_data=imageLoader(dataset.X_test, dataset.Y_test),
                        # FIXME: Estou usando as 654 imagens de teste para validação, o mais correto é utilizar uma parcela das de treinamento como validação, e deixar as imagens de teste exclusivamente para avaliação do método.
                        validation_steps=(len(dataset.X_test) // batch_size) + 1,
                        callbacks=[cbk])
    # callbacks=[cbk, history])

    # ----- Results ----- #
    # print(history.losses)

    # ----- Save ----- #
    if saveModel:
        model.save_weights('weights_%s.h5' % model_name)
        model.save('model_%s.h5' % model_name)  # FIXME: Não está salvando com o modelo da ResNet

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
