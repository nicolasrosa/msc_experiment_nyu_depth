# =========== #
#  Libraries  #
# =========== #
import numpy as np

from glob import glob
from keras.layers import Conv2D, Input, MaxPooling2D as Pool, BatchNormalization as BN, UpSampling2D
from keras.models import Model
from keras.optimizers import SGD
from scipy import misc
from skimage.transform import resize

# import cv2
# import sys, os
# from keras.applications.resnet50 import ResNet50, preprocess_input
# import keras.backend as K
# import glob
# from keras.applications import VGG16


# =========== #
#  Functions  #
# =========== #
def model_1():
    input_layer = Input(shape=(224,224,1))
    conv_1_a = Conv2D(8, 3, activation="relu", padding="same")(input_layer)
    conv_1_b = Conv2D(8, 3, activation="relu", padding="same")(conv_1_a)
    pool_1 = Pool((2,2))(conv_1_b)

    conv_2_a = Conv2D(16, 3, activation="relu", padding="same")(pool_1)
    conv_2_b = Conv2D(16, 3, activation="relu", padding="same")(conv_2_a)
    pool_2 = Pool((2,2))(conv_2_b)

    bn = BN()(pool_2)

    conv_3_a = Conv2D(32, 3, activation="relu", padding="same")(bn)
    conv_3_b = Conv2D(32, 3, activation="relu", padding="same")(conv_3_a)
    conv_3_c = Conv2D(32, 3, activation="relu", padding="same")(conv_3_b)
    pool_3 = Pool((2,2))(conv_3_c)

    up_1 = UpSampling2D((2,2))(pool_3)
    conv_4_a = Conv2D(16, 3, activation="relu", padding="same")(up_1)
    conv_4_b = Conv2D(16, 3, activation="relu", padding="same")(conv_4_a)

    up_2 = UpSampling2D((2,2))(conv_4_a)
    conv_5_a = Conv2D(8, 3, activation="relu", padding="same")(up_2)
    conv_5_b = Conv2D(8, 3, activation="relu", padding="same")(conv_5_a)

    up_3 = UpSampling2D((2,2))(conv_5_b)
    conv_out = Conv2D(1, 3, activation="sigmoid", padding="same")(up_3)

    model = Model(inputs=input_layer,outputs=conv_out)

    return model

def model_2():

    input_layer = Input(shape=(224,224,1))
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

    model = Model(inputs=input_layer,outputs=conv_out)

    return model

def model_3():

    input_layer = Input(shape=(224,224,3))

    from keras.layers import Conv2DTranspose as DeConv
    resnet = ResNet50(include_top=False, weights="imagenet")
    resnet.trainable = False

    res_features = resnet(input_layer)

    conv = DeConv(1024, padding="valid", activation="relu", kernel_size=3)(res_features)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(512, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(128, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(32, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(8, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(4, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = DeConv(1, padding="valid", activation="sigmoid", kernel_size=5)(conv)

    model = Model(inputs=input_layer, outputs=conv)
    return model

def model_4():
    input_layer = Input(shape=(224,224,3))
    conv_1_a = VGG16(weights='imagenet', include_top=False)(input_layer)
    # conv_1_a = Conv2D(8, 3, activation="relu", padding="same")(input_layer)
    conv_1_b = Conv2D(8, 3, activation="relu", padding="same")(conv_1_a)
    pool_1 = Pool((2,2))(conv_1_b)

    conv_2_a = Conv2D(16, 3, activation="relu", padding="same")(pool_1)
    conv_2_b = Conv2D(16, 3, activation="relu", padding="same")(conv_2_a)
    pool_2 = Pool((2,2))(conv_2_b)

    bn = BN()(pool_2)

    conv_3_a = Conv2D(32, 3, activation="relu", padding="same")(bn)
    conv_3_b = Conv2D(32, 3, activation="relu", padding="same")(conv_3_a)
    conv_3_c = Conv2D(32, 3, activation="relu", padding="same")(conv_3_b)
    pool_3 = Pool((2,2))(conv_3_c)

    up_1 = UpSampling2D((2,2))(pool_3)
    conv_4_a = Conv2D(16, 3, activation="relu", padding="same")(up_1)
    conv_4_b = Conv2D(16, 3, activation="relu", padding="same")(conv_4_a)

    up_2 = UpSampling2D((2,2))(conv_4_a)
    conv_5_a = Conv2D(8, 3, activation="relu", padding="same")(up_2)
    conv_5_b = Conv2D(8, 3, activation="relu", padding="same")(conv_5_a)

    up_3 = UpSampling2D((2,2))(conv_5_b)
    conv_out = Conv2D(1, 3, activation="sigmoid", padding="same")(up_3)

    model = Model(inputs=input_layer,outputs=conv_out)

    return model

def read_imageX(dsPath):
    depth_sparse = misc.imread(dsPath).astype(np.uint16)/1000.0
    depth_sparse_resized = resize(depth_sparse, output_shape=(224, 224))  # (480,640) -> Model Output (224, 224)

    return np.expand_dims(np.expand_dims(depth_sparse_resized, -1), 0)  # (224, 224) -> (1, 224, 224, 1)


def read_imageY(dPath):
    depth = misc.imread(dPath).astype(np.uint16)/1000.0
    depth_resized = resize(depth, output_shape=(224, 224))  # (480,640) -> Model Output (224, 224)

    return np.expand_dims(np.expand_dims(depth_resized,-1), 0)  # (224, 224) -> (1, 224, 224, 1)

# TODO: Se as funções acima funcionarem não preciso desta função
def read_image(i):
    dsPath = depth_sparse_filenames[i]
    dPath = depth_gt_filenames[i]
    print(i)

    depth_sparse = misc.imread(dsPath).astype(np.uint16)/1000.0
    depth_sparse_resized = resize(depth_sparse, output_shape=(224, 224))

    depth = misc.imread(dPath).astype(np.uint16)/1000.0
    depth_resized = resize(depth, output_shape=(224, 224))  # (480,640) -> Model Output (224, 224)

    # print(np.expand_dims(depth_sparse_resized,-1).shape)
    # print(np.expand_dims(depth_resized,-1).shape)
    # print(preprocess_input(np.expand_dims(depth_sparse_resized,-1)))
    # input("aki")

    # stacked_img = np.stack((depth_sparse_resized,) * 3, axis=-1)
    # print(stacked_img.shape)
    # input('aki')

    # return stacked_img, np.expand_dims(depth_resized,-1)
    return np.expand_dims(depth_sparse_resized,-1), np.expand_dims(depth_resized,-1)

    # return preprocess_input(np.expand_dims(depth_sparse_resized,-1)), np.expand_dims(depth_resized,-1)
    # return preprocess_input(misc.imread(rPath)/1.), np.expand_dims(misc.imread(dPath, 0),-1)/255.
    # return preprocess_input(cv2.imread(rPath)/1.), np.expand_dims(cv2.imread(dPath, 0),-1)/255.

class NYUDepth():
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

# fileList = listOfFiles

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

            yield (X_batch, Y_batch) # A tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size



# ====== #
#  Main  #
# ====== #
if __name__ == "__main__":
    dataset = NYUDepth()

    dataset.read() # FIXME: Remove Limite, but attention more than 1200 figures will freeze system!!!
    # dataset.train_test_split()
    dataset.summary()

    model = model_1()


    lr = 1e-3
    model.compile(loss="mse", optimizer=SGD(lr=lr, decay=1e-2))
    model.fit_generator(imageLoader(dataset.depth_sparse_filenames, dataset.depth_gt_filenames, batch_size=4), steps_per_epoch=100, epochs=10)



    # images = map(read_image, range(len(depth_sparse_filenames)))
    # X, Y = map(np.array, zip(*images))
    #
    # # model_num = int(sys.argv[1])
    # model_num = 1
    # model_name = ['hourglass','block','resglass'][model_num - 1]
    # if model_num == 1:
    #     model = model_1()
    # elif model_num == 2:
    #     model = model_2()
    # elif model_num == 3:
    #     # model = VGG16(weights='imagenet', include_top=False)
    #     model = model_4()
    # else:
    #     model = model_3()
    # model.summary()
    #
    # print(X.shape, Y.shape)
    # print('Training ...')
    #
    # out_folder = 'preds_final'
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # weightsPath = 'weigts_%s.h5' % model_name
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