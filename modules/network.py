# =========== #
#  Libraries  #
# =========== #
from keras.layers import Conv2D, Input, MaxPooling2D as Pool, BatchNormalization as BN, UpSampling2D, Concatenate
from keras.models import Model
from keras.applications.resnet50 import ResNet50


# from keras.applications import VGG16


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
    new_input_layer = Input(batch_shape=(None, 224, 224, 3),
                            name='input_1')  # FIXME: O mais correto seria (224, 224, 1)
    # new_input_conc = Concatenate()([new_input_layer, new_input_layer, new_input_layer])
    # new_conv1_pad = ZeroPadding2D(padding=(1, 1))(new_input_layer)
    # new_conv_1 = Conv2D(1, 3, activation="relu", padding="same")(new_conv1_pad)
    # new_outputs = resnet_model(new_conv1_pad)

    resnet_output = resnet_model(new_input_layer)

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
