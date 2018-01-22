from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input,  merge
from keras.layers import Convolution2D, MaxPooling2D, SeparableConvolution2D
from keras.layers.normalization import BatchNormalization

import math
def cf(x: float, eps=0.0001):
    return int(math.ceil(x - eps))

def get_kerasnet(nb_classes, input_shape, ch_axis):
    model = Sequential()
    model.add(BatchNormalization(axis=ch_axis, input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='poolspot'))
    model.add(Dropout(0.25, name='dropspot'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def get_xkerasnet(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    Y_to_UV = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_UV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_UV], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_full(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_V = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    # V_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    # V_to_U = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)
    # Y_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)
    # Y_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_elucon_exp(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolU)
    # U_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolU)
    # U_to_V = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolV)
    # V_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolV)
    # V_to_U = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='elu')(poolY)
    # Y_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='elu')(poolY)
    # Y_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_idcon_exp(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    # U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    # V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    # Y_to_UV = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)

    Ymap = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    Ymap = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(Ymap)
    Umap = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Umap)
    Vmap = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_idcon_exp2(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    y = .5
    u = .25
    v = .25
    Y64 = cf(64*y)
    U64 = cf(64*u)
    V64 = cf(64*v)

    Y128 = cf(128 * y)
    U128 = cf(128 * u)
    V128 = cf(128 * v)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(Y64, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(U64, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(V64, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(Y64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(U64, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(V64, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    # U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    # V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    # Y_to_UV = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)

    Ymap = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    Ymap = Convolution2D(Y64, 1, 1, border_mode='same', activation='relu')(Ymap)
    Umap = Convolution2D(U64, 1, 1, border_mode='same', activation='relu')(Umap)
    Vmap = Convolution2D(V64, 1, 1, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(Y128, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(U128, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(V128, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(Y128, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(U128, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(V128, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_nincon_exp(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_V = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_U = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_nincon_exp2(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(U_to_V)

    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(V_to_Y)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolV)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(V_to_U)

    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(Y_to_U)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(poolY)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_ninconelu_exp(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolU)
    U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(U_to_Y)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolU)
    U_to_V = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(U_to_V)

    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolV)
    V_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(V_to_Y)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(poolV)
    V_to_U = Convolution2D(16, 1, 1, border_mode='same', activation='elu')(V_to_U)

    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='elu')(poolY)
    Y_to_U = Convolution2D(32, 1, 1, border_mode='same', activation='elu')(Y_to_U)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='elu')(poolY)
    Y_to_V = Convolution2D(32, 1, 1, border_mode='same', activation='elu')(Y_to_V)

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model

def get_xkerasnet_xcepcon_exp(nb_classes, input_shape, ch_axis):
    def get_slice(axis, axis_id, input_shape):
        return Lambda(
            lambda x: x[
                [slice(None) if i != axis else slice(axis_id, axis_id + 1) for i in range(len(input_shape) + 1)]],
            output_shape=[p if i + 1 != axis else 1 for i, p in enumerate(input_shape)])

    inputYUV = Input(shape=input_shape)
    inputNorm = BatchNormalization(axis=ch_axis)(inputYUV)

    inputY = get_slice(ch_axis, 0, input_shape)(inputNorm)
    inputU = get_slice(ch_axis, 1, input_shape)(inputNorm)
    inputV = get_slice(ch_axis, 2, input_shape)(inputNorm)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(inputY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputV)

    convY = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    # U_to_Y = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(poolU)
    # U_to_Y = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(U_to_Y)
    U_to_Y = SeparableConvolution2D(16, 3, 3, activation='relu', border_mode='same')(poolU)
    U_to_Y = merge([poolU,U_to_Y], mode='sum')

    U_to_V = SeparableConvolution2D(16, 3, 3, activation='relu', border_mode='same')(poolU)
    U_to_V = merge([poolU, U_to_V], mode='sum')

    V_to_Y = SeparableConvolution2D(16, 3, 3, activation='relu', border_mode='same')(poolV)
    V_to_Y = merge([poolV, V_to_Y], mode='sum')

    V_to_U = SeparableConvolution2D(16, 3, 3, activation='relu', border_mode='same')(poolV)
    V_to_U = merge([poolV, V_to_U], mode='sum')

    Y_to_V = SeparableConvolution2D(32, 3, 3, activation='relu', border_mode='same')(poolY)
    Y_to_V = merge([poolY, Y_to_V], mode='sum')

    Y_to_U = SeparableConvolution2D(32, 3, 3, activation='relu', border_mode='same')(poolY)
    Y_to_U = merge([poolY, Y_to_U], mode='sum')

    Ymap = merge([poolY, U_to_Y, V_to_Y], mode='concat', concat_axis=ch_axis)
    Umap = merge([poolU, Y_to_U, V_to_U], mode='concat', concat_axis=ch_axis)
    Vmap = merge([poolV, Y_to_V, U_to_V], mode='concat', concat_axis=ch_axis)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(Ymap)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Umap)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(Vmap)

    convY = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(convY)
    convU = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convU)
    convV = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(convV)

    poolY = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convY)
    poolU = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convU)
    poolV = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(convV)

    poolY = Dropout(0.25)(poolY)
    poolU = Dropout(0.25)(poolU)
    poolV = Dropout(0.25)(poolV)

    concatenate_map = merge([poolY, poolU, poolV], mode='concat', concat_axis=ch_axis)

    reshape = Flatten()(concatenate_map)
    fc = Dense(512, activation='relu')(reshape)
    fc = Dropout(0.5)(fc)
    out = Dense(nb_classes, activation='softmax')(fc)

    model = Model(input=inputYUV, output=out)
    return model