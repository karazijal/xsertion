from keras.engine.topology import Container
from keras.models import Sequential, Model, model_from_json
from keras.layers import Convolution2D, Input, Dropout

import keras.backend as K
from keras.backend.tensorflow_backend import clear_session, set_session


from keras import optimizers

def simple_model_func(nb_filter):
    i = Input(shape=(32,32,3)) # the shape does not matter
    d = Dropout(.25, name='d')(i)
    c = Convolution2D(nb_filter, 1, 1, activation='relu', name='c')(d)
    return Model(input=i, output=c)

def get_Sequential():
    return Sequential

def get_Container():
    return Container

def get_Model():
    return Model

def m_f_json(desc : str):
    return model_from_json(desc)

def is_valid_optimizers(opt):
    if K.backend() == "tensorflow":
        import tensorflow as tf
        return isinstance(opt, (str, optimizers.Optimizer, tf.train.Optimizer))
    else:
        return isinstance(opt, (str, optimizers.Optimizer))

def is_valid_loss(loss):
    if isinstance(loss, str):
        return True
    elif isinstance(loss, (list, dict)):
        return len(loss)==1 # should only contain single loss function for all; mind need to relax this for

    else:
        import inspect
        return inspect.isfunction(loss)

def keras_conv2d_config(filters, *args, **kwargs):
    return Convolution2D(filters, *args, **kwargs).get_config()

def unwrap_sequential(model):
    assert isinstance(model, get_Model())
    # Work with Model if we have Sequential (Sequential is a wrapper)
    to_rtn = model
    if type(model) is get_Sequential():
        if not model.built:
            model.build()
        to_rtn = model.model
    assert issubclass(type(to_rtn), get_Container()), "Model is {}".format(type(to_rtn))
    return to_rtn


class ModelProxy():
    """Proxy model that saves model weihts outside of the tf session definition, so that sessions could be reset"""
    def __init__(self, model):
        if K.backend() == 'tensorflow':
            self.mode ='tf'
            self.model_dict = dict()
            for layer in model.layers:
                lname = layer.name
                self.model_dict[lname] = dict()
                self.model_dict[lname]['t'] = type(layer)
                self.model_dict[lname]['w'] = layer.get_weights()
        else:
            self.model = model

    def get_weights(self, lname):
        if self.mode=='tf':
            if lname in self.model_dict:
                return self.model_dict[lname]['w']
            else:
                raise ValueError
        else:
            return self.model.get_layer(name=lname).get_weights()


def transfer_weights(new_model : Model, old_model : ModelProxy, verbose=0):
        # logger.info('Performing model weight bootstrapping')
        # prev_model = prev_models[ind]
        full_transfer=True
        for layer in new_model.layers:
            lname = layer.name
            try:
                weights = old_model.get_weights(lname)
                layer.set_weights(weights)
                if verbose:
                    print(lname, "Weights transferred")

            except ValueError:
                full_transfer = False  # some layer did not transfer so models are not the same
                if verbose:
                    print('Weights did not transfer for', lname)
        return full_transfer

def average_weights(dest_model :Model, source_models :list, verbose=0):
    import numpy as np
    # dest_model = models[0]
    for layer in dest_model.layers:
        lname = layer.name
        weights = []
        for m in source_models:
            try:
                weight = m.get_weights(lname)
                weights.append(weight)
            except:
                if verbose:
                    print("some models did not contain layer", lname)
        try:
            new_weights = np.mean(weights, axis=0)
            layer.set_weights(new_weights)
            if verbose:
                print(lname, "Weights set")
        except ValueError:
            if verbose:
                print('Layer {} weights could not be set to base'.format(lname))
    return dest_model

def shape_match(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for i,j in zip(shape1, shape2):
        if i!=j:
            return False
    return True

def average_transfer_weights(new_model : Model, old_models : list, verbose=0):
    import numpy as np
    for layer in new_model.layers:
        lname= layer.name
        # weight_shape = np.array(layer.get_weights()).shape
        # weight = layer.get_weights()
        weights = []
        for m in old_models:
            try:
                w = m.get_weights(lname)
                if w is not None:
                    shape = np.array(w).shape
                    # if shape_match(shape, weight_shape):
                    weights.append(w)
            except ValueError:
                if verbose:
                    if verbose:
                        print("Model {} did not contain layer {}".format(old_models.index(m), lname))
        try:
            if weights:
                new_weights = np.mean(weights, axis=0) # try and hope for best?
                layer.set_weights(new_weights)
                if verbose:
                    print(lname, "Weights set")
            else:
                if verbose:
                    print(lname, "No weights found")
        except ValueError:
            if verbose:
                print("Averaging failed for layer", lname)
    return new_model

def tf_ses_ctl_reset(session_setter=None):
    if K.backend() == "tensorflow":
        clear_session()
        if session_setter is not None:
            set_session(session_setter())

