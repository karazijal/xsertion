from xsertion.keras_interface_util import keras_conv2d_config, get_Model, unwrap_sequential
import xsertion.layers
import inspect
import random
import json

class ConnectionFactory():
    def __call__(self, inbound_name, name, nb_filters):
        raise NotImplementedError

class XSpotConFactory(ConnectionFactory):
    """simple single convolution connection Factory"""
    def __init__(self, **kwargs):
        super(XSpotConFactory, self).__init__()
        self.args = [1, 1] # the one-by-one convolutions
        self.kwargs = kwargs

    def __call__(self, inbound_name, name, nb_filter):
        config = keras_conv2d_config(nb_filter, *self.args, **self.kwargs)
        config['name'] = name
        inb = [[[inbound_name, 0, 0]]]
        return [dict(name=name,
                           class_name="Convolution2D",
                           config=config,
                           inbound_nodes=inb)]

class FunctionalModelFactory(ConnectionFactory):
    def __init__(self, model_function):
        super(FunctionalModelFactory, self).__init__()
        if inspect.isfunction(model_function):
            spec= inspect.getfullargspec(model_function)
            if len(spec[0])==1:
                self.model_function = model_function
                if spec[2]:
                    self.kwargs = True
                else:
                    self.kwargs = False
            else:
                raise ValueError(
                    "{} is not a valid single argument function as model_func(nb_filter)".format(model_function))
        else:
            raise ValueError(
                "{} is not a valid single argument function as model_func(nb_filter)".format(model_function))
        if self.kwargs:
            test_model = self.model_function(random.randint(1, 3000), inbound_name='test_inbound', name='test')
        else:
            test_model = self.model_function(random.randint(1, 3000))
            # Try to create a connection
        if test_model and isinstance(test_model, get_Model()):
            test_model = unwrap_sequential(test_model)
        else:
            raise ValueError(
                "{} did not return a valid keras model".format(model_function))

    def __call__(self, inbound_name, name, nb_filters):
        if self.kwargs:
            model = unwrap_sequential(self.model_function(nb_filters, inbound_name=inbound_name, name=name))
        else:
            model = unwrap_sequential(self.model_function(nb_filters))
        layers, inputs, outputs = xsertion.layers.parse_model_description(json.loads(model.to_json())['config'])
        assert len(inputs) == 1, "Model does not have a single input"
        assert len(outputs) == 1, "Model does not have a single output"
        layers_no_inp = layers[1:]
        input_layer = layers[0]
        # print(inbound_name, input_layer.get_name(), input_layer.class_name)
        input_layer.set_name(inbound_name) # rename input_layer to inbound_name so that all internal connection are

        # consistent
        for layer in layers_no_inp:
            layer.set_name(name+'_'+layer.get_name()) # rename all of the layers
        outl, _, _ = outputs[0]
        outl.set_name(name) # rename the output layer to name
        model_desc = xsertion.layers.render(inputs, layers, outputs) # render the layers back to descriptions
        # including proxy input layer to maintain a legal model
        # use descriptions from this model
        ldescs = [ldesc for ldesc in model_desc['layers'] if ldesc['name'] != inbound_name] # add everything except the
        # the proxy input layer
        return ldescs



    # Above would produce behaviour similar to XSpotConFactory with this:
    #
    # def my_model_function(nb_filter):
    #     input_shape = (32, 32, 3) this doesn't matter so make it legal
    #     model = Sequential() # could use functional Model API as well
    #     model.add(Convolution2d(nb_filter, 1, 1, input_shape=input_shape))
    #     model.add(Dropout(.2)) # this adds dropout after conv. to show versatility of the function
    #     return model
    #
    # then pass this function to the XCNNBuilder to use custom connection
    # Naturally if output shape does not match where connection is being placed then this will fail

