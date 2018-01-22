from xsertion import XCNNBuilder
from data import get_cifar, val_split

from keras.models import Model
from keras.layers import Lambda, Input
from keras.callbacks import EarlyStopping

from kerasnet import get_kerasnet  # Specify the model in a separate file.

# Xsertion will log heavily on DEBUG level. Enable it if it is desired to see details logs.
import logging
logging.basicConfig(level=logging.DEBUG, filename='example.log')  # Will put logs into example.log in cwd

# A little utility to determine dimension ordering.

import keras.backend as K
input_shape = (3, 32, 32)
pr_axis = 1  # This is the "chanel axis" or here principal axis. We will seperate (and put back) modalities along this.
if K.image_dim_ordering() == 'tf':
    input_shape = (32, 32, 3)
    pr_axis = 3


# Prepare the data

p = 1.  # Use full dataset (i.e. retain 100% per-class examples
use_c10 = True  # Let's use CIFAR-10, set to false for 100

# get_cifar utility will perform linear transform into YUV space
nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=p, append_test=False, use_c10=use_c10)
X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)  # split training data 80/20 for Xsertion to use internally


blueprint_model = get_kerasnet(nb_classes, input_shape, pr_axis)  # get KerasNet appropriate for keras/tf/th config


# Xsertion start

# Xsertion understands specification of modalities as "input model" where outputs are modalities.
# This model is not touched by Xsertion, only its ouputs are used, feel free to go crazy here.
# However you define the input to this model, its inputs has to match the input data that's passed to Xsertion (Duh).
inp = Input(shape=input_shape)
if K.image_dim_ordering() == 'tf':
    lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
    lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
    lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
else:
    lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
    lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
    lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)
input_model = Model(input=inp, output=[lb1, lb2, lb3])

alpha = 1  # alpha hyper
beta = 2  # beta hyper
builder = XCNNBuilder(blueprint_model, input_model, alpha=alpha, beta=beta)
# create an XCNNBuilder, Xsertion will go crazy tearing apart blueprint_model and tranformaing/analysing it for internal
# use. Not considering training/data processing this is the most time consuming step.

builder.set_xspot_strategy(strategy='after_pooling')  # set after_pooling heuristic to be used for placing connections
# Alternativelly, go with resnet for branching/merging topologies. Or just specify layernames if you have exact idea
# where you want connections.

builder.set_xcons(use_bn=False, activation='relu', bias=False)  # Pass whatever keywords you could pass to Convolution,
# to customise the connection/adjust it for your model. Here just turning off bias as an example).
# Two special arguments are available, use_bn, which will insert approapriate BatchNormalistion along the connection.
# Alternativelly, use model_function to specify a function that returns a connection model (similar to input model)
# to customise connections. It's signiture is (nb_filter, inbount_name=None, name=None)->Model. nb_filter will be set
# to what Xsertion thinks number of filters along this connections should be.

builder.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
# Anything that would go into vanilla compile.

builder.fit(X_t, Y_t, batch_size=32, nb_epoch=80, validation_data=(X_v, Y_v), verbose=2,
            callbacks=[EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=15, verbose=2)])
# Anything that would go into fit. Note this does not perform any work yet.

# Kick off Xsertion

xcnn_model = builder.build()  # This is a good time to go grab a cup of tea.
builder.print_report()  # Get some information about how building went.

# Need to use different hypers?
builder.alpha = 2
xcnn_model = builder.build()  # It will use cached measures to quickly give another model.
# Note xcnn_model is not trained.

# Feeling like it could be better?
# Not sure if certain pairs of modalities are even sensible?
# Try
xcnn2_model = builder.build_scaled_double_xcon()  # This will use another round of measures between pairs of modilites.
# Though it should be noted, we did not observe much gain when using this methodology. Instead consider,
# Note xcnn2_model is not trained.

xcnn_iter_model = builder.build_iter(.1, Nasterov=True, iter_max=15, rep=1)  # This will commence combined learning
# to produce model. First parameter is initial learning rate. Nasterov controls whether nasterov accelation is applied
# to adaptive momentum. Building process is too noisy and steps seem to be not sensible. Try adjusting learning rate.
# If push comes to shove, increase reps (rep).

# This produces trained model. But only on 80% training data that was supplied. For fair comparison, it shoudl finish
# training.

# Train.

