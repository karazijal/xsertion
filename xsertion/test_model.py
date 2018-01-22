import os
import unittest
import warnings

import keras.backend as K

import xsertion.model
from xsertion.model import XCNNBuilder, InvalidModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence Tensorflow warnings

class NonModelParsingTestCase(unittest.TestCase):

    def setUp(self):
        from keras.layers import Input
        from keras.models import Model
        inp = Input(shape=(3,32,32))
        inp2 = Input(shape=(3,32,32))
        self.model = Model(input=inp, output=inp)
        self.input_model = Model(inp2, inp2)

    def test_nonemodel(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(None, self.input_model)

    def test_noneinputmodel(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(self.model, None)

    def test_non_none_model(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(dict(model=self.input_model), self.input_model)

    def test_non_none_inputmodel(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(self.model, dict())

class InputModelParsingTestCase(unittest.TestCase):

    def setUp(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            from keras.layers import Input, Lambda, BatchNormalization, Convolution2D, Activation, Flatten, Dense, MaxPooling2D
            from keras.models import Sequential, Model
            input_shape = (3, 32, 32)
            pr_axis = 1
            if K.image_dim_ordering() == 'tf':
                input_shape = (32, 32, 3)
                pr_axis = 3
            inp = Input(shape=input_shape)
            if K.image_dim_ordering() == 'tf':
                lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
                lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
                lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
                il_ip = Input(shape=(28, 28, 1))
                il_ip2 = Input(shape=(32,32, 2))
            else:
                lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
                lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
                lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)
                il_ip = Input(shape=(1,28,28))
                il_ip2 = Input(shape=(2,32, 32))
            input_model = Model(input=inp, output=[lb1, lb2, lb3])
            self.legal_input_model = input_model
            self.pr_axis = pr_axis
            self.il_axis = 3 if pr_axis==1 else 1
            self.il_inp_m = Model(input=il_ip, output=il_ip)
            self.il_inp_m2 = Model(input=il_ip2, output=il_ip2)
            model = Sequential()
            model.add(BatchNormalization(axis=pr_axis, input_shape=input_shape))
            model.add(Convolution2D(64, 3, 3, border_mode='same'))
            model.add(Activation('relu'))
            model.add(Convolution2D(64, 3, 3, border_mode='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(10))
            model.add(Activation('softmax'))

            self.legal_model = model

    def test_inputaxis_bounds(self):
        with self.assertRaises(ValueError):
            builder = XCNNBuilder(self.legal_model, self.legal_input_model, input_split_axis=5)

    def test_inputaxis_illegal(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(self.legal_model, self.legal_input_model, input_split_axis=self.il_axis)

    def test_inputdim_mismatch(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(self.legal_model, self.il_inp_m, input_split_axis=self.pr_axis)

    def test_inputaxis_mismatch(self):
        with self.assertRaises(InvalidModel):
            builder = XCNNBuilder(self.legal_model, self.il_inp_m2, input_split_axis=self.pr_axis)

class CompleteModelTestCase(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            from keras.layers import Input, Lambda, BatchNormalization, Convolution2D, Activation, Flatten, Dense, \
                MaxPooling2D
            from keras.models import Sequential, Model
            import keras.backend as K
            input_shape = (3, 32, 32)
            pr_axis = 1
            if K.image_dim_ordering() == 'tf':
                input_shape = (32, 32, 3)
                pr_axis = 3
            inp = Input(shape=input_shape)
            if K.image_dim_ordering() == 'tf':
                lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
                lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
                lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
            else:
                lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
                lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
                lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)
            self.input_model = Model(input=inp, output=[lb1, lb2, lb3])
            self.pr_axis = pr_axis
            model = Sequential()
            model.add(BatchNormalization(axis=pr_axis, input_shape=input_shape, name="Main_start"))
            model.add(Convolution2D(64, 3, 3, border_mode='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), name="Mah_pool"))
            model.add(Convolution2D(64, 3, 3, border_mode='same'))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(10))
            model.add(Activation('softmax', name='myout'))
            self.model = model

    def test_axis_discovery(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual(self.pr_axis, builder.split_axis)

    def test_axis_discovery_tf_ord(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            from keras.layers import Input, Lambda, BatchNormalization, Convolution2D, Activation, Flatten, Dense, \
                MaxPooling2D
            from keras.models import Sequential, Model
            inp = Input(shape=(32, 32, 3))
            lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
            lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
            lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
            input_model = Model(input=inp, output=[lb1, lb2, lb3])
            ch_axis = 3
            dim_ord = 'tf'
            model = Sequential()
            model.add(BatchNormalization(axis=ch_axis, input_shape=(32, 32, 3), name="Main_start"))
            model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering=dim_ord))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), name="Mah_pool", dim_ordering=dim_ord))
            model.add(Convolution2D(64, 3, 3, border_mode='same',dim_ordering=dim_ord))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(10))
            model.add(Activation('softmax', name='myout'))

            builder = XCNNBuilder(model, input_model)
            self.assertEqual(3, builder.split_axis)

    def test_axis_discovery_th_ord(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            from keras.layers import Input, Lambda, BatchNormalization, Convolution2D, Activation, Flatten, Dense, \
                MaxPooling2D
            from keras.models import Sequential, Model
            inp = Input(shape=(3, 32, 32))
            lb1 = Lambda(lambda x: x[:, 0:1, :, :], output_shape=(1, 32, 32))(inp)
            lb2 = Lambda(lambda x: x[:, 1:2, :, :], output_shape=(1, 32, 32))(inp)
            lb3 = Lambda(lambda x: x[:, 2:3, :, :], output_shape=(1, 32, 32))(inp)
            input_model = Model(input=inp, output=[lb1, lb2, lb3])
            ch_axis = 1
            dim_ord = 'th'
            model = Sequential()
            model.add(BatchNormalization(axis=ch_axis, input_shape=(3, 32, 32), name="Main_start"))
            model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering=dim_ord))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), name="Mah_pool", dim_ordering=dim_ord))
            model.add(Convolution2D(64, 3, 3, border_mode='same',dim_ordering=dim_ord))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(10))
            model.add(Activation('softmax', name='myout'))

            builder = XCNNBuilder(model, input_model)
            self.assertEqual(1, builder.split_axis)

    def test_superlayer_count(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual(3, builder.number_of_lanes)

    def test_input_cutoff0(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual(builder.model_factory.main_superlayer.layers[0].class_name, "BatchNormalization")

    def test_input_cutoff1(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual(builder.model_factory.main_superlayer.layers[0].get_name(), "Main_start")

    def test_output_cutoff0(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertTupleEqual(builder.model_factory.tail_superlayer.inputs_ps, (0,0))

    def test_output_cutoff1(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual(builder.model_factory.tail_superlayer.layers[0].class_name, 'Flatten')

    def test_output_xspot_count(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual(1, len(builder.model_factory.main_superlayer.xspots))

    def test_output_xspot_point(self):
        builder = XCNNBuilder(self.model, self.input_model)
        self.assertEqual("Mah_pool", list(builder.model_factory.main_superlayer.xspots[0].get_inbound())[0].get_name())

    def test_compile_optimizerNone(self):
        builder = XCNNBuilder(self.model, self.input_model)
        with self.assertRaises(ValueError):
            builder.compile(None, "mse")

    @unittest.skipUnless(K.backend() == "tensorflow", "Not using tensorflow")
    def test_compile_optimizer_tf(self):
        import tensorflow as tf
        builder = XCNNBuilder(self.model, self.input_model)
        adam = tf.train.AdamOptimizer()
        builder.compile(adam, 'mse')
        self.assertIsInstance(builder.optimizer, tf.train.Optimizer)

    def test_compile_lossNone(self):
        builder = XCNNBuilder(self.model, self.input_model)
        with self.assertRaises(ValueError):
            builder.compile('adam', None)

    def test_compile_lossListFail0(self):
        builder = XCNNBuilder(self.model, self.input_model)
        with self.assertRaises(ValueError):
            builder.compile('adam', [])

    def test_compile_lossListFail2(self):
        builder = XCNNBuilder(self.model, self.input_model)
        with self.assertRaises(ValueError):
            builder.compile('adam', ['mse', 'hinge'])

    def test_compile_lossListPass(self):
        builder = XCNNBuilder(self.model, self.input_model)
        builder.compile('adam', ['mse'])
        self.assertEqual(builder.loss[0], 'mse')

    def test_compile_lossDictFail(self):
        builder = XCNNBuilder(self.model, self.input_model)
        with self.assertRaises(ValueError):
            builder.compile('adam', dict())

    def test_compile_lossDictPass(self):
        builder = XCNNBuilder(self.model, self.input_model)
        builder.compile('adam', dict(myout='mse'))
        self.assertEqual('mse', builder.loss['myout'])

    def test_not_compiled(self):
        builder = XCNNBuilder(self.model, self.input_model)
        with self.assertRaises(ValueError):
            m1 = builder.build()

    def test_not_fit(self):
        builder = XCNNBuilder(self.model, self.input_model)
        builder.compile('adam', 'mse')
        with self.assertRaises(ValueError):
            m1 = builder.build()

    def test_pipeline_fit_notval(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            from data import get_cifar, val_split
            nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=0.01, append_test=False,
                                                                               use_c10=True)
            X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)

            builder = XCNNBuilder(self.model, self.input_model)

            builder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            with self.assertRaises(ValueError):
                builder.fit(X_t,Y_t,nb_epoch=1, verbose=0)

    def test_pipeline_first_fit(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            from data import get_cifar, val_split

            nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=0.01, append_test=False,
                                                                               use_c10=True)
            X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)

            import keras
            builder = XCNNBuilder(self.model, self.input_model)

            builder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            builder.fit(X_t, Y_t, nb_epoch=1, validation_data=(X_v, Y_v), verbose=0)
            m1 = None
            m1 = builder.build()
            self.assertNotEqual(m1, None)
            self.assertIsInstance(m1, keras.models.Model)
            self.assertTrue(builder.analysed_single_superlayer_accuracy)
            self.assertTrue(builder.analysed_base_scaling)

    def test_pipeline(self):
        from data import get_cifar, val_split
        nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=0.01, append_test=False,
                                                                               use_c10=True)
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
            builder = XCNNBuilder(self.model, self.input_model)

            builder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            builder.fit(X_t,Y_t,nb_epoch=1, validation_data=(X_v, Y_v), verbose=0)
            m2 = None
            m1 = builder.build()
            m2 = builder.build_iter(0.5, iter_max=0, rep=1)
        self.assertNotEqual(m2, None)

    def test_pipeline_cons_inserted(self):
        from data import get_cifar, val_split
        nb_classes, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_cifar(p=0.01, append_test=False,
                                                                               use_c10=True)
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', DeprecationWarning)
            X_t, Y_t, X_v, Y_v = val_split(0.2, X_train, Y_train)
            builder = XCNNBuilder(self.model, self.input_model)

            builder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            builder.fit(X_t,Y_t,nb_epoch=1, validation_data=(X_v, Y_v), verbose=0)
            m2 = None
            m1 = builder.build()
        self.assertEqual(m1.get_layer(name='xc_0_2to1').get_config()['bias'], True)
        self.assertEqual(m1.get_layer(name='xmerge_0_1_mrg').get_config()['mode'], 'concat')



class SuplFuncTestCase(unittest.TestCase):

    def test_in_pairs2(self):
        self.assertListEqual([(0,1)], xsertion.model.in_pairs(2))

    def test_in_pairs5(self):
        self.assertListEqual([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)], xsertion.model.in_pairs(5))

    def test_secondfiltfunc05(self):
        pair_accs = [(l1,l2,.5) for l1,l2 in [(0,2),(2,1),(1,0)]]
        base_accs = [.5,.5,.5]
        priors = [1, 1, 1]
        self.assertEqual(.5, xsertion.model.second_filter_func(0, 1, priors, base_accs, pair_accs))

    def test_secondfiltfunc01(self):
        pair_accs = [(l1,l2,.1) for l1,l2 in [(0,2),(2,1),(1,0)]]
        base_accs = [.1,.1,.1]
        priors = [1, 1, 1]
        self.assertEqual(.5, xsertion.model.second_filter_func(1, 0, priors, base_accs, pair_accs))

    def test_secondfiltfunc2(self):
        pair_accs = [(l1,l2,.3) for l1,l2 in [(0,2),(2,1),(1,0)]]
        base_accs = [.3,.3,.3]
        priors = [1, 1, 1]
        self.assertEqual(.5, xsertion.model.second_filter_func(2, 1, priors, base_accs, pair_accs))






if __name__=="__main__":
    unittest.main()