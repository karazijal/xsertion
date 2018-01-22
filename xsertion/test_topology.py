import unittest
import warnings
from xsertion.topology import *
from xsertion.layers import parse_model_description, cf
from xsertion.xcons import XSpotConFactory
from xsertion.test_layers import desc
from keras.layers import Input, MaxPooling2D, Convolution2D, Activation, merge, Dense, Flatten, Lambda
from keras.models import Model



class InputSuperLayerTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', DeprecationWarning)

    def test_input_superlayer_creation(self):
        inp = Input(shape=(32,32,3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        model = Model(inp, [lb1, lb2, lb3])
        layers, inps, outs = parse_model_description(desc(model))
        input_super_layer = InputSuperLayer(layers, inps, outs)

        for l1, l2 in zip(layers, input_super_layer.layers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_input_superlayer_repl(self):
        inp = Input(shape=(32,32,3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        lb1 = Activation('relu')(lb1)
        lb2 = Activation('relu')(lb2)
        lb3 = Activation('relu')(lb3)
        lb1 = Flatten()(lb1)
        lb2 = Flatten()(lb2)
        lb3 = Flatten()(lb3)
        model = Model(inp, [lb1, lb2, lb3])
        layers, inps, outs = parse_model_description(desc(model))
        input_super_layer = InputSuperLayer(layers, inps, outs)

        for l1, l2 in zip(layers, input_super_layer.layers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_input_superlayer_attaching(self):
        inp = Input(shape=(32,32,3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        lb1 = Activation('relu')(lb1)
        lb2 = Activation('relu')(lb2)
        lb3 = Activation('relu')(lb3)
        lb1 = Flatten()(lb1)
        lb2 = Flatten()(lb2)
        lb3 = Flatten()(lb3)
        model = Model(inp, [lb1, lb2, lb3])
        layers, inps, outs = parse_model_description(desc(model))
        input_super_layer = InputSuperLayer(layers, inps, outs)
        llayers, inp2, outs2 = input_super_layer.attach(None)
        for l1, l2 in zip(layers, llayers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

class TailSuperLayerTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', DeprecationWarning)

    def test_tailsuperlayer_creation(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        tsp = TailSuperLayer(layers[1:],(0,0), outs, 3)
        for l1, l2 in zip(layers[1:], tsp.layers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_tailsuperlayer_attching_single(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        tsp = TailSuperLayer(layers[1:], (0, 0), outs, 3)
        llayers, outs = tsp.attach([layers[0]])
        for l1, l2 in zip(layers[1:], llayers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_tailsuperlayer_attaching_multi_merge(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        tsp = TailSuperLayer(layers[1:], (0, 0), outs, 3)
        inp = Input(shape=(32, 32, 3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        inp_model = Model(inp, [lb1, lb2, lb3])
        inp_layers, inps2, outs2 = parse_model_description(desc(inp_model))

        llayers, outs = tsp.attach(inp_layers[-3:])
        self.assertEqual(llayers[0].get_name(), 'merge_tl')


    def test_tailsuperlayer_attaching_multi_con(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        tsp = TailSuperLayer(layers[1:], (0, 0), outs, 3)
        inp = Input(shape=(32, 32, 3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        inp_model = Model(inp, [lb1, lb2, lb3])
        inp_layers, inps2, outs2 = parse_model_description(desc(inp_model))

        llayers, outs = tsp.attach(inp_layers[-3:])
        for layer in llayers[0].get_inbound():
            self.assertTrue(layer in inp_layers[-3:])

class MainSuperLayerTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', DeprecationWarning)

    def test_mainsuperlayer_creation(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        main_super_layer = MainSuperLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                          list(layers[-1].get_inbound()), [], 3)
        for l1, l2 in zip(layers[1:-1], main_super_layer.layers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_mainsuperlayer_attaching(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        main_super_layer = MainSuperLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                          list(layers[-1].get_inbound()), [], 3)
        llayers, _, _ = main_super_layer.attach([(layers[0], 0, 0)])

        for l1, l2 in zip(layers[1:-1], llayers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_mainsuperlayer_renaming(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        main_super_layer = MainSuperLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                          list(layers[-1].get_inbound()), [], 3)
        llayers, _, _ = main_super_layer.attach([(layers[0], 0, 0)], rename='_test')

        for l1, l2 in zip(layers[1:-1], llayers):
            self.assertEqual(l1.get_name() + "_test", l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_mainsuperlayer_rescaling(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        main_super_layer = MainSuperLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                          list(layers[-1].get_inbound()), [], 3)
        for i in range(1, 101):
            rescale = float(i) / 100
            with self.subTest(msg="r={}".format(rescale), i=i):
                llayers, _, _ = main_super_layer.attach([(layers[0], 0, 0)], rename='_test', rescales=[rescale])

                for l1, l2 in zip(layers[1:-1], llayers):
                    self.assertEqual(l1.get_name() + "_test", l2.get_name())
                    self.assertEqual(l1.class_name, l2.class_name)
                    if l1.has_property('nb_filter'):
                        self.assertEqual(cf(l1.get_property('nb_filter')*rescale), l2.get_property('nb_filter'))


class MultiLayerTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', DeprecationWarning)
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = MaxPooling2D((1, 1))(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = MaxPooling2D((1, 1))(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        self.layers, _, _ = parse_model_description(desc(model))

    def test_multilayer_creation(self):
        layers = self.layers
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                          list(layers[-1].get_inbound()), [], 3)
        for l1, l2 in zip(layers[1:-1], main_super_layer.layers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_multilayer_no_xspots_ins(self):
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        layers, inps, outs = parse_model_description(desc(model))
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                          list(layers[-1].get_inbound()), [], 3)
        main_super_layer = main_super_layer.insert_x_spots(ModelBuilder.after_pooling)

        for l1, l2 in zip(layers[1:-1], main_super_layer.layers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_multilayer_xspots_ins(self):
        layers = self.layers
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                      list(layers[-1].get_inbound()), [], 3)
        main_super_layer = main_super_layer.insert_x_spots(ModelBuilder.after_pooling)

        self.assertEqual(main_super_layer.layers[3].get_name(), 'xspot_0')
        self.assertEqual(main_super_layer.layers[6].get_name(), 'xspot_1')

    def test_multilayer_attach_single(self):
        layers = self.layers
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                      list(layers[-1].get_inbound()), [], 3)

        llayers, xspots, tail_attach_point = main_super_layer.attach([(layers[0], 0, 0)])

        for l1, l2 in zip(layers[1:-1], llayers):
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertEqual(l1.class_name, l2.class_name)

    def test_multilayer_attach_single_xspots(self):
        layers = self.layers
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                      list(layers[-1].get_inbound()), [], 3)
        main_super_layer = main_super_layer.insert_x_spots(ModelBuilder.after_pooling)

        llayers, xspots, tail_attach_point = main_super_layer.attach([(layers[0], 0, 0)])

        self.assertEqual(main_super_layer.layers[3].get_name(), 'xspot_0')
        self.assertEqual(main_super_layer.layers[6].get_name(), 'xspot_1')

        self.assertEqual(len(xspots), 2)

    def test_multilayer_attach_multi_nores(self):
        layers = self.layers
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                      list(layers[-1].get_inbound()), [], 3)

        inp = Input(shape=(32, 32, 3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        inp_model = Model(inp, [lb1, lb2, lb3])
        inp_layers, inps2, outs2 = parse_model_description(desc(inp_model))

        llayers, xspots, tail_attach_point = main_super_layer.attach([(l, 0,0) for l in inp_layers[-3:]], rename=[0, 1, 2])
        self.assertEqual(len(llayers), 3*7)
        self.assertEqual(len(tail_attach_point), 3)

    def test_multilayer_attach_multi_res(self):
        layers = self.layers
        main_super_layer = MultiLayer(layers[1:-1], list(layers[0]._get_outbound()),
                                      list(layers[-1].get_inbound()), [], 3)

        inp = Input(shape=(32, 32, 3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1))(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1))(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1))(inp)
        inp_model = Model(inp, [lb1, lb2, lb3])
        inp_layers, inps2, outs2 = parse_model_description(desc(inp_model))

        llayers, xspots, tail_attach_point = main_super_layer.attach([(l, 0,0) for l in inp_layers[-3:]],
                                                                     [.5, .25, .25], rename=[0, 1, 2])
        self.assertEqual(len(llayers), 3*7)
        self.assertEqual(len(tail_attach_point), 3)

        self.assertEqual(llayers[0].get_property('nb_filter'), 16)
        self.assertEqual(llayers[-1].get_property('nb_filter'), 8)


class ModelBuilderTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', DeprecationWarning)
        inp = Input(shape=(32, 32, 3))
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(inp)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = MaxPooling2D((1, 1))(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = MaxPooling2D((1, 1))(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf', name='scale_test')(x)
        x = Convolution2D(32, 3, 3, dim_ordering='tf')(x)
        model = Model(inp, x)
        self.layers, _, mouts = parse_model_description(desc(model))
        inp = Input(shape=(32, 32, 3))
        lb1 = Lambda(lambda x: x[:, :, :, 0:1], output_shape=(32, 32, 1), name='test1')(inp)
        lb2 = Lambda(lambda x: x[:, :, :, 1:2], output_shape=(32, 32, 1), name='test2')(inp)
        lb3 = Lambda(lambda x: x[:, :, :, 2:3], output_shape=(32, 32, 1), name='test3')(inp)
        model = Model(inp, [lb1, lb2, lb3])
        layers, inps, outs = parse_model_description(desc(model))
        self.isl = InputSuperLayer(layers, inps, outs)
        self.msl = MultiLayer(self.layers[1:-1], list(self.layers[0]._get_outbound()),
                                      list(self.layers[-1].get_inbound()), [], 3)

        self.tsl = TailSuperLayer([self.layers[-1]], (0,0), mouts, 3)

        self.d = {
            "keras_version": "1.2.2",
            "class_name": "Model",
            "config" : None
        }

    def test_creation(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, dict())

        self.assertEqual(mbl.num_of_lanes, 3)
        self.assertTrue('test1' in mbl.input_layersnames)
        self.assertTrue('test2' in mbl.input_layersnames)
        self.assertTrue('test3' in mbl.input_layersnames)

    def test_input_ordering(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, dict())

        self.assertEqual(mbl.num_of_lanes, 3)
        self.assertListEqual(['test1', 'test2', 'test3'], mbl.input_layer_lids)

    def test_build_no_inp(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, dict())

        with self.assertRaises(AssertionError):
            mbl.build_single_lane('lambda_1')

    def test_build_naming(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, self.d)

        model = mbl.build_single_lane('test1')
        self.assertEqual(model.name, 'test1_superlayer_model')

    def test_build_custom_naming(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, self.d)

        model = mbl.build_single_lane('test1', modelname="testmodel")
        self.assertEqual(model.name, "testmodel")

    def test_build_multilane_rescale(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, self.d)

        model = mbl.build_multi_lane(['test1', 'test2', 'test3'], [.5, .25, .25])
        self.assertEqual(model.get_layer(name='scale_test_0').get_config()['nb_filter'], 16)
        self.assertEqual(model.get_layer(name='scale_test_1').get_config()['nb_filter'], 8)
        self.assertEqual(model.get_layer(name='scale_test_2').get_config()['nb_filter'], 8)

    def test_build_multilane_rescale_xcon(self):
        mbl = ModelBuilder(self.isl, self.msl, self.tsl, self.d)
        mbl.insert_x_spots(ModelBuilder.after_pooling)
        p = 1.0
        filts = {
            0: {0: p, 1: p, 2: p},
            1: {0: p, 1: p, 2: p},
            2: {0: p, 1: p, 2: p},
        }
        con_fact = XSpotConFactory(bias=False, activation='relu')
        model = mbl.build_multi_lane(['test1', 'test2', 'test3'], [.5, .25, .25], params=(filts, True, con_fact))
        self.assertEqual(model.get_layer(name='scale_test_0').get_config()['nb_filter'], 16)
        self.assertEqual(model.get_layer(name='scale_test_1').get_config()['nb_filter'], 8)
        self.assertEqual(model.get_layer(name='scale_test_2').get_config()['nb_filter'], 8)
        self.assertEqual(model.get_layer(name='xc_0_2to1').get_config()['nb_filter'], 8)
        self.assertEqual(model.get_layer(name='xc_1_0to2').get_config()['bias'], False)

if __name__=="__main__":
    unittest.main()