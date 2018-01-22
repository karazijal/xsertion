import unittest
from xsertion.layers import *
from keras.layers import Input, MaxPooling2D, Convolution2D, Activation, merge, Dense, Flatten
from keras.models import Model

import json


def desc(model : Model):
    base_model_disc = json.loads(model.to_json())
    return base_model_disc['config']

def topo_check(layerlist):
    ind = {layer: i for i,layer in enumerate(layerlist)}
    for i, layer in enumerate(layerlist):
        if any(ind[l] > i for l in layer.get_inbound()): # all incoming must be before i
            return False
        if any(ind[l] < i for l in layer._get_outbound()): # all outgoing must be after i
            return False
    return True

class ParsingTestCase(unittest.TestCase):

    def test_layer_con_and_config(self):
        it = Input(shape=(3, 32, 32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu', name='TestLayer')
        config = json.loads(json.dumps(c1.get_config())) # transform all tuples to lists
        model=Model(input=it, output=c1(it))
        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        self.assertDictEqual(config, layers[1].config)
        self.assertEqual(layers[0], list(layers[1].get_inbound())[0])
        self.assertEqual(layers[1], list(layers[0]._get_outbound())[0])
        self.assertTrue(topo_check(layers))

    def test_linear_model(self):
        it = Input(shape=(3,32,32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu')(it)
        a1 = Flatten()(c1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)

        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        self.assertEqual(4, len(layers))
        self.assertEqual(1, len(model_inputs))
        self.assertEqual(1, len(model_outputs))

        self.assertEqual("TestInput", layers[0].get_name())
        self.assertEqual("TestOutput", layers[-1].get_name())
        self.assertTrue(topo_check(layers))

    def test_branching_model(self):
        it = Input(shape=(3,32,32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu')(it)
        c2 = Convolution2D(32, 3, 3, activation='relu')(it)
        c3 = Convolution2D(32, 3, 3, activation='relu')(it)
        m1 = merge([c1, c2, c3], mode='sum')
        a1 = Flatten()(m1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)

        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        self.assertEqual(7, len(layers))
        self.assertEqual(1, len(model_inputs))
        self.assertEqual(1, len(model_outputs))

        self.assertEqual("TestInput", layers[0].get_name())
        self.assertEqual("TestOutput", layers[-1].get_name())

        self.assertTrue(topo_check(layers))

    def test_branching_multistage_model(self):
        it = Input(shape=(3,32,32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu')(it)
        b1 = Activation('relu')(c1)
        c2 = Convolution2D(32, 3, 3, activation='relu')(it)
        b2 = Activation('relu')(c2)
        c3 = Convolution2D(32, 3, 3, activation='relu')(it)
        b3 = Activation('relu')(c3)
        m1 = merge([b1, b2, b3], mode='sum')
        a1 = Flatten()(m1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)

        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        self.assertEqual(10, len(layers))
        self.assertEqual(1, len(model_inputs))
        self.assertEqual(1, len(model_outputs))

        self.assertEqual("TestInput", layers[0].get_name())
        self.assertEqual("TestOutput", layers[-1].get_name())

        self.assertTrue(topo_check(layers))

    def test_skip_connnection(self):
        it = Input(shape=(3,32,32), name='TestInput')
        c1 = Convolution2D(3, 3, 3, border_mode='same', dim_ordering='th')(it) #dim_ordering to force match on inputshape
        b1 = Activation('relu')(c1)
        m1 = merge([b1, it], mode='sum')
        a1 = Flatten()(m1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)

        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        self.assertEqual(6, len(layers))
        self.assertEqual(1, len(model_inputs))
        self.assertEqual(1, len(model_outputs))

        self.assertEqual("TestInput", layers[0].get_name())
        self.assertEqual("TestOutput", layers[-1].get_name())

        self.assertTrue(topo_check(layers))

    def test_complex_skip(self):
        l1 = Input((3,32,32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l2)
        l5 = merge([l2,l3], name='5')
        l6 = merge([l1,l4], name='6')
        l7 = Activation('relu', name='7')(l5)
        l8 = merge([l6,l7], name='8')
        model = Model(input=l1, output=l8)

        layers, model_inputs, model_outputs = parse_model_description(desc(model))


        self.assertEqual('1', layers[0].get_name())
        self.assertTrue(topo_check(layers))
        self.assertListEqual(['1','2','3','4','5','7','6','8'], [l.get_name() for l in layers])

class ReplicationTestCase(unittest.TestCase):

    def test_replication_layer_properties(self):
        #use keras layers to quickly fill the list
        l1 = Input((3, 32, 32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l2)
        l5 = merge([l2, l3], name='5')
        l6 = merge([l1, l4], name='6')
        l7 = Activation('relu', name='7')(l5)
        l8 = merge([l6, l7], name='8')
        model = Model(input=l1, output=l8)
        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        repl_list = replicate_layerlist(layers)
        for l1, l2 in zip(layers, repl_list):
            self.assertEqual(l1.class_name, l2.class_name)
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertDictEqual(l1.config, l2.config)

    def test_replication_layer_connections(self):
        # use keras layers to quickly fill the list
        l1 = Input((3, 32, 32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l2)
        l5 = merge([l2, l3], name='5')
        l6 = merge([l1, l4], name='6')
        l7 = Activation('relu', name='7')(l5)
        l8 = merge([l6, l7], name='8')
        model = Model(input=l1, output=l8)
        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        def assertSameLayer(l1, l2):
            self.assertEqual(l1.class_name, l2.class_name)
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertDictEqual(l1.config, l2.config)

        repl_list = replicate_layerlist(layers)
        for l1, l2 in zip(layers, repl_list):
            # build matching inbound lists
            for il in l1.get_inbound():
                for il2 in l2.get_inbound():
                    if layers.index(il) == repl_list.index(il2):
                        assertSameLayer(il, il2)

    def test_replication_layer_con_consitency(self):
        # use keras layers to quickly fill the list
        l1 = Input((3, 32, 32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l2)
        l5 = merge([l2, l3], name='5')
        l6 = merge([l1, l4], name='6')
        l7 = Activation('relu', name='7')(l5)
        l8 = merge([l6, l7], name='8')
        model = Model(input=l1, output=l8)
        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        llayers = layers[3:] # only take 4, 5, 6, 7, 8
        repl_layers = replicate_layerlist(llayers)

        self.assertEqual(0, len(repl_layers[0].get_inbound())) # no connections for 4 been inserted
        self.assertEqual(0, len(repl_layers[1].get_inbound())) # no connections for 5 has been inserted
        self.assertEqual(1, len(repl_layers[3].get_inbound())) # only connection to 4 has been included for 6

        def assertSameLayer(l1, l2):
            self.assertEqual(l1.class_name, l2.class_name)
            self.assertEqual(l1.get_name(), l2.get_name())
            self.assertDictEqual(l1.config, l2.config)
        assertSameLayer(list(repl_layers[3].get_inbound())[0], layers[3])

    def test_xspot_replication(self):
        # use keras layers to quickly fill the list
        l1 = Input((3, 32, 32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l2)
        l5 = merge([l2, l3], name='5')
        l6 = merge([l1, l4], name='6')
        l7 = Activation('relu', name='7')(l5)
        l8 = merge([l6, l7], name='8')
        model = Model(input=l1, output=l8)
        layers, model_inputs, model_outputs = parse_model_description(desc(model))

        xspot = XLayerBP.insertXspot(layers[4], 16)
        layers.insert(5, xspot)

        repl_layers = replicate_layerlist(layers)
        self.assertEqual('XSpot', repl_layers[5].class_name)
        self.assertEqual(4, repl_layers.index(list(repl_layers[5].get_inbound())[0]))

class XLayerTestCase(unittest.TestCase):

    def test_xspot_insertion_simple(self):
        l1 = Input((3, 32, 32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l1)
        l5 = merge([l2, l3, l4], name='5')
        model = Model(l1, l5)
        layers, model_inputs, model_outputs = parse_model_description(desc(model))
        xspot = XLayerBP.insertXspot(layers[2], 16) # insert after 3

        # check that l3 is now only connected to xspot
        self.assertEqual(list(layers[2]._get_outbound())[0], xspot)

    def test_xspot_insertion_branching(self):
        l1 = Input((3, 32, 32), name='1')
        l2 = Activation('relu', name='2')(l1)
        l3 = Activation('relu', name='3')(l1)
        l4 = Activation('relu', name='4')(l1)
        l5 = merge([l2, l3, l4], name='5')
        model = Model(l1, l5)
        layers, model_inputs, model_outputs = parse_model_description(desc(model))
        xspot = XLayerBP.insertXspot(layers[2], 16)  # insert after 3

        # check that l5 is now connected to l2, l4, and xspot
        b = layers[-1].get_inbound()
        self.assertTrue(xspot in b)
        self.assertTrue(layers[1] in b)
        self.assertTrue(layers[3] in b)

class RenderingTestCase(unittest.TestCase):
    def test_parse_render(self):
        it = Input(shape=(3, 32, 32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b1 = Activation('relu')(c1)
        c2 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b2 = Activation('relu')(c2)
        c3 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b3 = Activation('relu')(c3)
        m1 = merge([b1, b2, b3], mode='sum')
        a1 = Flatten()(m1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)
        mdescs = desc(model)
        layers, model_inputs, model_outputs = parse_model_description(mdescs)
        rend_descs = render(model_inputs, layers, model_outputs)
        for inp in mdescs['input_layers']:
            nm, p1, p2 = inp[0], inp[1], inp[2]
            for inp2 in rend_descs['input_layers']:
                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                if nm2 == nm and p12==p1 and p22==p2:
                    self.assertTrue(True)
                    break
            else:
                self.assertTrue(False)

        for inp in mdescs['output_layers']:
            nm, p1, p2 = inp[0], inp[1], inp[2]
            for inp2 in rend_descs['output_layers']:
                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                if nm2 == nm and p12==p1 and p22==p2:
                    self.assertTrue(True)
                    break
            else:
                self.assertTrue(False)

        for layer in mdescs['layers']:
            for llayer in rend_descs['layers']:
                if layer['name'] == llayer['name']:
                    self.assertDictEqual(layer['config'], llayer['config'])
                    if len(layer['inbound_nodes']) > 0:
                        for inp in layer['inbound_nodes'][0]:
                            nm, p1, p2 = inp[0], inp[1], inp[2]
                            for inp2 in llayer['inbound_nodes'][0]:
                                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                                if nm2 == nm and p12 == p1 and p22 == p2:
                                    self.assertTrue(True)
                                    break
                            else:
                                self.assertTrue(False)

                    break
            else:
                self.assertTrue(False)

    def test_render_xsport_skip(self):
        it = Input(shape=(3, 32, 32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b1 = Activation('relu')(c1)
        c2 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b2 = Activation('relu')(c2)
        c3 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b3 = Activation('relu')(c3)
        m1 = merge([b1, b2, b3], mode='sum')
        a1 = Flatten()(m1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)
        mdescs = desc(model)
        layers, model_inputs, model_outputs = parse_model_description(mdescs)
        xspots = []
        for i,layer in enumerate(layers):
            if layer.class_name == "Convolution2D":
                xspot = XLayerBP.insertXspot(layer, 32)
                xspots.append((i, xspot))
        for c, (i, xs) in enumerate(xspots):
            layers.insert(i+1+c, xs)

        rend_descs = render(model_inputs, layers, model_outputs)
        for inp in mdescs['input_layers']:
            nm, p1, p2 = inp[0], inp[1], inp[2]
            for inp2 in rend_descs['input_layers']:
                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                if nm2 == nm and p12==p1 and p22==p2:
                    self.assertTrue(True)
                    break
            else:
                self.assertTrue(False)

        for inp in mdescs['output_layers']:
            nm, p1, p2 = inp[0], inp[1], inp[2]
            for inp2 in rend_descs['output_layers']:
                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                if nm2 == nm and p12==p1 and p22==p2:
                    self.assertTrue(True)
                    break
            else:
                self.assertTrue(False)

        for layer in mdescs['layers']:
            for llayer in rend_descs['layers']:
                if layer['name'] == llayer['name']:
                    self.assertDictEqual(layer['config'], llayer['config'])
                    if len(layer['inbound_nodes']) > 0:
                        for inp in layer['inbound_nodes'][0]:
                            nm, p1, p2 = inp[0], inp[1], inp[2]
                            for inp2 in llayer['inbound_nodes'][0]:
                                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                                if nm2 == nm and p12 == p1 and p22 == p2:
                                    self.assertTrue(True)
                                    break
                            else:
                                self.assertTrue(False)

                    break
            else:
                self.assertTrue(False)

    def test_render_xsport_skip_merge(self):
        it = Input(shape=(3, 32, 32), name='TestInput')
        c1 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b1 = Activation('relu')(c1)
        c2 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b2 = Activation('relu')(c2)
        c3 = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th')(it)
        b3 = Activation('relu')(c3)
        m1 = merge([b1, b2, b3], mode='sum')
        a1 = Flatten()(m1)
        d1 = Dense(10, activation='softmax', name="TestOutput")(a1)
        model = Model(input=it, output=d1)
        mdescs = desc(model)
        layers, model_inputs, model_outputs = parse_model_description(mdescs)
        xspots = []
        for i, layer in enumerate(layers):
            if layer.class_name == "Activation":
                xspot = XLayerBP.insertXspot(layer, 32)
                xspots.append((i, xspot))
        for c, (i, xs) in enumerate(xspots):
            layers.insert(i + 1 + c, xs)

        rend_descs = render(model_inputs, layers, model_outputs)
        for inp in mdescs['input_layers']:
            nm, p1, p2 = inp[0], inp[1], inp[2]
            for inp2 in rend_descs['input_layers']:
                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                if nm2 == nm and p12==p1 and p22==p2:
                    self.assertTrue(True)
                    break
            else:
                self.assertTrue(False)

        for inp in mdescs['output_layers']:
            nm, p1, p2 = inp[0], inp[1], inp[2]
            for inp2 in rend_descs['output_layers']:
                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                if nm2 == nm and p12==p1 and p22==p2:
                    self.assertTrue(True)
                    break
            else:
                self.assertTrue(False)

        for layer in mdescs['layers']:
            for llayer in rend_descs['layers']:
                if layer['name'] == llayer['name']:
                    self.assertDictEqual(layer['config'], llayer['config'])
                    if len(layer['inbound_nodes']) > 0:
                        for inp in layer['inbound_nodes'][0]:
                            nm, p1, p2 = inp[0], inp[1], inp[2]
                            for inp2 in llayer['inbound_nodes'][0]:
                                nm2, p12, p22 = inp2[0], inp2[1], inp2[2]
                                if nm2 == nm and p12 == p1 and p22 == p2:
                                    self.assertTrue(True)
                                    break
                            else:
                                self.assertTrue(False)

                    break
            else:
                self.assertTrue(False)

if __name__=="__main__":
    unittest.main()