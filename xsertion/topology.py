import json
import logging

import xsertion.strategies as strategies
from xsertion.layers import LayerBP, build_merge, XLayerBP, replicate_layerlist, scale_filters, render

logger = logging.getLogger(__name__)


class SuperLayer():
    def __init__(self):
        self.identifier = None
        self.layers = None

    def get_lane(self, lane_id='', rescale=1.0):
        raise NotImplementedError

    def attach(self, attach_points, rescales=None, rename=''):
        # attach this superlayer to the attach_point collections
        # copy the layers and return appropriate
        return self.get_lane()

    def print_report(self):
        raise NotImplementedError


class InputSuperLayer(SuperLayer):
    def __init__(self, input_layerstack, inputs, outputs):
        super(InputSuperLayer, self).__init__()
        self.identifier = "Inputs"
        self.layers = replicate_layerlist(input_layerstack)
        self.inputs = [(self.layers[input_layerstack.index(l)], p1, p2) for l, p1, p2 in inputs]  # Model inputs
        self.outputs = [(self.layers[input_layerstack.index(l)], p1, p2) for l, p1, p2 in
                        outputs]  # Input model outputs / points for attaching
        # p1, p2 /don't matter, these get preserved on the inputs to the main layer
        self.identifier = 'input'

    def get_lane(self, lane_id='', rescale=1.0):
        # just return copy of the layers
        new_layers = replicate_layerlist(self.layers)
        new_inputs = [(new_layers[self.layers.index(l)], p1, p2) for l, p1, p2 in self.inputs]
        new_attach = [(new_layers[self.layers.index(l)], p1, p2) for l, p1, p2 in self.outputs]
        return new_layers, new_inputs, new_attach

    def get_layer(self, layer):
        return self.layers.index(layer)

    def print_report(self):
        rep_layers, rep_inputs, rep_attach = self.attach([], None)
        print('Input - Layers')
        for layer in rep_layers:
            print(layer.get_name(), [l.get_name() for l in layer.get_inbound()])
        print('Input - Inputs')
        for l, p1, p2 in rep_inputs:
            print(l.get_name(), '@', p1, p2)
        print('Input - Attach Points:', [(l.get_name(), p1, p2) for l, p1, p2 in rep_attach])


class TailSuperLayer(SuperLayer):
    def __init__(self, tail_layerstack, inputs, outputs, concat_axis):
        # tail superlayer has to have a single attach point (otherwise merging does not make sense)
        super(TailSuperLayer, self).__init__()
        self.identifier = "Tail"
        self.layers = replicate_layerlist(tail_layerstack)
        if len(self.layers[0].get_inbound()) != 0:
            raise ValueError("Tail Superlayer does not contain a valid - single input start node")
        self.inputs_ps = inputs  # points to attach the main layer inputs, at p1, p2
        self.outputs = [(self.layers[tail_layerstack.index(l)], p1, p2) for l, p1, p2 in outputs]
        self.axis = concat_axis

    def attach_single(self, con_layer):
        new_layers, start_layer, new_outputs = self.get_lane()
        p1, p2 = self.inputs_ps
        start_layer.add_inbound(con_layer, p1, p2)
        return new_layers, new_outputs

    def attach_multi(self, con_layers):
        merge_layer = LayerBP(build_merge([], name='merge_tl', concat_axis=self.axis))
        for layer in con_layers:
            merge_layer.add_inbound(layer)

        new_layers, new_outputs = self.attach_single(merge_layer)
        new_layers.insert(0, merge_layer)  # prepend the connected merge layer
        return new_layers, new_outputs

    def get_lane(self, lane_id='', rescale=1.0):
        new_layers = replicate_layerlist(self.layers)
        start_layer = new_layers[0]
        assert len(start_layer.get_inbound()) == 0, "Tail layer does not have a single inbound node"
        new_outputs = [(new_layers[self.layers.index(l)], p1, p2) for l, p1, p2 in self.outputs]
        return new_layers, start_layer, new_outputs

    def attach(self, attach_points, rescales=None, rename=''):
        # tail layer also ignores rescales, rename, params
        if isinstance(attach_points, (list, tuple)):
            if len(attach_points) > 1:
                return self.attach_multi(attach_points)
            else:
                return self.attach_single(attach_points[0])
        else:
            return self.attach_single(attach_points)

    def get_layer(self, layer):
        return self.layers[self.layers.index(layer)]

    def print_report(self):
        proxy_attach_layer = LayerBP(dict(name="tail_attach_point", class_name='tail_attach',
                                          config=dict(name="tail_attach_point")))
        rep_layers, rep_outputs = self.attach_single(proxy_attach_layer)
        print('Tail - Layers')
        for layer in rep_layers:
            print(layer.get_name(), [l.get_name() for l in layer.get_inbound()])
        print('Tail - Outputs')
        for l, p1, p2 in rep_outputs:
            print(l.get_name(), '@', p1, p2)


class MainSuperLayer(SuperLayer):
    def __init__(self, main_layers, input_attach_points, tail_attach_points, xspots, concat_axis):
        super(MainSuperLayer, self).__init__()
        self.identifier = 'Main SuperLayer'
        if len(tail_attach_points) > 1:
            raise ValueError('Multiple tail attach points identified')

        tail_attach_point = tail_attach_points[0]
        new_layers = replicate_layerlist(main_layers)
        new_inp_attach_points = [new_layers[main_layers.index(l)] for l in input_attach_points]
        new_tail_attach_point = new_layers[main_layers.index(tail_attach_point)]
        self.concat_axis = concat_axis

        self.xspots = [new_layers[main_layers.index(l)] for l in xspots]  # likely will not have any

        self.layers = new_layers
        self.input_attach_points = new_inp_attach_points
        self.tail_attach_point = new_tail_attach_point

    def get_lane(self, lane_id='', rescale=1.0):
        """rename by appending lane_id and rescale filters by rescale"""
        sp_layers = replicate_layerlist(self.layers)
        sp_inp_attach_points = [sp_layers[self.layers.index(l)] for l in self.input_attach_points]
        sp_tail_attach_point = sp_layers[self.layers.index(self.tail_attach_point)]
        sp_xspots = [sp_layers[self.layers.index(xl)] for xl in self.xspots]

        # rename
        if lane_id != '':
            for layer in sp_layers:
                layer.set_name(layer.get_name() + lane_id)

        # rescale
        if rescale != 1.0:
            scale_filters(sp_layers, rescale)

        return sp_layers, sp_inp_attach_points, sp_tail_attach_point, sp_xspots

    def get_layer(self, layer):
        return self.layers[self.layers.index(layer)]

    def attach(self, attach_points, rescales=None, rename=''):
        if len(attach_points) < 1:
            raise ValueError('need attach points')
        if not rescales:
            rescales = [1.0]
        if rescales and not isinstance(rescales, (list, tuple)):
            raise ValueError('rescales need to be a list')
        attach_point = attach_points[0]  # default to 0 attach point
        if rename != '':
            # mode should be renaming string
            assert isinstance(rename, str)
            sp_layers, sp_inp_attach_points, sp_tail_attach_point, sp_xspots = self.get_lane(lane_id=rename,
                                                                                             rescale=rescales[0])
        else:
            sp_layers, sp_inp_attach_points, sp_tail_attach_point, sp_xspots = self.get_lane(lane_id='',
                                                                                             rescale=rescales[0])

        # attach to input (presumably)
        input_attach_layer, p1, p2 = attach_point
        for layer in sp_inp_attach_points:
            layer.add_inbound(input_attach_layer, p1, p2)

        return sp_layers, sp_xspots, sp_tail_attach_point


class MultiLayer(MainSuperLayer):
    def insert_x_spots(self, xspot_function):
        sp_layers, sp_inp_attach_points, sp_tail_attach_point, sp_xspots = self.get_lane()
        assert len(sp_xspots) == 0, "X-spots already inserted"
        xspots = []
        added = []
        for i, (base_filter_count, before_layer) in enumerate(xspot_function(sp_layers)):
            ii = sp_layers.index(before_layer)
            xspot = XLayerBP.insertXspot(before_layer, base_filter_count, depth_id=i)
            xspots.append(xspot)
            added.append((xspot, ii))
        for c, (xspot, i) in enumerate(added):
            sp_layers.insert(i + 1 + c, xspot)  # add xspot at position i+1 *just after before_layer, c-controls

        return MultiLayer(sp_layers, sp_inp_attach_points, [sp_tail_attach_point], xspots, self.concat_axis)

    def attach(self, attach_points, rescales=None, rename=''):
        # attach for each attach point, rescaled by rescales [i], ignore renames, rename to "_{lid}"
        assert isinstance(attach_points, list)
        if rescales is None:
            rescales = [1.0 for p in attach_points]
        assert isinstance(rescales, (list, tuple))
        assert len(attach_points) == len(rescales)
        if len(attach_points) == 1 and isinstance(rename, str):
            return super(MultiLayer, self).attach(attach_points, rescales=rescales, rename=rename)
        elif not isinstance(rename, list):
            # renames should be lids to use
            raise ValueError("Rename did not contain a list of lids")
        assert len(attach_points) == len(rename)
        sps_layers, sps_xspots, sps_tail_attach_points = [], [], []
        for lid, attach_point, rescale in zip(rename, attach_points, rescales):
            sp_layers, sp_xspots, sp_tail_attach_point = super(MultiLayer, self).attach([attach_point],
                                                                                        rescales=[rescale],
                                                                                        rename='_{}'.format(lid))
            for xspot in sp_xspots:
                xspot.set_lid(lid)

            sps_layers.append(sp_layers)
            sps_xspots.append(sp_xspots)
            sps_tail_attach_points.append(sp_tail_attach_point)

        num_of_layers = len(sps_layers[0])
        num_of_xspots = len(sps_xspots[0])

        # flatten main_layers
        main_layers = []
        for i in range(num_of_layers):
            for sp_layers in sps_layers:
                assert len(sp_layers) == num_of_layers
                main_layers.append(sp_layers[i])

        xspots_by_depth = []
        # group xspots by depth
        for i in range(num_of_xspots):
            xspot_group = []
            for xspots in sps_xspots:
                assert len(xspots) == num_of_xspots
                xspot_group.append(xspots[i])
            xspots_by_depth.append(xspot_group)

        return main_layers, xspots_by_depth, sps_tail_attach_points


class ModelBuilder():
    def __init__(self, input_superlayer, main_superlayer, tail_superlayer, outerdict):
        self.input_superlayer = input_superlayer
        self.main_sl = main_superlayer
        self.main_superlayer = main_superlayer
        self.tail_superlayer = tail_superlayer

        self.outerdict = outerdict

        self.input_layersnames = dict()
        self.input_layer_lids = []

        for lid, (l, p1, p2) in enumerate(self.input_superlayer.outputs):
            self.input_layersnames[l.get_name()] = lid
            self.input_layer_lids.append(l.get_name())

        self.num_of_lanes = len(self.input_layer_lids)

        logger.info("Created ModelBuilder with {} different input attach points".format(self.num_of_lanes))
        for lid in range(self.num_of_lanes):
            logger.debug("Superlayer {} after input layer {}".format(lid, self.input_layer_lids[lid]))

    def get_xspot_before_layers(self):
        return [list(xspot.get_inbound())[0] for xspot in self.main_superlayer.xspots]

    def get_xspot_before_layernames(self):
        return [l.get_name() for l in self.get_xspot_before_layers()]

    def has_in_main(self, name):
        for layer in self.main_sl.layers:
            if layer.get_name() == name:
                return True
        return False

    def has_in_tail(self, name):
        for layer in self.tail_superlayer.layers:
            if layer.get_name() == name:
                return True
        return False

    def bake(self, name, inputs, layers, outputs):
        model_desc = self.outerdict.copy()
        logger.debug('Rendering')
        model_desc['config'] = render(inputs, layers, outputs)
        model_desc['config']['name'] = name

        # with open('dev_model.json', 'w') as out:
        #     out.write(json.dumps(model_desc, indent=4))
        logger.debug('Building')
        from keras.models import model_from_json
        model = model_from_json(json.dumps(model_desc))

        return model

    def build_single_lane(self, input_attach, rescale=1.0, modelname=None):
        lid = None
        lname = None
        if isinstance(input_attach, int):
            assert input_attach < self.num_of_lanes and input_attach >= 0, "Unknown input_attach point"
            lid = input_attach
            lname = self.input_layer_lids[input_attach]
        elif isinstance(input_attach, str):
            assert input_attach in self.input_layersnames, "Unknown layer name for input_attach point"
            lid = self.input_layersnames[input_attach]
            lname = input_attach
        else:
            raise ValueError('Input attach point must int id or str name')

        new_ilayers, new_inputs, new_attach = self.input_superlayer.attach(None)
        new_mlayers, xspots, tail_attach = self.main_superlayer.attach([new_attach[lid]], rescales=[rescale])
        new_olayers, new_outputs = self.tail_superlayer.attach(tail_attach)

        new_layers = new_ilayers + new_mlayers + new_olayers
        if modelname and isinstance(modelname, str):
            return self.bake(modelname, new_inputs, new_layers, new_outputs)
        else:
            return self.bake(lname + "_superlayer_model", new_inputs, new_layers, new_outputs)


    # stategies to use
    after_pooling = strategies.xspot_after_pooling
    resnet = strategies.xspots_resnet_before_sum_merge
    layername = strategies.named_layer_strategy_maker

    def insert_x_spots(self, xspot_function):
        # get x_spots
        self.main_superlayer = self.main_sl.insert_x_spots(xspot_function)
        logger.info("Found {} xspots".format(len(self.main_superlayer.xspots)))
        for xspot in self.main_superlayer.xspots:
            logger.info("xspot after {}".format(",".join( [l.get_name() for l in xspot.get_inbound()])))

    def build_multi_lane(self, lanes, rescales, modelname=None, params=None):
        lids = None
        lnames = None
        assert isinstance(lanes, list)
        assert isinstance(rescales, list)

        assert len(lanes) == len(rescales)

        if all(isinstance(lid, int) and lid < self.num_of_lanes and lid >= 0 for lid in lanes):
            lids = lanes
            lnames = [self.input_layer_lids[l] for l in lanes]
        elif all(isinstance(lid, str) and lid in self.input_layersnames for lid in lanes):
            lids = [self.input_layersnames[l] for l in lanes]
            lnames = lanes
        else:
            raise ValueError("Lids not present")

        name = None
        if modelname and isinstance(modelname, str):
            name = modelname
        else:
            name = "_".join(lnames) + "_model"

        il, inps, attach_pts = self.input_superlayer.attach(None)
        ml, xspots_by_depth, t_a_pts = self.main_superlayer.attach([attach_pts[lid] for lid in lids],
                                                                   rescales=rescales,
                                                                   rename=lids)
        tl, outs = self.tail_superlayer.attach(t_a_pts)

        layers = il + ml + tl

        if params:
            # build xcross connections
            logger.debug("Building xcons using: "+str(params))
            filters, use_bn, con_fact = params
            for xcon_group in xspots_by_depth:
                XLayerBP.buildXconnections(xcon_group, filters, self.main_superlayer.concat_axis, use_bn, con_fact)

        model = self.bake(name, inps, layers, outs)
        return model
