import copy
import math
import logging

import xsertion.xcons as xcons

logger = logging.getLogger(__name__)

def cf(x: float, eps=0.0001):
    return int(math.ceil(x - eps))


def build_internal_adj_lists(layerlist):
    adj_lists = dict()
    for layer in layerlist:
        adj_list = dict()
        for inbl in layer.get_inbound():
            if inbl in layerlist:  # keep the layers list internally consistent
                p1, p2 = layer.inb[inbl]
                adj_list[inbl] = (p1, p2)
        adj_lists[layer] = adj_list
    return adj_lists


def translate_adj_lists(adj_lists, translate_dict):
    new_adj_lists = dict()  # nlayer -> new_adj_list
    for layer in adj_lists:
        if layer in translate_dict:
            nlayer = translate_dict[layer]
            new_adj_list = dict()  # ninbl -> p1, p2
            for inbl in adj_lists[layer]:
                if inbl in translate_dict:
                    ninbl = translate_dict[inbl]
                    p1, p2 = adj_lists[layer][inbl]
                    new_adj_list[ninbl] = (p1, p2)
            new_adj_lists[nlayer] = new_adj_list
    return new_adj_lists


def apply_adj_lists(layerstack, adj_lists):
    assert all(l in adj_lists for l in layerstack), "layers do not occur in the adj_lists"
    for l in layerstack:
        l.set_inbound([])  # remove any connection
        l._set_outbound([])  # remove any connection
    for layer in layerstack:
        for inb in adj_lists[layer]:
            p1, p2 = adj_lists[layer][inb]
            layer.add_inbound(inb, p1, p2)


def replicate_layerlist(layerlist):
    new_layers = [l.replicate() for l in layerlist]
    trans_dict = {ol: nl for ol, nl in zip(layerlist, new_layers)}
    old_adj_lists = build_internal_adj_lists(layerlist)
    new_adj_lists = translate_adj_lists(old_adj_lists, trans_dict)
    apply_adj_lists(new_layers, new_adj_lists)
    return new_layers


def parse_model_description(base_config):
    # Build _Layers from model config
    layers = []  # Layer Objects
    layerdict = dict()  # Mapping layer_name -> Layer Object: Layer name is only used here!!!

    _aux_added_layers = []
    for ldesc in base_config['layers']:
        layer = LayerBP(ldesc)
        lname = ldesc['name']
        layers.append(layer)
        layerdict[lname] = layer
        inbs = []
        if ldesc['inbound_nodes']:
            for l in ldesc['inbound_nodes'][0]:
                layer_name, po1, po2 = l[0], l[1], l[2]
                inbs.append((layer_name, po1, po2))
        _aux_added_layers.append((layer, inbs))

    for dest_layer, inbound_cons in _aux_added_layers:
        # form a connections
        for ori_layer_name, p1, p2 in inbound_cons:
            ori_layer = layerdict[ori_layer_name]
            dest_layer.add_inbound(ori_layer, p1=p1, p2=p2)

    model_inputs = []
    for inp in base_config['input_layers']:
        nm, p1, p2 = inp[0], inp[1], inp[2]
        model_inputs.append((layerdict[nm], p1, p2))

    model_outputs = []
    for out in base_config['output_layers']:
        nm, p1, p2 = out[0], out[1], out[2]
        model_outputs.append((layerdict[nm], p1, p2))

    # Toposort layers:

    edge_dict = {layer: len(layer.get_inbound()) for layer in layers}

    topo_sorted_layer = []
    empty_layers = {layer for layer in edge_dict if edge_dict[layer] == 0}

    start = True
    while empty_layers:
        if not start:
            min_l = None
            for l in empty_layers:
                if not min_l:
                    min_l = l
                else:
                    if len(min_l.get_inbound()) > len(l.get_inbound()):
                        min_l = l
                    elif len(min_l.get_inbound()) == len(l.get_inbound()):
                        min_l = min(min_l, l, key=lambda lyr: lyr.get_name())  # lexicographically
            cur_layer = min_l
        else:
            cur_layer = model_inputs[0][0]  # start with input
            assert edge_dict[cur_layer] == 0
            assert cur_layer in empty_layers
            start = False
        empty_layers.remove(cur_layer)
        topo_sorted_layer.append(cur_layer)
        for layer in cur_layer._get_outbound():
            edge_dict[layer] -= 1
            if edge_dict[layer] == 0:
                empty_layers.add(layer)

    # remove outputs from the topo sort
    outputs = []
    for ol, p1, p2 in model_outputs:
        topo_sorted_layer.remove(ol)
        outputs.append(ol)
    # add them to the very end
    topo_sorted_layer.extend(outputs)

    return topo_sorted_layer, model_inputs, model_outputs


def filter_used_layers(layers, ignore):
    new_layers = []
    # keep only those layers that have their output used
    for layer in layers:
        if len(layer._get_outbound()) > 0 or layer in ignore:
            new_layers.append(layer)
    return new_layers


def render(inputs, layers, outputs):
    config = dict(name="placeholder_model", input_layers=[], output_layers=[], layers=[])
    flayers = filter_used_layers(layers, {l for l, p1, p2 in outputs})
    for layer in flayers:
        layer_descs = layer.get_descs()
        for ldesc in layer_descs:
            config['layers'].append(ldesc)
    for inp, p1, p2 in inputs:
        config['input_layers'].append([str(inp.get_con_name()), p1, p2])
    for out, p1, p2 in outputs:
        config['output_layers'].append([str(out.get_con_name()), p1, p2])
    return config


def rename(layers, count_start=0, suffix=""):
    for layer in layers:
        new_name = layer.class_name.lower() + "_{}{}".format(count_start, suffix)
        layer.set_name(new_name)
        count_start += 1
    return count_start


def scale_filters(layerstack, scale):
    for layer in layerstack:
        if issubclass(type(layer), XLayerBP):
            continue
        elif layer.has_property('nb_filter'):
            new_filter = cf(float(layer.get_property('nb_filter')) * scale)
            layer.set_property('nb_filter', new_filter)


class LayerBP():
    """Container class that holds layers build from json"""

    def __init__(self, desc: dict):
        self.name = desc['name']
        self.class_name = desc['class_name']
        self.config = desc['config'].copy()
        assert (self.config['name'] == self.name)
        self.inb = dict()  # (layer_object) -> (p1, p2)
        self.outb = set()  # (layer_object)

    def replicate(self):
        proxy_config = copy.deepcopy(self.config)
        proxy_dict = dict(name=self.name, class_name=self.class_name, config=proxy_config)
        return LayerBP(proxy_dict)

    def _inbound(self):
        # calculates the inbound connection list:
        inb = []
        for l in self.inb:
            p1, p2 = self.inb[l]
            inb.append([l.get_con_name(), p1, p2])
        if inb:
            return [inb]
        else:
            return inb

    def get_inbound(self):
        return self.inb

    def add_inbound(self, layer, p1=0, p2=0):
        if layer not in self.inb:
            layer._add_outbound(self)
            self.inb[layer] = p1, p2

    def set_inbound(self, layers, p1s=None, p2s=None):
        if p2s is None:
            p2s = [0 for l in layers]
        if p1s is None:
            p1s = [0 for l in layers]
        assert (len(layers) == len(p1s) == len(p2s))
        self.inb = dict()
        for layer, p1, p2 in zip(layers, p1s, p2s):
            layer._add_outbound(self)
            self.inb[layer] = p1, p2

    def replace_inbound(self, new_layer, old_layer, np1=None, np2=None):
        # logger.debug("Replacing {} with {} for {}".format(old_layer.get_con_name(), new_layer.class_name, self.get_con_name()))
        if old_layer in self.inb:
            op1, op2 = self.inb[old_layer]
            if np1 is None:
                np1 = op1
            if np2 is None:
                np2 = op2

            old_layer._remove_outbound(self)  # remove old connection
            del self.inb[old_layer]
            self.add_inbound(new_layer, p1=np1, p2=np2)

    def _add_outbound(self, layer):
        if layer not in self.outb:
            self.outb.add(layer)

    def _set_outbound(self, layers):
        self.outb = set(layers)

    def _get_outbound(self):
        return self.outb

    def _replace_outbound(self, new_layer, old_layer):
        self._remove_outbound(old_layer)
        self._add_outbound(new_layer)

    def _remove_outbound(self, rm_layer):
        if rm_layer in self.outb:
            self.outb.remove(rm_layer)

    def set_name(self, new_name):
        """Handels name for connections"""
        self.name = new_name
        self.config['name'] = new_name

    def get_name(self):
        return self.name

    def get_con_name(self):
        """return names for identifiers in connections"""
        return self.get_name()

    def has_property(self, property):
        return property in self.config

    def set_property(self, property, value):
        if property == "name":
            self.set_name(value)
        elif self.has_property(property):
            self.config[property] = value

    def get_property(self, property):
        if self.has_property(property):
            return self.config.get(property, None)
        else:
            return None

    def get_descs(self):
        # build desc
        return [dict(
            name=self.get_con_name(),
            class_name=self.class_name,
            inbound_nodes=self._inbound(),
            config=self.config
        )]


class XLayerBP(LayerBP):
    """A layer object which either exists as a passthrough for normal use, or when x-connected, renders to x-connections"""

    @classmethod
    def buildXconnections(cls, xspots, filters, concat_axis, use_bn, con_fact):
        assert isinstance(con_fact, xcons.ConnectionFactory)
        logger.debug('Xconnecting: {}'.format(xspots))
        logger.debug('Connections with BN ({}) using {}'.format(use_bn, con_fact))
        for i, xspot in enumerate(xspots):
            other_spots = xspots[:i] + xspots[i + 1:]
            logger.debug("SP {}:Connecting {} with {}".format(xspot.lid, xspot.get_name(),
                    ", ".join(["{} from sp({})".format(l.get_name(), l.lid) for l in other_spots])))
            xspot.xconnect(other_spots, con_fact, filters, concat_axis, use_bn)

    @classmethod
    def insertXspot(cls, before_layer: LayerBP, base_filter_count: int, depth_id=0):
        xlayer = XLayerBP()
        for after_layer in list(before_layer._get_outbound()):
            # instead of being connected to before_layer, reconnect to xlayer
            after_layer.replace_inbound(xlayer, before_layer)
        xlayer.add_inbound(before_layer)
        xlayer.set_name(xlayer.get_name() + '_' + str(depth_id))
        xlayer.base_nb_filter = base_filter_count
        return xlayer

    def __init__(self):
        name = 'xspot'
        class_name = 'XSpot'
        desc = dict(name=name, config=dict(name=name), class_name=class_name)
        super(XLayerBP, self).__init__(desc)
        self.name = name
        self.class_name = class_name
        self.xconnected = False
        self.base_nb_filter = -2 # -2 error state; -1 perform search ; 0 do not form connection
        self.lid = None

    def set_lid(self, lid: int):
        self.lid = lid

    def replicate(self):
        to_rtn = XLayerBP()
        to_rtn.set_name(self.get_name())
        to_rtn.class_name = self.class_name
        to_rtn.base_nb_filter = self.base_nb_filter
        to_rtn.lid = self.lid
        return to_rtn

    def unx_get_con_name(self):
        assert len(self.inb) > 0, "XLayerBP not connected, but asked to give connection name"
        assert len(self.inb) == 1, "XLayerBP connected to multiple other layers"
        before_layer = list(self.get_inbound())[0]
        return before_layer.get_con_name()  # pass through the layer when rendering

    def unx_get_descs(self):
        return []  # Do not render anything

    def get_descs(self):
        if self.xconnected:
            return self.x_get_descs()
        else:
            return self.unx_get_descs()

    def get_con_name(self):
        if self.xconnected:
            logger.debug("{} got asked for name! {} returned as xc-ed".format(self.name, self.con_name))
            return self.x_get_con_name()
        else:
            logger.debug('{} got asked for name! {} returned as passthrough'.format(self.name, self.unx_get_con_name()))
            return self.unx_get_con_name()

    def trace_filt(self):
        if hasattr(self, 'nb_filter'):
            return self.nb_filter
        else:
            logger.info("Tracing nb_filter for xcon {} in {}".format(self.get_name(), self.lid))
            before_layer = list(self.get_inbound())[0]
            current_check_layer = before_layer
            check_queue = []
            logger.debug("Before_layer layer is {}".format(before_layer.get_con_name()))
            while not current_check_layer.has_property('nb_filter'):
                logger.debug('{} did not have filters'.format(current_check_layer.get_name()))
                for layer in list(current_check_layer.get_inbound()):
                    check_queue.append(layer)
                if check_queue:
                    current_check_layer = check_queue.pop(0)
                else:
                    # ran out of layers... either fallback count of no connection
                    current_check_layer = None
                    break
            if current_check_layer is not None:
                base_nb_filter = current_check_layer.get_property('nb_filter')
                logger.debug('Found {} filters used by {}'.format(base_nb_filter, current_check_layer.get_con_name()))
            else:
                base_nb_filter = 0
                logger.debug('Cound not find local filter counts, set to 0')
            self.nb_filter = base_nb_filter
            return self.nb_filter


    def xconnect(self, inbound_xspots: list, con_fact : xcons.ConnectionFactory, filters: dict, concat_axis: int, use_bn : bool):
        assert self.base_nb_filter > -2, "Filters unassigned"
        assert self.lid is not None, "Layer has not been assigned a lane"
        # X-connection are formed lazilly - they only get planted when rendering
        self.concat_axis = concat_axis
        self.xconvs = dict()  # stores lid-> (inb_layer, fl_count)
        self.con_fact = con_fact
        self.use_bn = use_bn
        self.con_name = self.get_name().replace('xspot', 'xmerge')
        for xspot in inbound_xspots:
            before_layer = list(xspot.get_inbound())[0]
            ilid = xspot.lid
            nbf = filters[ilid][self.lid]


            if self.base_nb_filter == -1:
                base_nb_filter = xspot.trace_filt()
            else:
                base_nb_filter = self.base_nb_filter
            nb_filter = cf(base_nb_filter * nbf)
            logger.info("SP {}: Registered inbound xcon from {} with {}*{}={} filters".format(self.lid, ilid,
                                                                                            round(nbf, 3), base_nb_filter,
                                                                                            nb_filter))
            self.xconvs[ilid] = before_layer, nb_filter  # save params from ilid -> self.lid connection
        self.xconnected = True

    def x_get_con_name(self):
        return self.con_name

    def x_get_descs(self):
        """Builds X-connection"""
        # print(self.base_nb_filter)
        layer_desc = []
        inline_connection_name = list(self.get_inbound())[0].get_con_name()
        merge_layer_names = []
        for lid in self.xconvs:
            bl, nb_filter = self.xconvs[lid]
            name = "_".join(self.get_name().split('_')[:-1]).replace('spot', 'c') + '_{}to{}'.format(lid, self.lid)
            if nb_filter > 0:
                descs = self.con_fact(bl.get_con_name(), name, nb_filter)
                layer_desc.extend(descs)
                merge_layer_names.append(name)
        if not merge_layer_names: # if no iter-superlayer connection exist
            if self.use_bn:
                # if we use BN then just insert BN on the inline connection
                layer_desc.append(build_bn(inline_connection_name, self.x_get_con_name(), self.concat_axis))
            else:
                # nothing to merge and no bn to insert
                self.xconnected = False #
                logger.debug("SP {}: XP {}| nothing merged (un-xconnecting into passthrough)".format(self.lid,
                                                                                                     self.get_name()))
        else:
            merge_layer_names.insert(0, inline_connection_name)
            layer_desc.extend(build_xcon_merge(merge_layer_names, self.x_get_con_name(), self.concat_axis, self.use_bn))
        return layer_desc


def build_merge(input_names, name, concat_axis):
    """function which adds concat merging of input_names"""
    return {
        "class_name": "Merge",
        "name": name,
        "config": {
            "output_mask": None,
            "output_mask_type": "raw",
            "dot_axes": -1,
            "output_shape": None,
            "concat_axis": concat_axis,
            "mode": "concat",
            "name": name,
            "output_shape_type": "raw",
            "arguments": {},
            "mode_type": "raw"
        },
        "inbound_nodes": [
            [[nm, 0, 0] for nm in input_names]
        ]
    }


def build_bn(input_name, name, axis):
    return {
        "config": {
            "trainable": True,
            "epsilon": 0.001,
            "momentum": 0.99,
            "axis": axis,
            "name": name,
            "mode": 0,
            "gamma_regularizer": None,
            "beta_regularizer": None
        },
        "class_name": "BatchNormalization",
        "inbound_nodes": [
            [
                [
                    input_name,
                    0,
                    0
                ]
            ]
        ],
        "name": name
    }


def build_xcon_merge(input_names, name, concat_axis, use_bn=True):
    """
    :param input_names: names of layers which are being merged
    :param name: final identifier of the merged connection; also unique, so use for derivatice intermediate layer names
    :param concat_axis: concat_axis along which to perform the merge
    :return: list of layer_desk
    """
    if use_bn:
        mrg = build_merge(input_names, name + "_mrg", concat_axis)
        bn = build_bn(name + "_mrg", name, concat_axis)
        return [mrg, bn]
    else:
        return [build_merge(input_names, name, concat_axis)]


