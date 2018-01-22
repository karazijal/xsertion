from xsertion.layers import LayerBP


def is_xspot(layer: LayerBP):
    """
    :param layer: layer that is being check that could be input of x-conn connection
    :return: None if not a xspot or layer (not necessarily the same) that could be asked
    """
    # perform a check that layer is something that could be connected to
    if 'pooling' in layer.class_name.lower():  # Look for pooling layers:
        if layer._get_outbound():  # layer has connections (not final one)
            llayers = list(layer._get_outbound())
            if len(llayers) == 0:
                # only single
                llayer = llayers[0]
                if any(ltype in llayer.class_name.lower() for ltype in ['merge', 'batchnormal']):
                    # the single layer is some type of merge, batchnorm, so try use that instead
                    if llayer._get_outbound():
                        return llayer  # it is actually used
                    else:
                        return None  # layer is followed by only llayer which isn't used
                elif 'dropout' in llayer.class_name.lower():  # if we have dropout following the pooling, use pooling
                    if llayer._get_outbound():
                        return layer
                    else:
                        return None
                else:
                    # some other, likely functional layer type, so use layer
                    return layer
            else:
                # have layer that is consumed by multiple others,
                if any(bool(llayer._get_outbound()) for llayer in layer._get_outbound()):
                    return layer  # some of those likely functional other are also consumed (not dead splitting point)
                else:
                    return None
        else:
            # this layer is not used by any other layers in main part
            return None
    return None


def xspot_after_pooling(layers):

    xspots = []
    before_layers = set()
    last_seen_filters = -1  # Nothing fallback
    for layer in layers:
        if layer.has_property('nb_filter') and 'convolution2d' in layer.class_name.lower():
            last_seen_filters = layer.get_property('nb_filter')
        bl = is_xspot(layer)
        if bl and bl not in before_layers:  # not None:
            # xspots.append((last_seen_filters, bl))
            xspots.append((-1, bl))
            before_layers.add(bl)
    return xspots


def xspots_resnet_before_sum_merge(layers):
    sum_merges_bottlenecks = []
    ind = {layer: i for i, layer in enumerate(layers)}
    max_filters = 16  # min 16 filters for connections
    for layer in layers:
        if layer.class_name == "Merge" and layer.has_property('mode') and layer.get_property('mode') == 'sum' and \
                        len(layer.get_inbound()) == 2:
            sum_merges_bottlenecks.append(layer)
        if "Convolution" in layer.class_name and layer.get_property('nb_row') > 1 and layer.get_property('nb_col') > 1:
            if layer.get_property('nb_filter') > max_filters:
                max_filters = layer.get_property('nb_filter')

    xspots = []
    for sum_merge in sum_merges_bottlenecks:
        # have two connections:
        # add the non-sum-merge_ones:
        inbs = list(sum_merge.get_inbound())
        assert len(inbs) == 2
        for l in sorted(inbs, key=lambda lyr: ind[lyr]):  # sort to preserve overall ordering
            if l.class_name != "Merge":
                xspots.append((-1, l))

    return xspots

def named_layer_strategy_maker(layernames:list, filters = None):
    def named_layer_strategy(layers):
        lnames = set(layernames)
        last_seen_filters = 32  # Nothing fallback

        before_layers = []
        for layer in layers:
            if layer.has_property('nb_filter') and 'convolution2d' in layer.class_name.lower():
                last_seen_filters = layer.get_property('nb_filter')
            if layer.get_name() in lnames:
                if filters is not None and isinstance(filters, int):
                    f = filters
                elif filters and isinstance(filters, list) and len(filters)>= len(layernames):
                    f = filters[layernames.index(layer.get_name())]
                else:
                    f = last_seen_filters
                before_layers.append((f, layer))
        return before_layers
    return named_layer_strategy