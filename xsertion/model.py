import json
import logging

from xsertion.keras_interface_util import get_Model, is_valid_optimizers, is_valid_loss, unwrap_sequential, \
    transfer_weights, average_weights, average_transfer_weights, ModelProxy, tf_ses_ctl_reset, simple_model_func
from xsertion.layers import parse_model_description, cf
from xsertion.topology import InputSuperLayer, TailSuperLayer, MultiLayer, ModelBuilder
from xsertion.xcons import XSpotConFactory, FunctionalModelFactory

logger = logging.getLogger(__name__)


class InvalidModel(Exception):
    def __init__(self, model=None, reason="Model was found to not work", summary=False):
        super(InvalidModel, self).__init__()
        self.model = model
        self.reason = reason
        if self.model:
            self.msg = "Reason: {} |\n In model: {} (mem_ref)".format(self.reason, self.model) if not summary else \
                "Reason: {} |\n In model: {} (mem_ref) |\n{}".format(self.reason, self.model, self.model.summary())


class XCNNBuilder():
    """Parse a prebuild model to look for spots where to place cross-modal connections"""

    def __init__(self, model, input_model, input_split_axis='auto', tailstart_name=None, alpha=1, beta=2):
        self.train_set = False
        self.compile_set = False

        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha is not int/float")

        self.alpha = alpha

        if not isinstance(beta, (int, float)):
            raise ValueError("beta is not int/float")

        self.beta = beta

        if not model or not issubclass(type(model), get_Model()):
            raise InvalidModel(model=model, reason="Not a Model")

        if isinstance(model.input_shape, tuple):
            if len(model.input_shape) != 4:
                raise InvalidModel(model=model, reason="Input does not have 4 batch dimensions")
        else:
            raise InvalidModel(model=model, reason="Multiple/unknown input dimensions")

        if not input_model or not issubclass(type(model), get_Model()):
            raise InvalidModel(model=input_model, reason="Not a Model")

        if isinstance(input_model.output_shape, list):
            if len(input_model.output_shape) <= 1:
                raise InvalidModel(model=input_model, reason='Input model does not have multiple outputs')
            elif any(len(out) != 4 for out in input_model.output_shape):
                raise InvalidModel(model=input_model, reason='Input model outputs are not 4 dimensional')
        else:
            raise InvalidModel(model=input_model, reason='Input model does not have multiple outputs')

        self.base_model = model
        # TODO: check input
        if isinstance(input_split_axis, str):
            if input_split_axis != 'auto':
                raise ValueError("Unknown input split axis value: {}".format(input_split_axis))
            else:
                assert model.input_shape[0] is None, "batch shape does not have None"
                # one dimesions should not match
                for i in range(1, 4):
                    if all(model.input_shape[i] == out[i] for out in input_model.output_shape):
                        pass  # Dimension matched! should something else happen?
                    elif all(model.input_shape[i] != out[i] for out in input_model.output_shape):
                        input_split_axis = i
                        logger.info('Input split axis detected: {}'.format(i))
                        break
                    else:
                        raise InvalidModel(model=input_model, reason="Dimensions of input model's outputs "
                                                                     "are not consistent with base model")
                else:
                    raise ValueError("Could not determine input split axis, please specify")
        elif isinstance(input_split_axis,
                        int) and input_split_axis >= 1 and input_split_axis <= 4:  # because 4 dimensional
            for i in range(1, 4):
                if i != input_split_axis:  # all dimensions other than inpu_split_axis must agree
                    if not all(model.input_shape[i] == out[i] for out in input_model.output_shape):
                        raise InvalidModel(model=input_model, reason="Outputs of input model have different "
                                                                     "dimensions other than input split axis")
        else:
            raise ValueError("Please specify axis along which different 'splits' of wide data are presented")
        self.split_axis = input_split_axis
        self.base_model = model

        # extract training parameters from model -??


        # Work with Model if we have Sequential (Sequential is a wrapper)
        self.base_model_name = self.base_model.name
        self.base_model = unwrap_sequential(self.base_model)
        base_model_disc = json.loads(self.base_model.to_json())
        self.outerdict = {
            "class_name": "Model",
            "keras_version": base_model_disc['keras_version'],
            "config": dict()
        }

        # with open('dev_base.json', 'w') as out:
        #     json.dump(base_model_disc, out, indent=4)

        layers_topo, model_inputs, model_outputs = parse_model_description(base_model_disc['config'])
        # layers_by_depth is a list of list containing LayerBP object created by proccesing layer descriptions
        if len(model_inputs) > 1:
            raise InvalidModel(model=model, reason="Model does not have a single input")
        if len(model_outputs) > 1:
            raise InvalidModel(model=model, reason="Model does not have a single output")
        assert layers_topo[0] == model_inputs[0][0]  # the input is first
        assert layers_topo[-1] == model_outputs[0][0]  # the output is last

        topo_check_flag = True
        ind = {layer: i for i, layer in enumerate(layers_topo)}
        topo_check_flag = any(
            not (any(ind[l] > i for l in layer.get_inbound()) or any(ind[l] < i for l in layer._get_outbound()))
            for i, layer in enumerate(layers_topo)
        )
        if not topo_check_flag:
            raise InvalidModel(model=model, reason="Model could not be topologically sorted")
        main_model_layer_names = {l.get_name() for l in layers_topo}

        # Identify the input:
        input_layer = layers_topo[0]
        if input_layer.class_name != "InputLayer":
            raise InvalidModel(model=self.base_model, reason="Model did not contain valid single input")

        if input_layer.has_property("batch_input_shape"):
            self.input_shape = input_layer.get_property("batch_input_shape")[1:]
        else:
            raise InvalidModel(model=self.base_model, reason="Could not determine input shape")
        main_model_layer_names -= {input_layer.get_name()}

        for phrase in ['merge_tl', 'xspot', 'xcon']:
            if any(phrase in name for name in main_model_layer_names):
                raise InvalidModel(model=self.base_model,
                                   reason="Model contained layers with {} in name".format(phrase))

        # Cut off the input
        main_tail = layers_topo[1:]
        tail_start = 0
        # detect tail layers:

        if tailstart_name:
            if not isinstance(tailstart_name, str):
                raise ValueError('Supplied tailstart_name is not a string')
            if tailstart_name not in main_model_layer_names:
                raise InvalidModel(model=self.base_model, reason="Layer named {} was given as a start for tail-end "
                                                                 "of the model but does not occur in it".format(
                    tailstart_name))
            for i, layer in enumerate(main_tail):
                if layer.get_name() == tailstart_name:
                    tail_start = i
                    break
            else:
                raise InvalidModel(model=self.base_model, reason="Tail start layer could no be located")
        else:  # was None or empty string!, detect automatically!
            for i in range(len(main_tail)):
                ind = -i - 1  # go backwards
                layer = main_tail[ind]
                if layer.class_name == "Flatten":
                    tail_start = ind + len(main_tail)
                    break
            else:
                # Fallback to using the first dense layer!
                for i in range(len(main_tail)):
                    ind = -i - 1  # go backwards
                    layer = main_tail[ind]
                    if layer.class_name == "Dense":
                        tail_start = ind + len(main_tail)
                        break
                else:
                    raise InvalidModel(model=self.base_model, reason="Could not identify the tail of the model (Flatten"
                                                                     " or Dense layer)")

        # validate tail start
        ind = {layer: i for i, layer in enumerate(main_tail)}
        for layer in main_tail[tail_start + 1:]:
            for inb_layer in layer.get_inbound():
                if ind[inb_layer] < tail_start:
                    raise InvalidModel(model=self.base_model, reason=
                    "The identified tail is layer {}, but layer {} requires {}".format(main_tail[tail_start].get_name(),
                                                                                       layer.get_name(),
                                                                                       inb_layer.get_name()))

        logger.info("Tail start at {} {}".format(tail_start, main_tail[tail_start].get_name()))
        main = main_tail[:tail_start]
        tail = main_tail[tail_start:]

        assert len(list(tail[0].get_inbound())) == 1, "Could not identify single tail attach point"
        tail_attach_point = list(tail[0].get_inbound())[0]
        tail_attach_point_ps = tail[0].get_inbound()[tail_attach_point]

        input_layer_by_depth_flat, input_model_input, input_model_output = parse_model_description(
            json.loads(input_model.to_json())['config'])

        self.number_of_lanes = len(input_model_output)

        input_superlayer = InputSuperLayer(input_layer_by_depth_flat, input_model_input, input_model_output)
        tail_superlayer = TailSuperLayer(tail, tail_attach_point_ps, model_outputs, self.split_axis)
        main_superlayer = MultiLayer(main, list(input_layer._get_outbound()), [tail_attach_point], [], self.split_axis)

        self.model_factory = ModelBuilder(input_superlayer, main_superlayer, tail_superlayer, self.outerdict)

        self.set_xcons(simple_model_func)  # default
        self.set_xspot_strategy('after_pooling')

        self.analysed_base_scaling = False
        self.analysed_single_superlayer_accuracy = False
        self.analysed_xcon_pair = False
        self.iter_build = False

        self.curr_mes = None
        self.curr_mes_res = None
        self.curr_mes_completed_states = None

        self.add_persistence()

    def get_single_lane(self, attach_id, rescale=1.0):
        if attach_id < 0 or attach_id >= self.number_of_lanes:
            raise ValueError("Wrong attach id")
        return self.model_factory.build_single_lane(attach_id, rescale=rescale)

    def get_all_lanes(self, rescales):
        if not isinstance(rescales, list):
            raise ValueError('Rescales parameters is not a list')
        if not all(isinstance(r, float) for r in rescales):
            raise ValueError('Rescales should be floats')
        if len(rescales) != self.number_of_lanes:
            raise ValueError('Length of rescales does not equal number of lanes')
        lids = list(range(self.number_of_lanes))
        return self.model_factory.build_multi_lane(lids, rescales)

    def get_pair(self, l1, l2, rescales):
        if not isinstance(rescales, list):
            raise ValueError('Rescales parameters is not a list')
        if not all(isinstance(r, float) for r in rescales):
            raise ValueError('Rescales should be floats')
        if len(rescales) != self.number_of_lanes:
            raise ValueError('Length of rescales does not equal number of lanes')
        if l1 < 0 or l1 >= self.number_of_lanes or l2 < 0 or l2 >= self.number_of_lanes:
            raise ValueError('SuperLayer identifiers out of bounds')
        return self.model_factory.build_multi_lane([min(l1, l2), max(l1, l2)],
                                                   rescales=[rescales[min(l1, l2)], rescales[max(l1, l2)]])

    def get_all_lanes_with_xcon(self, rescales, filter_mults, name=''):
        if not isinstance(rescales, list):
            raise ValueError('Rescales parameters is not a list')
        if not all(isinstance(r, float) for r in rescales):
            raise ValueError('Rescales should be floats')
        if len(rescales) != self.number_of_lanes:
            raise ValueError('Length of rescales does not equal number of lanes')
        lids = list(range(self.number_of_lanes))
        params = filter_mults, self.use_bn, self.connection_factory
        model_name = self.base_model_name if name == '' else name
        return self.model_factory.build_multi_lane(lids, rescales, modelname=model_name, params=params)

    def get_xpair(self, l1, l2, rescales, filter_mults):
        if not isinstance(rescales, list):
            raise ValueError('Rescales parameters is not a list')
        if not all(isinstance(r, float) for r in rescales):
            raise ValueError('Rescales should be floats')
        if len(rescales) != self.number_of_lanes:
            raise ValueError('Length of rescales does not equal number of lanes')
        if l1 < 0 or l1 >= self.number_of_lanes or l2 < 0 or l2 >= self.number_of_lanes:
            raise ValueError('SuperLayer identifiers out of bounds')
        if l1 not in filter_mults or l2 not in filter_mults or l1 not in filter_mults[l2] or l2 not in filter_mults[l1]:
            raise ValueError('Superlayer identifiers not found in filter multpliers')
        params = filter_mults, self.use_bn, self.connection_factory
        return self.model_factory.build_multi_lane([min(l1, l2), max(l1, l2)],
                                                   rescales=[rescales[min(l1, l2)], rescales[max(l1, l2)]],
                                                   modelname=None, params=params)

    def set_xspot_strategy(self, strategy=None, layernames=None, filters=None):
        strats = {'after_pooling': ModelBuilder.after_pooling, 'resnet': ModelBuilder.resnet}
        if strategy:
            assert strategy in strats
            strategy_func = strats[strategy]
        elif layernames:
            if isinstance(layernames, str):
                layernames = [layernames]
            assert isinstance(layernames, list)
            for name in layernames:
                if not self.model_factory.has_in_main(name):
                    raise ValueError('Layer {} not in the mainpart of the model'.format(name))
            strategy_func = ModelBuilder.layername(layernames, filters)  # build a layername strategy
        else:
            raise ValueError('Specify either stategy from {} or layer names'.format(list(strats)))
        logger.debug(strategy_func)
        self.startegy_func = strategy_func
        self.model_factory.insert_x_spots(strategy_func)
        self.analysed_xcon_pair = False  # invalidates the previous findings

    def set_xcons(self, model_function=None, use_bn=True, **kwargs):
        if model_function:
            self.connection_factory = FunctionalModelFactory(model_function)
        else:
            self.connection_factory = XSpotConFactory(**kwargs)
        self.use_bn = use_bn
        self.analysed_xcon_pair = False  # invalidates the previous findings

    def compile(self, optimizer, loss, **kwargs):
        # optimizer must be str identifier, Optimizer instance or tensorflow optimizer
        if not is_valid_optimizers(optimizer):
            raise ValueError("optimizer must be either string or instance of keras.optimizers.Optimizer class")

        if not is_valid_loss(loss):
            raise ValueError("loss must be either string identifier or objective function; lists and dicts do not work"
                             "for single output")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = kwargs['metrics'] if 'metrics' in kwargs else None
        if self.metrics is None:
            self.metrics = []
        if 'accuracy' not in self.metrics:
            self.metrics.append('accuracy')
        kwargs['metrics'] = self.metrics
        self.compile_kwargs = kwargs
        self.compile_set = True

    def fit(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.fit_kwargs = kwargs
        if ('validation_data' not in self.fit_kwargs and 'validation_split' not in self.fit_kwargs) or \
           ('validation_data' not in self.fit_kwargs and 'validation_split' in self.fit_kwargs and \
            self.fit_kwargs['validation_split'] <=0.):
            raise ValueError("Please provide validation data")
        self.nb_epoch = self.fit_kwargs["nb_epoch"] if "nb_epoch" in self.fit_kwargs else 10
        self.verbose = self.fit_kwargs["verbose"] if "verbose" in self.fit_kwargs else 0
        self.train_set = True

    def checkpoint_train_state(self, model, train_phase, build_phase, epoch):
        assert train_phase in {'training', 'complete'}
        assert build_phase in {'single_{}'.format(i) for i in range(self.number_of_lanes)} | \
                              {'double_{}'.format(i) for i in range(self.number_of_lanes)}
        self.curr_train_model = model
        self.curr_train_phase = train_phase
        self.curr_build_phase = build_phase
        self.curr_epoch = epoch

    def model_train(self, model, from_last=None):
        assert self.compile_set and self.train_set, "Does not have compilation and training parameters"
        model.compile(optimizer=self.optimizer, loss=self.loss, **self.compile_kwargs)
        kwargs = self.fit_kwargs.copy()
        kwargs.update(nb_epoch=self.nb_epoch, verbose=self.verbose)
        hist = model.fit(x=self.x, y=self.y, **kwargs)

        acc = hist.history['val_acc']
        loss = hist.history['val_loss']
        if from_last is None or len(acc)<=1:
            max_acc_i = max(range(len(acc)), key=lambda i: acc[i])
            min_loss_i = min(range(len(loss)), key=lambda i: loss[i])
            assert acc[max_acc_i] == max(acc)
            assert loss[min_loss_i] == min(loss)
        else:
            start_ind = len(acc) - int(from_last * len(acc))
            max_acc_i = max(range(start_ind, len(acc)), key=lambda i: acc[i])
            min_loss_i = min(range(start_ind, len(acc)), key=lambda i: loss[i])

        logger.info("Max acc: {} at epoch {} (loss {})".format(acc[max_acc_i], max_acc_i, loss[max_acc_i]))
        return acc[max_acc_i], loss[min_loss_i]

    def set_lane_scale(self, lane_scale=1.0):
        base_scale = 1. / self.number_of_lanes
        tp, ntp = get_params(self.base_model)
        self.base_scale = base_scale
        self.lane_scale = lane_scale
        self.tp = tp
        self.analysed_base_scaling = True
        logger.info('Lane Scale: {}'.format(self.lane_scale))
        self.save_state()

    def analyse_lane_scale(self):
        # calculate scaling of one lane to match target parameter count
        sign = lambda x: 1 if x >= 0 else -1
        base_scale = 1. / self.number_of_lanes
        lane_scale = None
        tp, ntp = get_params(self.base_model)
        target_tp = tp * base_scale
        a = 1.0
        b = 2.0

        # f = lambda x:  get_params(self.get_single_lane(0, rescale=base_scale * x))[0] - target_tp
        def f(x):
            m = self.get_single_lane(0, rescale=base_scale * x)
            tmp, tnp = get_params(m)
            logger.debug("Lane Scaling: target tp {}; currently {}; trying {}".format(target_tp, tmp, x))
            return tmp - target_tp, tmp

        fa, _ = f(a)
        fb, _ = f(b)
        sfa = sign(fa)
        sfb = sign(fb)
        iter = 50
        TOL = 0.00001
        while iter > 0:
            c = (b + a) / 2.
            fc, tmp = f(c)
            if fc == 0. or (b - a) / 2. < TOL or (tmp / target_tp >= 0.98 and tmp / target_tp < 1.02):
                lane_scale = c
                break  # c is the point
            else:
                iter -= 1
                sfc = sign(fc)
                if sfa == sfb:
                    lane_scale = b
                    break  # have a plateau interval or there is critical point in the interval take b as the answer
                if sfc == sfa:
                    a = c
                    fa = fc
                    sfa = sfc
                else:
                    b = c
                    fb = fc
                    sfb = sfc

        assert lane_scale is not None
        self.set_lane_scale(lane_scale)

    def set_single_superlayer_accs(self, accs: list):
        if len(accs) != self.number_of_lanes:
            raise ValueError('Wrong number of accuracies')
        try:
            accs = list(map(float, accs))
        except TypeError as e:
            raise ValueError("Supplied accuracies were not (could not be converted to) floats")
        self.single_mes = accs
        logger.info("Single measuremets: {}".format(self.single_mes))
        self.analysed_single_superlayer_accuracy = True
        self.save_state()

    def measure_single_superlayer(self):
        if self.curr_mes_res is None and self.curr_mes_completed_states is None:
            self.curr_mes = 'single'
            self.curr_mes_res = []
            self.curr_mes_completed_states = []

        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and training parameters')
        if not self.analysed_base_scaling:
            self.analyse_lane_scale()

        measurements_secondary = []
        for lid in range(self.number_of_lanes):
            if lid in self.curr_mes_completed_states:
                continue  # the measurement was made
            model = self.get_single_lane(lid, rescale=self.base_scale * self.lane_scale)
            acc, loss = self.model_train(model)
            self.curr_mes_res.append(acc)
            measurements_secondary.append(loss)
            self.curr_mes_completed_states.append(lid)
            self.save_state()

        # commit results
        self.single_mes2 = measurements_secondary
        self.set_single_superlayer_accs(self.curr_mes_res)

        # reset to non-measurment state
        self.curr_mes = None
        self.curr_mes_res = None
        self.curr_mes_completed_states = None
        self.save_state()

    def measure_xpair_superlayers(self, rescales, filter_mults):
        if self.curr_mes_res is None and self.curr_mes_completed_states is None:
            self.curr_mes = 'xpair'
            self.curr_mes_res = dict()
            self.curr_mes_completed_states = []

        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and training parameters')
        if not self.analysed_xcon_pair:

            measurements_secondary = dict()
            for l1, l2 in in_pairs(self.number_of_lanes):
                if (l1, l2) in self.curr_mes_completed_states:
                    continue
                if l1 not in self.curr_mes_res:
                    self.curr_mes_res[l1] = dict()
                    measurements_secondary[l1] = dict()
                if l2 not in self.curr_mes_res:
                    self.curr_mes_res[l2] = dict()
                    measurements_secondary[l2] = dict()
                model = self.get_xpair(l1, l2, rescales=rescales, filter_mults=filter_mults)
                acc, loss = self.model_train(model)
                self.curr_mes_res[l1][l2] = acc
                self.curr_mes_res[l2][l1] = acc

                measurements_secondary[l1][l2] = loss
                measurements_secondary[l2][l1] = loss
                self.save_state()

            self.xpair_mes = self.curr_mes_res
            self.xpari_mes2 = measurements_secondary
            # print(self.xpair_mes)
            self.analysed_xcon_pair = True

        # reset to non-measurment state
        self.curr_mes = None
        self.curr_mes_res = None
        self.curr_mes_completed_states = None
        self.save_state()

    def build(self):
        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and taining parameters')

        if not self.analysed_base_scaling:
            self.analyse_lane_scale()

        if not self.analysed_single_superlayer_accuracy:
            self.measure_single_superlayer()

        # per_superlayer_scaling = [x / sum(self.single_mes) * self.lane_scale for x in self.single_mes]


        # self.per_sp_scaling = [x / sum(self.single_mes) * self.lane_scale for x in self.single_mes]
        self.per_sp_scaling = scale_func(self.single_mes, self.lane_scale, self.alpha)
        self.single_filter_mults = dict()
        for l1, l2 in in_pairs(self.number_of_lanes):
            if l1 not in self.single_filter_mults:
                self.single_filter_mults[l1] = dict()
            if l2 not in self.single_filter_mults:
                self.single_filter_mults[l2] = dict()

            # l1->l2 filtercount
            nbf_12 = filter_func2(l1, l2, [1 for n in range(self.number_of_lanes)], self.single_mes, self.beta)
            self.single_filter_mults[l1][l2] = nbf_12

            # l2->l1 filtercount
            nbf_21 = filter_func2(l2, l1, [1 for n in range(self.number_of_lanes)], self.single_mes, self.beta)
            self.single_filter_mults[l2][l1] = nbf_21

        final_model_name = self.base_model_name + '_s'
        final_model = self.get_all_lanes_with_xcon(self.per_sp_scaling, self.single_filter_mults,
                                                   name=final_model_name)
        self.single_model_tp, _ = get_params(final_model)
        self.save_state()
        return final_model


    @DeprecationWarning
    def build_scaled_double_xcon(self):
        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and taining parameters')

        if len(self.model_factory.get_xspot_before_layernames()) == 0:
            raise ValueError(
                'The model contains no X-spots, consider changing strategy with XCNNBuilder.set_xcon_strategy')

        if not self.analysed_base_scaling:
            self.analyse_lane_scale()

        if not self.analysed_single_superlayer_accuracy:
            self.measure_single_superlayer()
        self.build()


        if not self.analysed_xcon_pair:
            self.measure_xpair_superlayers(self.per_sp_scaling, self.single_filter_mults)

        pair_accs = [(l1, l2, self.xpair_mes[l1][l2]) for l1, l2 in in_pairs(self.number_of_lanes)]

        self.double_filter_mults = dict()
        for l1, l2 in in_pairs(self.number_of_lanes):
            if l1 not in self.double_filter_mults:
                self.double_filter_mults[l1] = dict()
            if l2 not in self.double_filter_mults:
                self.double_filter_mults[l2] = dict()

            nbf_12 = second_filter_func2(l1, l2, [1 for n in range(self.number_of_lanes)], self.single_mes, pair_accs,
                                         self.beta)
            self.double_filter_mults[l1][l2] = nbf_12

            nbf_21 = second_filter_func2(l2, l1, [1 for n in range(self.number_of_lanes)], self.single_mes, pair_accs,
                                         self.beta)
            self.double_filter_mults[l2][l1] = nbf_21

        final_model_name = self.base_model_name + '_d'
        final_model = self.get_all_lanes_with_xcon(self.per_sp_scaling, self.double_filter_mults,
                                                   name=final_model_name)
        self.double_model_tp, _ = get_params(final_model)
        self.save_state()
        return final_model

    @DeprecationWarning
    def build_scaled_iter_boosting(self, w_cutoff=.1, iter_max=4):
        import math
        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and taining parameters')

        if len(self.model_factory.get_xspot_before_layernames())==0:
            raise ValueError('The model contains no X-spots, consider changing strategy with XCNNBuilder.set_xcon_strategy')

        if not self.analysed_base_scaling:
            self.analyse_lane_scale()

        if not self.analysed_single_superlayer_accuracy:
            self.measure_single_superlayer()
        self.build()

        # start with 0 measurement - pairs with no x_cons:
        base_zero_filt_counts = dict()
        for l1, l2 in in_pairs(self.number_of_lanes):
            if l1 not in base_zero_filt_counts:
                base_zero_filt_counts[l1] = dict()
            if l2 not in base_zero_filt_counts:
                base_zero_filt_counts[l2] = dict()
            base_zero_filt_counts[l1][l2] = 0
            base_zero_filt_counts[l2][l1] = 0

        prev_measure = self.measure_xpair_sps_iter(self.per_sp_scaling, base_zero_filt_counts)
        curr_measure = self.measure_xpair_sps_iter(self.per_sp_scaling, self.single_filter_mults)
        new_filt_mults = self.single_filter_mults
        for i in range(iter_max):
            # calculate gains

            hist_gains = []
            prod_gain = 1.
            count = 0
            for l1, l2 in in_pairs(self.number_of_lanes):
                gain = float(prev_measure[l1][l2]) / float(curr_measure[l1][l2])
                prod_gain *= gain
                hist_gains.append((l1, l2, gain))
                count += 1

            norm_gain = math.pow(prod_gain, 1./ float(count))

            # calculate new_filter mults
            new_filt_mults = dict()

            for l1, l2, gain in hist_gains:
                if l1 not in new_filt_mults:
                    new_filt_mults[l1] = dict()
                if l2 not in new_filt_mults:
                    new_filt_mults[l2] = dict()

                nbf_12 = filter_func2(l1, l2, [], self.single_mes, self.beta) * gain / norm_gain
                nbf_12 = 0 if nbf_12 < w_cutoff else nbf_12

                new_filt_mults[l1][l2] = nbf_12



                nbf_21 = filter_func2(l2, l1, [], self.single_mes, self.beta) * gain / norm_gain
                nbf_21 = 0 if nbf_21 < w_cutoff else nbf_21

                new_filt_mults[l2][l1] = nbf_21
                logger.info("{} ::| {} :|: {} :> {} <: {}".format(i, l1, l2, nbf_12, nbf_21))
            prev_measure = curr_measure
            curr_measure = self.measure_xpair_sps_iter(self.per_sp_scaling, new_filt_mults)

        final_model_name = self.base_model_name + '_i'
        final_model = self.get_all_lanes_with_xcon(self.per_sp_scaling, new_filt_mults,
                                                   name=final_model_name)

        return final_model

    def _get_con_filt(self, l1, l2, filters):
        xspot_filters = [(l, l.base_nb_filter) for l in self.model_factory.main_superlayer.xspots]
        per_lane_spot_filters = []
        per_lane_spot_filters.extend([cf(
            filters[l1][l2] * cf(l.trace_filt() * self.per_sp_scaling[l1] * self.lane_scale)) if x == -1 else cf(
            filters[l1][l2] * x) for l, x in xspot_filters])
        per_lane_spot_filters.extend([cf(
            filters[l2][l1] * cf(l.trace_filt() * self.per_sp_scaling[l2] * self.lane_scale)) if x == -1 else cf(
            filters[l2][l1] * x) for l, x in xspot_filters])
        return per_lane_spot_filters

    def _comp_con_filts(self, filts1, filts2):
        if len(filts1)==len(filts2):
            for x,y in zip(filts1, filts2):
                if x!=y:
                    return False
        return True

    def build_iter(self, learning_rate, Nasterov=True, iter_max=5, rep=2, final_weight_transfer=True):
        import math
        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and taining parameters')

        if len(self.model_factory.get_xspot_before_layernames()) == 0:
            raise ValueError(
                'The model contains no X-spots, consider changing strategy with XCNNBuilder.set_xcon_strategy')

        if not self.analysed_base_scaling:
            self.analyse_lane_scale()

        if not self.analysed_single_superlayer_accuracy:
            self.measure_single_superlayer()
        self.build()

        # start with 0 measurement - pairs with no x_cons:
        base_filt_counts = dict()
        for l1, l2 in in_pairs(self.number_of_lanes):
            if l1 not in base_filt_counts:
                base_filt_counts[l1] = dict()
            if l2 not in base_filt_counts:
                base_filt_counts[l2] = dict()
            base_filt_counts[l1][l2] = .5
            base_filt_counts[l2][l1] = .5

        if final_weight_transfer:
            prev_models = self.get_all_lanes_with_xcon(self.per_sp_scaling, self.single_filter_mults,
                                                       name="pre_init_bootsrap")
            nb_epoch = self.nb_epoch
            self.nb_epoch = max(int(self.nb_epoch // 4),1)
            self.model_train(prev_models)
            prev_models = ModelProxy(prev_models)
            self.nb_epoch = nb_epoch

        #tracked_trough_generations # indexes by [gen][ind]
        measure_gens = []
        filt_mult_gens = []
        model_gens = []
        model_filt_gens= []
        best_gens = [] # indexed by [ind]; tracks (gen) pairs


        def measure_generation(filter_mults, gen, prev_init_models=None):
            next_mes = []
            next_models = []  # new_generation of models
            next_filt_gen = []
            for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
                rep_acc = []
                rep_models = []

                # check if previously had this model!

                curr_filts = self._get_con_filt(l1, l2, filter_mults)
                prev_gen_ind = -1
                if model_filt_gens:
                    for gen_ind in range(len(model_filt_gens)):
                        if self._comp_con_filts(curr_filts, model_filt_gens[gen_ind][ind]):
                            prev_gen_ind = gen_ind
                            break

                if prev_gen_ind==-1:
                    logger.info('Novel generation - performing measurement!')
                    for r in range(rep):
                        model = self.get_xpair(l1, l2, rescales=self.per_sp_scaling, filter_mults=filter_mults)
                        logger.info('Measuring lanes {} {} | {}'.format(l1, l2, gen))
                        # add weights from previous generation
                        if model_gens:
                            prev_model = model_gens[-1][ind]
                            tranfer_with_difference = transfer_weights(model, prev_model)
                            if tranfer_with_difference:
                                print("_________ NON DIFFERENCE ________")
                        elif prev_init_models:
                            prev_model = prev_init_models[ind]
                            tranfer_with_difference = transfer_weights(model, prev_model)
                            if tranfer_with_difference:
                                print("_________ NON DIFFERENCE ________")
                        # v = self.verbose
                        # self.verbose = 0
                        try:
                            acc, loss = self.model_train(model, from_last=.8)
                        except Exception as e:
                            print('EXCEPTION: generation {} | {} {} rep {} |')
                            print('Filters mults:')
                            print(filter_mults)
                            model.summary()
                            raise e

                        rep_acc.append(acc)
                        rep_models.append(ModelProxy(model))
                        # self.verbose = v
                        tf_ses_ctl_reset()
                else:
                    logger.info("Same as previous generation {} for w({},{}) - using old data".format(prev_gen_ind-2,
                                round(filt_mult_gens[prev_gen_ind][l1][l2],4), round(filt_mult_gens[prev_gen_ind][l2][l1],4)))
                    rep_acc.append(measure_gens[prev_gen_ind][ind])
                    rep_models.append(model_gens[prev_gen_ind][ind])

                acc = sum(rep_acc) / len(rep_acc)
                logger.info("acc {}".format(acc))

                next_mes.append(acc)

                if len(rep_models) > 1:
                    fin_model = self.get_xpair(l1, l2, rescales=self.per_sp_scaling, filter_mults=filter_mults)
                    fin_model = average_weights(fin_model, rep_models)
                    fin_model = ModelProxy(fin_model)
                else:
                    fin_model = rep_models[0]

                next_models.append(fin_model)

                next_filt_gen.append(curr_filts)

                if len(best_gens)==len(in_pairs(self.number_of_lanes)):
                    if acc > measure_gens[best_gens[ind]][ind]:
                        logger.info("{} {} in {} better than {} ({}), updating".format(l1, l2, gen, best_gens[ind],
                                                                                       measure_gens[best_gens[ind]][ind]))
                        best_gens[ind] = gen
                else:
                    best_gens.append(gen)


            measure_gens.append(next_mes)
            filt_mult_gens.append(filter_mults)
            model_gens.append(next_models)
            model_filt_gens.append(next_filt_gen)

            return best_gens

        if final_weight_transfer:
            measure_generation(base_filt_counts, 0, [prev_models for l1, l2 in in_pairs(self.number_of_lanes)])
            measure_generation(base_filt_counts, 0)
            measure_generation(self.single_filter_mults, 1)

        else:
            measure_generation(base_filt_counts, 0)
            measure_generation(self.single_filter_mults, 1)

        # optimisation parameters


        self.iter_build_report = ""
        l = 0.001
        beta1 = 0.9
        beta2 = 0.999
        schedule_decay = 0.004
        eps = 1e-8

        # initialise momentums & velocities
        momentums = dict()
        velocities = dict()
        m_schedule = 1.
        for (l1, l2) in in_pairs(self.number_of_lanes):
            if l1 not in momentums:
                momentums[l1] = dict()
                velocities[l1] = dict()
            if l2 not in momentums:
                momentums[l2] = dict()
                velocities[l2] = dict()

            momentums[l1][l2] = 0.
            velocities[l1][l2] = 0.

            momentums[l2][l1] = 0.
            velocities[l2][l1] = 0.

        for i in range(iter_max):
            new_filt_gen = dict()

            # per iteration opt parameter update


            t = i + 1
            if not Nasterov:
                lr_t = learning_rate * (math.sqrt(1. - math.pow(beta2, t)) /
                                        (1. - math.pow(beta1, t)))
            else:
                momentum_cache_t = beta1 * (1. - 0.5 * (math.pow(0.96, t * schedule_decay)))
                momentum_cache_t_1 = beta1 * (1. - 0.5 * (math.pow(0.96, (t + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                m_schedule = m_schedule_new

            for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
                # get derivatives
                d_mes = measure_gens[-1][ind] - measure_gens[-2][ind]
                d_nbf_12 = filt_mult_gens[-1][l1][l2] - filt_mult_gens[-2][l1][l2]
                d_nbf_21 = filt_mult_gens[-1][l2][l1] - filt_mult_gens[-2][l2][l1]
                # include weight decay to prefer lower weights
                d_mes_dnbf_12 = (d_mes - l * filt_mult_gens[-1][l1][l2] ** 2 + l * filt_mult_gens[-2][l1][l2] ** 2) / (
                d_nbf_12) if d_nbf_12 != 0. else 0.
                d_mes_dnbf_21 = (d_mes - l * filt_mult_gens[-1][l2][l1] ** 2 + l * filt_mult_gens[-2][l2][l1] ** 2) / (
                d_nbf_21) if d_nbf_21 != 0. else 0.

                if l1 not in new_filt_gen:
                    new_filt_gen[l1] = dict()
                if l2 not in new_filt_gen:
                    new_filt_gen[l2] = dict()

                if not Nasterov:

                    # ADAM

                    momentums[l1][l2] = beta1 * momentums[l1][l2] + (1 - beta1) * d_mes_dnbf_12
                    velocities[l1][l2] = beta2 * velocities[l1][l2] + (1 - beta2) * (d_mes_dnbf_12 ** 2)
                    new_filt_gen[l1][l2] = filt_mult_gens[-1][l1][l2] + lr_t * momentums[l1][l2] / (
                    math.sqrt(velocities[l1][l2]) + eps)

                    momentums[l2][l1] = beta1 * momentums[l2][l1] + (1 - beta1) * d_mes_dnbf_21
                    velocities[l2][l1] = beta2 * velocities[l2][l1] + (1 - beta2) * (d_mes_dnbf_21 ** 2)
                    new_filt_gen[l2][l1] = filt_mult_gens[-1][l2][l1] + lr_t * momentums[l2][l1] / (
                    math.sqrt(velocities[l2][l1]) + eps)
                else:
                    # Nasterov ADAM:
                    # following equations in
                    # Dozat, T., 2015. Incorporating Nesterov momentum into Adam.
                    # Stanford University, Tech. Rep., 2015.[Online].
                    # Available: http://cs229.stanford.edu/proj2015/054 report. pdf.

                    g_prime = d_mes_dnbf_12 / (1. - m_schedule_new)
                    m_t = beta1 * momentums[l1][l2] + (1. - beta1) * d_mes_dnbf_12
                    m_t_prime = m_t / (1. - m_schedule_next)
                    v_t = beta2 * velocities[l1][l2] + (1. - beta2) * (d_mes_dnbf_12 ** 2)
                    v_t_prime = v_t / (1. - math.pow(beta2, t))
                    m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

                    new_filt_gen[l1][l2] = filt_mult_gens[-1][l1][l2] + learning_rate * m_t_bar / (
                    math.sqrt(v_t_prime) + eps)

                    g_prime = d_mes_dnbf_21 / (1. - m_schedule_new)
                    m_t = beta1 * momentums[l2][l1] + (1. - beta1) * d_mes_dnbf_12
                    m_t_prime = m_t / (1. - m_schedule_next)
                    v_t = beta2 * velocities[l2][l1] + (1. - beta2) * (d_mes_dnbf_12 ** 2)
                    v_t_prime = v_t / (1. - math.pow(beta2, t))
                    m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

                    new_filt_gen[l2][l1] = filt_mult_gens[-1][l2][l1] + learning_rate * m_t_bar / (
                    math.sqrt(v_t_prime) + eps)

                # constaint weights to stay positive
                new_filt_gen[l1][l2] = max(new_filt_gen[l1][l2], 0.)
                new_filt_gen[l2][l1] = max(new_filt_gen[l2][l1], 0.)

                logger.info("{}/{} |{} {} |O: {} | N: {} | w_o ({},{}), w_n ({}, {}), d ({},{})".format(
                    i + 1, iter_max, l1, l2,
                    round(measure_gens[-2][ind], 4), round(measure_gens[-1][ind], 4),
                    round(filt_mult_gens[-1][l1][l2], 4), round(filt_mult_gens[-1][l2][l1], 4),
                    round(new_filt_gen[l1][l2], 4), round(new_filt_gen[l2][l1], 4),
                    round(d_mes_dnbf_12, 4), round(d_mes_dnbf_21, 4)
                ))

            measure_generation(new_filt_gen, i+2)

        self.iter_build_report = ""
        for gen in range(len(measure_gens)):
            for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
                line = ("{}/{} | {} {} | ACC: {} |  W ({},{})".format(
                    gen - 1, iter_max, l1, l2,
                    round(measure_gens[gen][ind], 4),
                    round(filt_mult_gens[gen][l1][l2], 4), round(filt_mult_gens[gen][l2][l1], 4)
                ))
                logger.info(line)
                self.iter_build_report += line + '\n'

        best_models = []
        final_filt_counts = dict()
        for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
            if l1 not in final_filt_counts:
                final_filt_counts[l1] = dict()
            if l2 not in final_filt_counts:
                final_filt_counts[l2] = dict()

            best_gen = best_gens[ind]
            logger.info("Best gen for {} {} is {} ({})".format(l1, l2, best_gen, measure_gens[best_gen][ind]))
            self.iter_build_report += "Best gen for {} {} is {} ({})\n".format(l1, l2, best_gen,
                                                                               measure_gens[best_gen][ind])
            final_filt_counts[l1][l2] = filt_mult_gens[best_gen][l1][l2]
            final_filt_counts[l2][l1] = filt_mult_gens[best_gen][l2][l1]
            best_models.append(model_gens[best_gen][ind])

        self.iter_build_filt_mults = final_filt_counts
        tf_ses_ctl_reset()
        final_model_name = self.base_model_name + '_i'
        final_model = self.get_all_lanes_with_xcon(self.per_sp_scaling, final_filt_counts,
                                                   name=final_model_name)
        self.iter_model_tp, _ = get_params(final_model)
        self.iter_build = True
        if final_weight_transfer:
            final_model = average_transfer_weights(final_model, best_models, verbose=self.verbose)
        return final_model

    def build_iter_global(self, learning_rate, Nasterov=True, iter_max=5, rep=2, final_weight_transfer=True):
        import math
        if not self.compile_set or not self.train_set:
            raise ValueError('Please specify compilation and taining parameters')

        if len(self.model_factory.get_xspot_before_layernames()) == 0:
            raise ValueError(
                'The model contains no X-spots, consider changing strategy with XCNNBuilder.set_xcon_strategy')

        if not self.analysed_base_scaling:
            self.analyse_lane_scale()

        if not self.analysed_single_superlayer_accuracy:
            self.measure_single_superlayer()
        self.build()

        # start with 0 measurement - pairs with no x_cons:
        base_filt_counts = dict()
        for l1, l2 in in_pairs(self.number_of_lanes):
            if l1 not in base_filt_counts:
                base_filt_counts[l1] = dict()
            if l2 not in base_filt_counts:
                base_filt_counts[l2] = dict()
            base_filt_counts[l1][l2] = .5
            base_filt_counts[l2][l1] = .5

        if final_weight_transfer:
            prev_models = self.get_all_lanes_with_xcon(self.per_sp_scaling, self.single_filter_mults,
                                                       name="pre_init_bootsrap")
            nb_epoch = self.nb_epoch
            self.nb_epoch = int(self.nb_epoch // 4)
            self.model_train(prev_models)
            prev_models = ModelProxy(prev_models)
            self.nb_epoch = nb_epoch

        def get_measure(rescales, filter_mults, gen, prev_models=[], best_measures=[], best_models=[]):
            res = []
            next_models = []  # new_generation of models

            for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
                rep_acc = []
                rep_models = []
                for r in range(rep):
                    model = self.get_xpair(l1, l2, rescales=rescales, filter_mults=filter_mults)
                    logger.info('Measuring lanes {} {} | {}'.format(l1, l2, gen))
                    # add weights from previous generation
                    if prev_models:
                        flat_prev_models = []
                        for m in prev_models:
                            if isinstance(m, (list, tuple)):
                                flat_prev_models.extend(m)
                            else:
                                flat_prev_models.append(m)
                        for m in flat_prev_models:
                            assert isinstance(m, ModelProxy)
                        #average across all pair where this modality was used training the same weights
                        model = average_transfer_weights(model, flat_prev_models, verbose=self.verbose)
                    # v = self.verbose
                    # self.verbose = 0
                    try:
                        acc, loss = self.model_train(model, from_last=.8)
                    except Exception as e:
                        print('EXCEPTION: generation {} | {} {} rep {} |')
                        print('Filters mults:')
                        print(filter_mults)
                        model.summary()
                        raise e

                    rep_acc.append(acc)
                    rep_models.append(ModelProxy(model))
                    # self.verbose = v
                    tf_ses_ctl_reset()

                acc = sum(rep_acc) / len(rep_acc)
                res.append(acc)
                if len(rep_models) > 1:
                    fin_model = self.get_xpair(l1, l2, rescales=rescales, filter_mults=filter_mults)
                    fin_model = average_weights(fin_model, rep_models)
                    fin_model = ModelProxy(fin_model)
                else:
                    fin_model = rep_models[0]
                next_models.append(fin_model)
                if acc > best_measures[ind][0]:
                    logger.info("{} {} in {} better than {} ({}), updating".format(l1, l2, gen, best_measures[ind][1],
                                                                                   best_measures[ind][0]))
                    best_measures[ind] = (acc, gen)
                    best_models[ind] = fin_model
            return res, next_models, best_measures, best_models

        if final_weight_transfer:
            prev_measure, prev_models, best_measures, best_models = get_measure(self.per_sp_scaling, base_filt_counts,
                                                                                gen='base',
                                                                                prev_models=[prev_models for l1, l2 in
                                                                                             in_pairs(
                                                                                                 self.number_of_lanes)],
                                                                                best_measures=[(.0, None) for l1, l2 in
                                                                                               in_pairs(
                                                                                                   self.number_of_lanes)],
                                                                                best_models=[None for l1, l2 in
                                                                                               in_pairs(
                                                                                                   self.number_of_lanes)])
            curr_measure, curr_models, best_measures, best_models = get_measure(self.per_sp_scaling,
                                                                                self.single_filter_mults, gen='single',
                                                                                prev_models=prev_models,
                                                                                best_measures=best_measures,
                                                                                best_models=best_models)
        else:
            prev_measure, prev_models, best_measures, best_models = get_measure(self.per_sp_scaling, base_filt_counts,
                                                                                gen='base',
                                                                                best_measures=[.0 for l1, l2 in
                                                                                               in_pairs(
                                                                                                   self.number_of_lanes)],
                                                                                best_models=[None for l1, l2 in
                                                                                             in_pairs(
                                                                                                 self.number_of_lanes)])
            curr_measure, curr_models, best_measures, best_models = get_measure(self.per_sp_scaling,
                                                                                self.single_filter_mults, gen='single',
                                                                                prev_models=prev_models,
                                                                                best_measures=best_measures,
                                                                                best_models=best_models)
        # prev_measure, prev_models = [.331125, .3505, .25687], []
        # curr_measure, curr_models = [.3385, .34725, 0.259375], []
        filt_mult_generations = [base_filt_counts, self.single_filter_mults]  # this is cheap mem-wise
        measure_generations = [prev_measure, curr_measure]
        model_generations = [prev_models, curr_models]

        # optimisation parameters


        self.iter_build_report = ""

        beta1 = 0.9
        beta2 = 0.999
        schedule_decay = 0.004
        eps = 1e-8

        # initialise momentums & velocities
        momentums = dict()
        velocities = dict()
        m_schedule = 1.
        for (l1, l2) in in_pairs(self.number_of_lanes):
            if l1 not in momentums:
                momentums[l1] = dict()
                velocities[l1] = dict()
            if l2 not in momentums:
                momentums[l2] = dict()
                velocities[l2] = dict()

            momentums[l1][l2] = 0.
            velocities[l1][l2] = 0.

            momentums[l2][l1] = 0.
            velocities[l2][l1] = 0.

        for i in range(iter_max):
            new_filt_gen = dict()

            # per iteration opt parameter update


            t = i + 1
            if not Nasterov:
                lr_t = learning_rate * (math.sqrt(1. - math.pow(beta2, t)) /
                                        (1. - math.pow(beta1, t)))
            else:
                momentum_cache_t = beta1 * (1. - 0.5 * (math.pow(0.96, t * schedule_decay)))
                momentum_cache_t_1 = beta1 * (1. - 0.5 * (math.pow(0.96, (t + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                m_schedule = m_schedule_new

            for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
                # get derivatives
                d_mes = measure_generations[-1][ind] - measure_generations[-2][ind]
                d_nbf_12 = filt_mult_generations[-1][l1][l2] - filt_mult_generations[-2][l1][l2]
                d_nbf_21 = filt_mult_generations[-1][l2][l1] - filt_mult_generations[-2][l2][l1]
                d_mes_dnbf_12 = d_mes / (d_nbf_12) if d_nbf_12 != 0. else 0.
                d_mes_dnbf_21 = d_mes / (d_nbf_21) if d_nbf_21 != 0. else 0.

                if l1 not in new_filt_gen:
                    new_filt_gen[l1] = dict()
                if l2 not in new_filt_gen:
                    new_filt_gen[l2] = dict()

                if not Nasterov:

                    # ADAM

                    momentums[l1][l2] = beta1 * momentums[l1][l2] + (1 - beta1) * d_mes_dnbf_12
                    velocities[l1][l2] = beta2 * velocities[l1][l2] + (1 - beta2) * (d_mes_dnbf_12 ** 2)
                    new_filt_gen[l1][l2] = filt_mult_generations[-1][l1][l2] + lr_t * momentums[l1][l2] / (
                    math.sqrt(velocities[l1][l2]) + eps)

                    momentums[l2][l1] = beta1 * momentums[l2][l1] + (1 - beta1) * d_mes_dnbf_21
                    velocities[l2][l1] = beta2 * velocities[l2][l1] + (1 - beta2) * (d_mes_dnbf_21 ** 2)
                    new_filt_gen[l2][l1] = filt_mult_generations[-1][l2][l1] + lr_t * momentums[l2][l1] / (
                    math.sqrt(velocities[l2][l1]) + eps)
                else:
                    # Nasterov ADAM for weight gradient ascent:
                    # following equations in
                    # Dozat, T., 2015. Incorporating Nesterov momentum into Adam.
                    # Stanford University, Tech. Rep., 2015.[Online].
                    # Available: http://cs229.stanford.edu/proj2015/054 report. pdf.

                    g_prime = d_mes_dnbf_12 / (1. - m_schedule_new)
                    m_t = beta1 * momentums[l1][l2] + (1. - beta1) * d_mes_dnbf_12
                    m_t_prime = m_t / (1. - m_schedule_next)
                    v_t = beta2 * velocities[l1][l2] + (1. - beta2) * (d_mes_dnbf_12 ** 2)
                    v_t_prime = v_t / (1. - math.pow(beta2, t))
                    m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

                    new_filt_gen[l1][l2] = filt_mult_generations[-1][l1][l2] + learning_rate * m_t_bar / (
                    math.sqrt(v_t_prime) + eps)

                    g_prime = d_mes_dnbf_21 / (1. - m_schedule_new)
                    m_t = beta1 * momentums[l2][l1] + (1. - beta1) * d_mes_dnbf_12
                    m_t_prime = m_t / (1. - m_schedule_next)
                    v_t = beta2 * velocities[l2][l1] + (1. - beta2) * (d_mes_dnbf_12 ** 2)
                    v_t_prime = v_t / (1. - math.pow(beta2, t))
                    m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

                    new_filt_gen[l2][l1] = filt_mult_generations[-1][l2][l1] + learning_rate * m_t_bar / (
                    math.sqrt(v_t_prime) + eps)

                # constaint weights to stay positive
                new_filt_gen[l1][l2] = max(new_filt_gen[l1][l2], 0.)
                new_filt_gen[l2][l1] = max(new_filt_gen[l2][l1], 0.)

                logger.info("{}/{} |{} {} |O: {} | N: {} | w_o ({},{}), w_n ({}, {}), d ({},{})".format(
                    i + 1, iter_max, l1, l2,
                    round(measure_generations[-2][ind], 4), round(measure_generations[-1][ind], 4),
                    round(filt_mult_generations[-1][l1][l2], 4), round(filt_mult_generations[-1][l2][l1], 4),
                    round(new_filt_gen[l1][l2], 4), round(new_filt_gen[l2][l1], 4),
                    round(d_mes_dnbf_12, 4), round(d_mes_dnbf_21, 4)
                ))

            new_measure, prev_models, best_measures, best_models = get_measure(self.per_sp_scaling, new_filt_gen, gen=i,
                                                                               prev_models=model_generations,
                                                                               best_measures=best_measures,
                                                                               best_models=best_models)

            filt_mult_generations.append(new_filt_gen)
            measure_generations.append(new_measure)
            model_generations.append(prev_models)

        self.iter_build_report = ""
        for gen in range(len(measure_generations)):
            for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
                line = ("{}/{} | {} {} | ACC: {} |  W ({},{})".format(
                    gen - 1, iter_max, l1, l2,
                    round(measure_generations[gen][ind], 4),
                    round(filt_mult_generations[gen][l1][l2], 4), round(filt_mult_generations[gen][l2][l1], 4)
                ))
                logger.info(line)
                self.iter_build_report += line + '\n'

        final_filt_counts = dict()
        best_gens = []
        best_models = []
        for ind, (l1, l2) in enumerate(in_pairs(self.number_of_lanes)):
            if l1 not in final_filt_counts:
                final_filt_counts[l1] = dict()
            if l2 not in final_filt_counts:
                final_filt_counts[l2] = dict()

            best_gen = 0
            for i in range(len(measure_generations)):
                if measure_generations[i][ind] > measure_generations[best_gen][ind]:
                    best_gen = i  # find a best generation based on highest measure for that pair
            logger.info("Best gen for {} {} is {} ({})".format(l1, l2, best_gen, measure_generations[best_gen][ind]))
            self.iter_build_report += "Best gen for {} {} is {} ({})\n".format(l1, l2, best_gen,
                                                                               measure_generations[best_gen][ind])
            final_filt_counts[l1][l2] = filt_mult_generations[best_gen][l1][l2]
            final_filt_counts[l2][l1] = filt_mult_generations[best_gen][l2][l1]
            best_gens.append(best_gen)
            best_models.append(model_generations[best_gen][ind])

        self.iter_build_filt_mults = final_filt_counts
        tf_ses_ctl_reset()
        final_model_name = self.base_model_name + '_i'
        final_model = self.get_all_lanes_with_xcon(self.per_sp_scaling, final_filt_counts,
                                                   name=final_model_name)
        self.iter_model_tp, _ = get_params(final_model)
        self.iter_build = True
        if final_weight_transfer:

            final_tranfered_model = self.get_all_lanes_with_xcon(self.per_sp_scaling, final_filt_counts,
                                                                 name=final_model_name)
            final_tranfered_model = average_transfer_weights(final_tranfered_model, best_models, verbose=1)
            return final_tranfered_model, final_model

        return final_model

    def print_report(self):
        print(self.report())

    def report(self):
        out = []

        def print(*args):
            out.append(" ".join(str(i) for i in args))

        print('Hypers: alpha {}; beta {}'.format(self.alpha, self.beta))
        print('Number of lanes:', self.number_of_lanes,
              [l.get_name() for l, _, _ in self.model_factory.input_superlayer.outputs])
        if self.analysed_base_scaling:
            print('Total parameters:', self.tp)
            print('Base scaling for lanes:', self.base_scale)
            print('Parameter match scaling:', self.lane_scale)
            print('---------')
        if self.analysed_single_superlayer_accuracy:
            xspot_filters = [(l, l.base_nb_filter) for l in self.model_factory.main_superlayer.xspots]
            print(xspot_filters)
            for lane_id in range(self.number_of_lanes):
                print("Lane {}| Base Acc: {}| Scale: {}".format(lane_id, round(self.single_mes[lane_id], 4),
                                                                round(self.per_sp_scaling[lane_id], 4)))
                for lid in range(self.number_of_lanes):
                    if lid != lane_id:
                        nbf = self.single_filter_mults[lane_id][lid]
                        print("    {}->{}:{}".format(lane_id, lid, round(nbf, 4)),
                              [cf(nbf * cf(l.trace_filt() * self.per_sp_scaling[lane_id] * self.lane_scale)) if x == -1
                               else cf(nbf * x) for l, x in xspot_filters])
            print('Single model params:', self.single_model_tp,
                  '({} %)'.format(round(100 * self.single_model_tp / self.tp, 4)))
            print('---------')
        if self.analysed_xcon_pair:
            xspot_filters = [l.base_nb_filter for l in self.model_factory.main_superlayer.xspots]
            pair_accs = [(l1, l2, self.xpair_mes[l1][l2]) for l1, l2 in in_pairs(self.number_of_lanes)]
            for lid1, lid2, acc in pair_accs:
                mlid = min(lid1, lid2)
                plid = max(lid1, lid2)
                nbf_12 = self.double_filter_mults[mlid][plid]
                nbf_21 = self.double_filter_mults[plid][mlid]
                print('Pair {} {}: Acc {}'.format(mlid, plid, round(acc, 4)))

                print(
                    '{}->{}:{} {}| {}->{}:{} {}'.format(mlid, plid, round(self.double_filter_mults[mlid][plid], 4),
                                                        [cf(nbf_12 * cf(l.trace_filt() * self.per_sp_scaling[
                                                            mlid] * self.lane_scale)) if x == -1
                                                         else cf(nbf_12 * x) for l, x in xspot_filters],
                                                        plid, mlid, round(self.double_filter_mults[plid][mlid], 4),
                                                        [cf(nbf_21 * cf(l.trace_filt() * self.per_sp_scaling[
                                                            plid] * self.lane_scale)) if x == -1
                                                         else cf(nbf_21 * x) for l, x in xspot_filters]))
            print('Double model params:', self.double_model_tp,
                  '({} %)'.format(round(100 * self.double_model_tp / self.tp, 4)))
        print('---------')
        if self.iter_build:
            xspot_filters = [(l, l.base_nb_filter) for l in self.model_factory.main_superlayer.xspots]
            print(self.iter_build_report)
            for l1, l2 in in_pairs(self.number_of_lanes):
                print(
                    '{}->{}:{} {}| {}->{}:{} {}'.format(l1, l2, round(self.iter_build_filt_mults[l1][l2], 4),
                                                        [cf(self.iter_build_filt_mults[l1][l2] * cf(l.trace_filt() * self.per_sp_scaling[
                                                            l1] * self.lane_scale)) if x == -1
                                                         else cf(self.iter_build_filt_mults[l1][l2] * x) for l, x in xspot_filters],
                                                        l2, l1, round(self.iter_build_filt_mults[l2][l1], 4),
                                                        [cf(self.iter_build_filt_mults[l2][l1] * cf(l.trace_filt() * self.per_sp_scaling[
                                                            l2] * self.lane_scale)) if x == -1
                                                         else cf(self.iter_build_filt_mults[l2][l1] * x) for l, x in xspot_filters]))
            print('Iter model params:', self.iter_model_tp,
                  '({} %)'.format(round(100 * self.iter_model_tp / self.tp, 4)))
        return '\n'.join(out)

    def add_persistence(self, path=None):
        self.persistence_path = path
        logger.info("Persistence File: {}".format(self.persistence_path))
        self.save_state()

    def save_state(self):
        if self.persistence_path is None:
            return
        save_dict = dict()
        save_dict['split_axis'] = self.split_axis
        save_dict['number_of_lanes'] = self.number_of_lanes

        # building parameters
        save_dict['startegy_func'] = None  # TODO: load appropriate strategy function serialisation
        save_dict['con_factory'] = None
        save_dict['use_bn'] = self.use_bn

        # analyses results
        save_dict['analysed_base_scaling'] = self.analysed_base_scaling
        if self.analysed_base_scaling:
            save_dict['lane_scale'] = self.lane_scale

        save_dict['analysed_single_superlayer_accuracy'] = self.analysed_single_superlayer_accuracy
        if self.analysed_single_superlayer_accuracy:
            save_dict['single_mes'] = self.single_mes

        save_dict['analysed_xcon_pair'] = self.analysed_xcon_pair
        if self.analysed_xcon_pair:
            save_dict['xpair_mes'] = self.xpair_mes

        # save current measurement state
        save_dict['curr_mes'] = self.curr_mes
        save_dict['curr_mes_res'] = self.curr_mes_res
        save_dict['curr_mes_completed_states'] = self.curr_mes_completed_states

        with open(self.persistence_path, 'w') as out:
            out.write(json.dumps(save_dict))

    def load_state(self, path):
        with open(path, 'r') as inp:
            save_dict = json.load(inp)
        print('State Loaded')
        assert self.split_axis == save_dict['split_axis']
        assert self.number_of_lanes == save_dict['number_of_lanes']
        self.persistence_path = None

        self.use_bn = save_dict['use_bn']

        run_schedule = []

        if save_dict['analysed_base_scaling']:
            self.set_lane_scale(save_dict['lane_scale'])
            run_schedule.append((self.set_lane_scale, [save_dict['lane_scale']]))
            logger.info("restored lane scaling")

        if save_dict['analysed_single_superlayer_accuracy']:
            self.set_single_superlayer_accs(save_dict['single_mes'])
            run_schedule.append((self.set_single_superlayer_accs, [save_dict['single_mes']]))
            _ = self.build()  # back-fill the aux parameters by rebuilding the final model
            run_schedule.append((self.build, []))
            logger.info('restored single superlayer accuracy')

        if save_dict['analysed_xcon_pair']:
            self.xpair_mes = dict()
            for k in save_dict['xpair_mes']:
                self.xpair_mes[int(k)] = dict()
                for k2 in save_dict['xpair_mes'][k]:
                    self.xpair_mes[int(k)][int(k2)] = float(save_dict['xpair_mes'][k][k2])
            self.analysed_xcon_pair = True
            _ = self.build_scaled_double_xcon()
            logger.info('restored double superlayer accuracy')
        run_schedule.append((self.build_scaled_double_xcon, []))

        # restore the current measument/training phase:
        if save_dict['curr_mes'] is not None:
            logger.info('restoring a measurement')
            self.curr_mes = save_dict['curr_mes']
            self.curr_mes_res = save_dict['curr_mes_res']
            self.curr_mes_completed_states = save_dict['curr_mes_completed_states']
            logger.info('running {} skipping {}'.format(self.curr_mes, self.curr_mes_completed_states))
            if self.curr_mes is 'single':
                self.measure_single_superlayer()
            elif self.curr_mes is 'xpair':
                self.build_scaled_double_xcon()

        self.add_persistence(path)


def filter_func(lid1, lid2, priors, accs):
    nlid1 = accs[lid1] / sum(accs)
    nlid2 = accs[lid2] / sum(accs)
    scale = nlid1 / nlid2
    return scale * 1. / (len(priors) - 1)  #


def second_filter_func(lid1, lid2, priors, accs, pair_accs):
    base_nbf = filter_func(lid1, lid2, priors, accs)
    my_pair_acc = 0
    others = []
    for l1, l2, acc in pair_accs:
        if (l1 == lid1 and l2 == lid2) or (l1 == lid2 and l2 == lid1):  # a pair that both lid1 and lid2 are involved in
            my_pair_acc = acc
        elif l1 != lid1 and l2 != lid1:  # a pair that lid1 is not involved in
            others.append(acc)
    return base_nbf * my_pair_acc / (sum(others) / len(others))


def scale_func(mes, lane_scale, alpha=1):
    import math
    x = [math.pow(float(x), alpha) for x in mes]
    return [y * lane_scale / sum(x) for y in x]


def filter_func2(lid1, lid2, priors, accs, beta=2):
    # k =2
    import math
    return math.pow(accs[lid1], beta) / (math.pow(accs[lid1], beta) + math.pow(accs[lid2], beta))


def second_filter_func2(lid1, lid2, priors, accs, pair_accs, beta):
    approx_gains = []
    grp = 1.
    my_gain = 1.
    import math
    for l1, l2, acc in pair_accs:
        g = math.sqrt(acc ** 2 / (accs[l1] * accs[l2]))
        approx_gains.append(g)
        grp *= g
        if (l1 == lid1 and l2 == lid2) or (l1 == lid2 and l2 == lid1):
            my_gain = g
    return filter_func2(lid1, lid2, priors, accs, beta) * my_gain / (math.pow(grp, 1. / len(pair_accs)))


def in_pairs(n):
    # generate (i, j) n(n-1)/2 pairs:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    return pairs


def get_params(model):
    from keras.utils.layer_utils import count_total_params
    if hasattr(model, 'flattened_layers'):
        # Support for legacy Sequential/Merge behavior.
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers
    # (trainable_params, non-trainable_params)
    return count_total_params(flattened_layers, layer_set=None)

def mem_force_free():
    import gc
    col = gc.collect(0)
    print("Collected col")
