from torch.ao.quantization.observer import ObserverBase, UniformQuantizationObserverBase
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization import * 
import torch 

def IS_QSCHEME_PER_CHANNEL(qscheme):
    return qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)
def IS_QSCHEME_AFFINE(qscheme):
    return qscheme in (torch.per_tensor_affine, torch.per_channel_affine)
def IS_QSCHEME_PER_TENSOR(qscheme):
    return not IS_QSCHEME_PER_CHANNEL(qscheme)
def IS_QSCHEME_SYMMETRIC(qscheme):
    return not IS_QSCHEME_AFFINE(qscheme)

def init_mode(mod):
    """
    Enable Init Mode quantization for this module, if applicable. Example usage::
        # model is any PyTorch Model
        model.apply(init_mode)
    """
    if isinstance(mod, LSQFakeQuantize) or _is_fake_quant_script_module(mod):
        assert hasattr(mod, "init_mode"), "This module does not support init mode"
        mod.init_mode()

def training_mode(mod):
    if isinstance(mod, LSQFakeQuantize) or _is_fake_quant_script_module(mod):
        assert hasattr(mod, "init_mode"), "This module does not support training mode"
        mod.training_mode()

def _is_fake_quant_script_module(mod):
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        suffix = mod._c.qualified_name.split('.', 1)[1]
        name = re.sub(r'\.___torch_mangle_\d+', '', suffix)
        return name == 'torch.ao.quantization.fake_quantize.FakeQuantize' or \
                name == 'torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize'
    return False

class LSQObserver(UniformQuantizationObserverBase):
    scale : torch.Tensor
    
    def __init__(
            self,
            dtype = torch.qint8,
            qscheme = torch.per_tensor_affine,
            reduce_range = False,
            quant_min = None,
            quant_max = None,
            factory_kwargs = None,
            eps = torch.finfo(torch.float32).eps,
    ) -> None:
        super(LSQObserver, self).__init__(
                dtype = dtype,
                qscheme = qscheme,
                reduce_range = reduce_range,
                quant_min = quant_min,
                quant_max = quant_max,
                factory_kwargs = factory_kwargs,
                eps = eps,
        )

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype = torch.float))
        if (self.qscheme == torch.per_tensor_symmetric
                and self.reduce_range
                and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                    "Cannot reduce range for symmetric \
                            quantization for quint8"
            )

    def forward(self, x_orig):
        r"""Records Initialize value of scale """
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.scale.dtype)
        _scale = x.abs().mean() * 2 / (self.quant_max ** 0.5)
        self.scale.copy_(_scale)
        return x_orig

    # for initialization
    @torch.jit.export
    def calculate_qparams(self):
        return self.scale
    
    @torch.jit.export
    def extra_repr(self):
        pass
    
    @torch.jit.export
    def reset_min_max_vals(self):
        pass   

class LSQFakeQuantize(FakeQuantizeBase):
    s : torch.Tensor
    zero_point : torch.Tensor

    def __init__(self, observer = LSQObserver, quant_min = None, quant_max = None, **observer_kwargs):
        super().__init__()
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                    'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get("dtype", dtype)
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({"quant_min":quant_min, "quant_max" : quant_max})
        # not needed
        self.activation_post_process = observer(**observer_kwargs)

        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max

        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype = torch.float))
        self.register_buffer("zero_point", torch.tensor([0.0], dtype = torch.int32))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
                if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert IS_QSCHEME_PER_CHANNEL(self.qscheme) or \
                IS_QSCHEME_PER_TENSOR(self.qscheme), \
                'Only per channel and per tensor quantization are fake quantize' + ' got qscheme : ' + str(self.qscheme)
        self.is_per_channel = IS_QSCHEME_PER_CHANNEL(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point

    def forward(self, X):
        # not implemented for observed version
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
            self.scale.data.copy_(_scale)
    
        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                s_grad_scale = 1.0 / ((self.quant_max * X.numel()) ** 0.5)
            else:
                s_grad_scale = 1.0 / ((self.quant_max * X.numel()) ** 0.5)
            s_scale = (self.scale - s_grad_scale * self.scale).detach() + s_grad_scale * self.scale

            X = X / s_scale
            X = torch.clamp(X, self.quant_min, self.quant_max)
            X = (X.round() - X).detach() + X
            X = X * s_scale
        return X
    
    @torch.jit.export
    def init_mode(self):
        # Only one of the Observer and FakeQuant runs.
        self.enable_fake_quant(False)
        self.enable_observer(True)

    @torch.jit.export
    def training_mode(self):
        # Only one of the Observer and FakeQuant runs.
        self.enable_fake_quant(True)
        self.enable_observer(False)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled = {}, observer_enabled={}, '\
                'quant_min = {}, quant_max = {}, dtype = {}, qscheme = {}, ch_axis = {}, '\
                'scale = {}, zero_point ={}'.format(
                        self.fake_quant_enabled, self.observer_enabled, self.activation_post_process.quant_min,
                        self.activation_post_process.quant_max, 
                        self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)

default_affine_weight_fake_quant = LSQFakeQuantize.with_args(observer = LSQObserver, quant_min = -128, quant_max = 127,
                                                      dtype = torch.qint8, qscheme = torch.per_tensor_affine, reduce_range = False)
default_affine_activation_fake_quant = LSQFakeQuantize.with_args(observer = LSQObserver, quant_min = 0, quant_max = 255, 
                                                      dtype = torch.quint8, qscheme = torch.per_tensor_affine, reduce_range = True)
default_symmetric_weight_fake_quant = LSQFakeQuantize.with_args(observer = LSQObserver, quant_min = -128, quant_max = 127,
                                                      dtype = torch.qint8, qscheme = torch.per_tensor_symmetric, reduce_range = False)
default_symmetric_activation_fake_quant = LSQFakeQuantize.with_args(observer = LSQObserver, quant_min = 0, quant_max = 255,
                                                      dtype = torch.quint8, qscheme = torch.per_tensor_symmetric, reduce_range = True)
default_lsq_qconfig = torch.ao.quantization.QConfig(activation = default_affine_activation_fake_quant, weight = default_symmetric_weight_fake_quant)
