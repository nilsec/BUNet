from .nn import mc_dropout
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.layers.core import Dropout
from tensorflow.python.ops import array_ops

class MCDropout(Dropout):
    def __init__(self,
                 rate=0.5,
                 noise_shape=None,
                 seed=None,
                 name=None,
                 **kwargs):

        super(MCDropout, self).__init__(rate, noise_shape, seed, name, **kwargs)

    def call(self, inputs, training=False):
        def dropped_inputs():
            return nn.mc_dropout(inputs, 1  - self.rate,
                                 noise_shape=self._get_noise_shape(inputs),
                                 seed=self.seed)

        return utils.smart_cond(training,
                                dropped_inputs,
                                lambda: array_ops.identity(inputs))

def mc_dropout(inputs,
               rate=0.5,
               noise_shape=None,
               seed=None,
               training=False,
               name=None):

    layer = MCDropout(rate, noise_shape=noise_shape, seed=seed, name=name)
    return layer.apply(inputs, training=training)
