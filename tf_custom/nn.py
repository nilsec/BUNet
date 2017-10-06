from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops

def mc_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):

  """
    Computes dropout with option to keep the output unscaled.

  If scale is true, the scaling is so that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """

  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    # Do nothing if we know keep_prob == 1
    if tensor_util.constant_value(keep_prob) == 1:
      return x

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape,
                                               seed=seed,
                                               dtype=x.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)

    # NOTE: Modify standard tf implementation to have no scaling
    ret = x * binary_tensor

    ret.set_shape(x.get_shape())

    return ret


