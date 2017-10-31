import tensorflow as tf
import tf_custom 
import os
import numpy as np

def conv_drop_pass(fmaps_in,
                  kernel_size,
                  num_fmaps,
                  num_repetitions,
                  drop_rate,
                  kernel_prior,
                  activation='relu',
                  name='conv_drop_pass'):

    '''
    Create a convolution pass followed by mc_dropout:
        f_in --> f_1 --> ... --> f_n
        where each ``-->`` is a convolution followed by a (non-linear) activation
        function and ``n`` ``num_repetitions``. Each convolution will decrease the
        size of the feature maps by ``kernel_size-1``.

    Args:
        f_in:
            The input tensor of shape ``(batch_size, channels, depth, height, width)``.
        kernel_size:
            Size of the kernel. Forwarded to tf.layers.conv3d.
        num_fmaps:
            The number of feature maps to produce with each convolution.
        num_repetitions:
            How many convolutions to apply.
        activation:
            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).
    '''

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i in range(num_repetitions):
        fmaps = tf.layers.conv3d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding='valid',
            data_format='channels_first',
            activation=activation,
            kernel_initializer=kernel_prior,
            name=name + '_%i'%i)
        
        fmaps = tf_custom.mc_dropout(
            inputs=fmaps,
            rate=drop_rate)

    return fmaps

def downsample(fmaps_in, factors, name='down'):

    fmaps = tf.layers.max_pooling3d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        name=name)

    return fmaps

def upsample(fmaps_in, factors, num_fmaps, drop_rate, kernel_prior, activation='relu', name='up'):

    activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv3d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        kernel_initializer=kernel_prior,
        name=name)

    fmaps = tf_custom.layers.mc_dropout(
        inputs=fmaps,
        rate=drop_rate)

    return fmaps

def crop_zyx(fmaps_in, shape):
    '''Crop only the spacial dimensions to match shape.
    Args:
        fmaps_in:
            The input tensor.
        shape:
            A list (not a tensor) with the requested shape [_, _, z, y, x].
    '''

    in_shape = fmaps_in.get_shape().as_list()

    offset = [
        0, # batch
        0, # channel
        (in_shape[2] - shape[2])//2, # z
        (in_shape[3] - shape[3])//2, # y
        (in_shape[4] - shape[4])//2, # x
    ]
    size = [
        in_shape[0],
        in_shape[1],
        shape[2],
        shape[3],
        shape[4],
    ]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps

def bunet(fmaps_in,
          num_fmaps,
          fmap_inc_factor,
          downsample_factors,
          drop_rate,
          kernel_prior,
          activation='relu',
          layer=0):

    '''Create a U-Net::
        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...
    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.
    The U-Net expects tensors to have shape ``(batch=1, channels, depth, height,
    width)``.
    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution.
    Args:
        fmaps_in:
            The input tensor.
        num_fmaps:
            The number of feature maps in the first layer. This is also the
            number of output feature maps.
        fmap_inc_factor:
            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.
        downsample_factors:
            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.
        activation:
            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).
        layer:
            Used internally to build the U-Net recursively.
    '''

    prefix = "    "*layer
    print(prefix + "Creating U-Net layer %i"%layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))

    # convolve
    f_left = conv_drop_pass(
        fmaps_in,
        kernel_size=[1,3,3], # NOTE: used to be 3, check when going back to 3D.
        num_fmaps=num_fmaps,
        num_repetitions=2,
        drop_rate=drop_rate,
        kernel_prior=kernel_prior,
        activation=activation,
        name='unet_layer_%i_left'%layer)

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))
    if bottom_layer:
        print(prefix + "bottom layer")
        print(prefix + "f_out: " + str(f_left.shape))
        return f_left

    # downsample
    g_in = downsample(
        f_left,
        downsample_factors[layer],
        'unet_down_%i_to_%i'%(layer, layer + 1))

    # recursive U-net
    g_out = bunet(
        g_in,
        num_fmaps=num_fmaps*fmap_inc_factor,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        drop_rate=drop_rate,
        kernel_prior=kernel_prior,
        activation=activation,
        layer=layer+1)

    print(prefix + "g_out: " + str(g_out.shape))

    # upsample
    g_out_upsampled = upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps,
        drop_rate=drop_rate,
        kernel_prior=kernel_prior,
        activation=activation,
        name='unet_up_%i_to_%i'%(layer + 1, layer))

    print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

    # copy-crop
    f_left_cropped = crop_zyx(f_left, g_out_upsampled.get_shape().as_list())

    print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

    # concatenate along channel dimension
    f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

    print(prefix + "f_right: " + str(f_right.shape))

    # convolve
    f_out = conv_drop_pass(
        f_right,
        kernel_size=[1,3,3],
        num_fmaps=num_fmaps,
        num_repetitions=2,
        drop_rate=drop_rate,
        kernel_prior=kernel_prior,
        name='unet_layer_%i_right'%layer)

    print(prefix + "f_out: " + str(f_out.shape))

    return f_out


def pre_sigmoid_split_conv_layer(f_out, drop_rate, kernel_prior, num_fmaps):
    # We apply dropout here as well, does it make sense?

    logits = conv_drop_pass(f_out,
                            kernel_size=1,
                            num_fmaps=2*num_fmaps, 
                            num_repetitions=1,
                            drop_rate=drop_rate,
                            kernel_prior=kernel_prior,
                            activation=None,
                            name="split_layer")

    return logits


def gaussian_noise_layer(logits, n_samples, num_fmaps):
    pred = logits[:,0:num_fmaps,:,:,:]
    sigma = logits[:,num_fmaps:,:,:,:]
    
    shape_pred = pred.get_shape().as_list()

    flat_pred = tf.reshape(pred, [-1], name="flat_pred")
    flat_sigma = tf.reshape(sigma, [-1], name="flat_sigma")

    pred_tiled = tf.tile(pred,
                         multiples=[n_samples, 1, 1, 1, 1],
                         name="tile_pred")

   
    normal_dist = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(tf.shape(flat_pred)), 
                                                                  flat_sigma,
                                                                  allow_nan_stats=False)
    
    flat_epsilon_tiled = normal_dist.sample(sample_shape=[n_samples], name="sample")
    epsilon_tiled = tf.reshape(flat_epsilon_tiled, tf.shape(pred_tiled), name="eps_tiled")

    logits_with_noise = tf.add(pred_tiled, epsilon_tiled, name="add_noise")
    return logits_with_noise
