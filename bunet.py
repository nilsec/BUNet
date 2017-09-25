import tensorflow as tf
import os
import numpy as np
import time
from toy_data import SquareDataProvider
import h5py

def conv_pass(fmaps_in,
              kernel_size,
              num_fmaps,
              num_repetitions,
              activation='relu',
              name='conv_pass'):

    '''Create a convolution pass::
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
    activation = getattr(tf.nn, activation)

    for i in range(num_repetitions):
        fmaps = tf.layers.conv3d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding='valid',
            data_format='channels_first',
            activation=activation,
            name=name + '_%i'%i)

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

def upsample(fmaps_in, factors, num_fmaps, activation='relu', name='up'):

    activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv3d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)

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

def unet(fmaps_in,
         num_fmaps,
         fmap_inc_factor,
         downsample_factors,
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
    f_left = conv_pass(
        fmaps_in,
        kernel_size=[1,3,3],
        num_fmaps=num_fmaps,
        num_repetitions=2,
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
    g_out = unet(
        g_in,
        num_fmaps=num_fmaps*fmap_inc_factor,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        activation=activation,
        layer=layer+1)

    print(prefix + "g_out: " + str(g_out.shape))

    # upsample
    g_out_upsampled = upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps,
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
    f_out = conv_pass(
        f_right,
        kernel_size=[1,3,3],
        num_fmaps=num_fmaps,
        num_repetitions=2,
        name='unet_layer_%i_right'%layer)

    print(prefix + "f_out: " + str(f_out.shape))

    return f_out

def sigmoid_conv_layer(f_out):
    affs_batched = conv_pass(f_out,
                             kernel_size=1,
                             num_fmaps=2,
                             num_repetitions=1,
                             activation='sigmoid')

    return affs_batched

def affinity_loss_layer(f_out_batched, gt_affs, loss_weights):
    output_shape_batched = f_out_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip batch dimension

    affs = tf.reshape(f_out_batched, output_shape)
    
    loss = tf.losses.mean_squared_error(
            gt_affs,
            affs,
            loss_weights)
   
    tf.summary.scalar('loss', loss)

    return loss

def affinity_training(loss, step):
    
    opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8)

    train_op = opt.minimize(loss, global_step=step)
    return train_op

def fill_feed_dict(data_provider_object,
                   data_path,
                   raw_placeholder, 
                   affinity_placeholder, 
                   loss_weight_placeholder,
                   loss_weight_scaling,
                   step,
                   f_out_shape,
                   group="training"):
    

    raw_vals, gt_affs_vals, loss_weight_vals =\
        data_provider_object.get_batch(data_path, step, group)
    
    start_x = int((np.shape(gt_affs_vals)[3] - f_out_shape[3])/2)
    start_y = int((np.shape(gt_affs_vals)[2] - f_out_shape[2])/2)
    
    end_x = int(start_x + f_out_shape[3])
    end_y = int(start_y + f_out_shape[2])

    gt_affs_vals = gt_affs_vals[:,:, start_y:end_y, start_x:end_x]
    loss_weight_vals = loss_weight_vals[:,:, start_y:end_y, start_x:end_x]
    loss_weight_vals *= loss_weight_scaling
    
    feed_dict = {raw_placeholder: raw_vals,
                 affinity_placeholder: gt_affs_vals,
                 loss_weight_placeholder: loss_weight_vals}

    return feed_dict

def evaluation(affinities, gt_affinities):
    diff = tf.abs(tf.subtract(affinities, gt_affinities))
    is_close = tf.less_equal(diff, tf.fill(tf.shape(diff), 0.1))

    n_is_close = tf.reduce_sum(tf.cast(is_close, tf.float32))
    frac_is_close = tf.divide(n_is_close, 56. * 56.)
    print(n_is_close)

    return frac_is_close, affinities

def do_eval(sess,
            eval_correct,
            data_provider_object,
            data_path,
            raw_placeholder,
            affinity_placeholder,
            loss_weight_placeholder,
            f_out_shape,
            group="validation"):
    """
    Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    frac_is_close = 0

    f = h5py.File(data_path, "a")
    try:
        pred = f.create_group("predictions")
        dset_affs = pred.create_dataset("affs",
                                        [1] + f_out_shape,
                                        maxshape=[None] + f_out_shape)
    except ValueError:
        dset_affs = f["predictions"]["affs"]
 

    for step in range(1000):
        feed_dict = fill_feed_dict(data_provider_object,
                                   data_path,
                                   raw_placeholder,
                                   affinity_placeholder,
                                   loss_weight_placeholder,
                                   1,
                                   step,
                                   f_out_shape,
                                   group)

        frac, affinities = sess.run(eval_correct, feed_dict=feed_dict)
        if step == 0:
            shape = list(dset_affs.shape)
            shape[0] += 1
            dset_affs.resize(shape)
            dset_affs[shape[0] - 1] = affinities

        frac_is_close += frac
            
    average = frac_is_close/1000.
    average *= 100
    print("Fraction of affinities differing less than 0.1: %s %%" % average)

def train():
    with tf.Graph().as_default():
        raw = tf.placeholder(tf.float32, shape=(1, 1, 268, 268))
        # Make a reshape in format tf expects with batch as one dim:
        raw_batched = tf.reshape(raw, (1,1,1,268,268))

        f_out_batched = unet(fmaps_in=raw_batched,
                             num_fmaps=12,
                             fmap_inc_factor=2,
                             downsample_factors=[[1,3,3],[1,3,3],[1,3,3]],
                             activation='relu')

        f_out_batched = sigmoid_conv_layer(f_out_batched)

        f_out_shape_batched = f_out_batched.get_shape().as_list()
        f_out_shape = f_out_shape_batched[1:] # strip batch dim

        gt_affs = tf.placeholder(tf.float32, shape=f_out_shape)
        loss_weights = tf.placeholder(tf.float32, shape=f_out_shape)

        loss = affinity_loss_layer(f_out_batched, gt_affs, loss_weights)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = affinity_training(loss, global_step)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter("./bunet_logs", sess.graph)

        data_provider = SquareDataProvider()
        data_path = "/media/nilsec/d0/bunet/square_data/data.hdf5"

        sess.run(init)

        for step in range(10000):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_provider,
                                       data_path,
                                       raw,
                                       gt_affs,
                                       loss_weights,
                                       100,
                                       step,
                                       f_out_shape,
                                       "training")

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time
            
            if step%100 == 0:
                print("Step %d: loss %.2f (%3f sec)" % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                
            if (step + 1) % 1000 == 0 or (step + 1) == 10000:
                checkpoint_file = os.path.join("./bunet_logs", "bunet.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                
                eval_correct = evaluation(f_out_batched, gt_affs)

                print("Training Data Eval:") 
                do_eval(sess,
                        eval_correct,
                        data_provider,
                        data_path,
                        raw,
                        gt_affs,
                        loss_weights,
                        f_out_shape,
                        group="training")

                print("Validation Data Eval:")
                do_eval(sess,
                        eval_correct,
                        data_provider,
                        data_path,
                        raw,
                        gt_affs,
                        loss_weights,
                        f_out_shape,
                        group="validation")

                print("Test Data Eval:")
                do_eval(sess,
                        eval_correct,
                        data_provider,
                        data_path,
                        raw,
                        gt_affs,
                        loss_weights,
                        f_out_shape,
                        group="test")
 
def test_affinity_loss_layer():
    with tf.Graph().as_default():
        affs_batched = tf.placeholder(tf.float32, (1,3,5,10,10))
        gt_affs = tf.placeholder(tf.float32, (3,5,10,10))
        loss_weights = tf.placeholder(tf.float32, shape=(3,5,10,10))

        loss = affinity_loss_layer(affs_batched, gt_affs, loss_weights)

        feed_dict = {affs_batched: np.ones([1,3,5,10,10]),
                     gt_affs: np.zeros([3,5,10,10]),
                     loss_weights: np.ones([3,5,10,10]) * 10}
        
        with tf.Session() as sess:
            print(sess.run(loss, feed_dict=feed_dict))
        
if __name__ == "__main__":
    if not os.path.exists("./bunet_logs"):
        os.makedirs("./bunet_logs")
        
    train()
    
