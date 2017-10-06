import tensorflow as tf
from unet import unet, sigmoid_conv_layer
import os
import numpy as np
import time
from data import SquareDataProvider
import h5py


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

        summary_writer = tf.summary.FileWriter("./unet_logs", sess.graph)

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
                checkpoint_file = os.path.join("./unet_logs", "unet.ckpt")
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
 

if __name__ == "__main__":
    if not os.path.exists("./unet_logs"):
        os.makedirs("./unet_logs")
        
    train()
    
