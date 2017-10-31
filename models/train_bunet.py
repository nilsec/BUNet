from bunet import bunet, gaussian_noise_layer, pre_sigmoid_split_conv_layer
import time
import tensorflow as tf
import os
import numpy as np
import h5py
import pdb
from tensorflow.python import debug as tf_debug

def stochastic_loss_layer(logits_with_noise,
                          gt_affs, 
                          loss_weights,
                          n_samples):

    shape_logits = logits_with_noise.get_shape()
    shape_gt = gt_affs.get_shape().as_list()

    gt_affs = tf.reshape(gt_affs, [1] + shape_gt)
    gt_affs_tiled = tf.tile(gt_affs,
                            multiples=[n_samples, 1, 1, 1, 1])

    loss_weights = tf.cast(tf.reshape(loss_weights, [1] + shape_gt), tf.float64)
    loss_weights_tiled = tf.tile(loss_weights,
                                 multiples=[n_samples, 1, 1, 1, 1])

    minus_one = tf.cast(tf.fill(shape_logits, -1), tf.float32)
    plus_minus = tf.pow(minus_one, gt_affs_tiled, name="plus_minus")

    two_logits = tf.multiply(2.0, logits_with_noise)
    two_logits = tf.cast(two_logits, tf.float64)
    plus_minus = tf.cast(plus_minus, tf.float64)
   
    log_exp_one = tf.log1p(tf.exp(tf.multiply(plus_minus, two_logits)))
    weighted_log_exp_one = tf.multiply(log_exp_one, loss_weights)
    loss = tf.divide(tf.reduce_sum(weighted_log_exp_one), float(n_samples))

    tf.summary.scalar('loss', loss)

    return loss


def optimize(loss, 
             step,
             learning_rate,
             beta1,
             beta2,
             epsilon):

    opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon)

    grads = opt.compute_gradients(loss)
    #for index, grad in enumerate(grads):
        #tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
    
    #train_op = opt.minimize(loss, global_step=step)
    train_step = opt.apply_gradients(grads)
    return train_step


def fill_feed_dict(data_provider_object,
                   data_path,
                   raw_placeholder, 
                   gt_aff_placeholder, 
                   loss_weight_placeholder,
                   loss_weight_scaling,
                   step,
                   f_out_shape,
                   group="train"):
    

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
                 gt_aff_placeholder: gt_affs_vals,
                 loss_weight_placeholder: loss_weight_vals}

    return feed_dict


def evaluation(affs,
               affs_noise,
               sigma,
               gt_affs,
               threshold=0.1):
    # Calculates what percentage of affinities
    # is below error threshold

    diff = tf.abs(tf.subtract(affs, gt_affs))
    diff_noise = tf.abs(tf.subtract(affs_noise, gt_affs))
    
    is_close = tf.less_equal(diff, tf.fill(tf.shape(diff), threshold))
    is_close_noise = tf.less_equal(diff_noise, tf.fill(tf.shape(diff_noise), threshold))
 

    n_is_close = tf.reduce_sum(tf.cast(is_close, tf.float32))
    n_is_close_noise = tf.reduce_sum(tf.cast(is_close_noise, tf.float32))

    frac_is_close = tf.divide(n_is_close, 56. * 56.)
    frac_is_close_noise = tf.divide(n_is_close_noise, 56. * 56.)

    return frac_is_close, frac_is_close_noise, affs, affs_noise, gt_affs, sigma


def do_eval(sess,
            eval_correct,
            data_provider_object,
            data_path,
            raw_placeholder,
            affinity_placeholder,
            loss_weight_placeholder,
            lw_scaling,
            f_out_shape,
            group="validation",
            snap_steps=[0],
            snapshot_save_path=None,
            snapshot_id=None):
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
    frac_is_close_noise = 0

    for step in range(1000):
        feed_dict = fill_feed_dict(data_provider_object,
                                   data_path,
                                   raw_placeholder,
                                   affinity_placeholder,
                                   loss_weight_placeholder,
                                   lw_scaling,
                                   step,
                                   f_out_shape,
                                   group)

        frac, frac_noise, affs, affs_noise, gt_affs, sigma = sess.run(eval_correct, feed_dict=feed_dict)

        if snapshot_save_path is not None:
            if step in snap_steps:
                save_snapshot(affs,
                              affs_noise,
                              gt_affs,
                              sigma,
                              snapshot_save_path,
                              snapshot_id
                              )

        frac_is_close += frac
        frac_is_close_noise += frac_noise
            
    average = frac_is_close/1000.
    average *= 100

    average_noise = frac_is_close_noise/1000.
    average_noise *= 100

    print("Fraction of affinities differing less than 0.1: %s %%" % average)
    print("Fraction of noisy affinities differing less than 0.1: %s %%" % average_noise)


def save_snapshot(aff_out_batched,
                  aff_out_batched_with_noise,
                  gt_affs,
                  sigma,
                  snapshot_save_path,
                  snapshot_id):

    shape = list(np.shape(aff_out_batched))
    shape = shape[1:]
    print(shape)

    f = h5py.File(snapshot_save_path, "a")

    try:
        snaps = f.create_group("snapshot_%s" % snapshot_id)
        dset_affs = snaps.create_dataset("affs",
                                        [1] + shape,
                                        maxshape=[None] + shape)

        dset_affs_noise = snaps.create_dataset("affs_noise",
                                        [1] + shape,
                                        maxshape=[None] + shape)

        dset_sigma = snaps.create_dataset("sigma",
                                        [1] + shape,
                                        maxshape=[None] + shape)

        dset_gtaffs = snaps.create_dataset("gt_affs",
                                           [1] + shape,
                                           maxshape=[None] + shape)
 
 
    except ValueError:
        dset_affs = f["snapshot_%s" % snapshot_id]["affs"]
        dset_affs_noise = f["snapshot_%s" % snapshot_id]["affs_noise"]
        dset_sigma = f["snapshot_%s" % snapshot_id]["sigma"]
        dset_gtaffs = f["snapshot_%s" % snapshot_id]["gt_affs"]

    
    dset_shape = list(dset_affs.shape)
    dset_shape[0] += 1

    dset_affs.resize(dset_shape) 
    dset_affs_noise.resize(dset_shape)
    dset_sigma.resize(dset_shape)
    dset_gtaffs.resize(dset_shape)

    idx = dset_shape[0] - 1
    dset_affs[idx] = aff_out_batched[0, :,:,:,:]
    dset_affs_noise[idx] = aff_out_batched_with_noise[0, :,:,:,:]
    dset_sigma[idx] = sigma[0, :,:,:,:]     
    dset_gtaffs[idx] = gt_affs 

def train(steps,
          drop_rate,
          mc_samples,
          lw_scaling,
          data_path,
          data_provider,
          log_dir,
          kernel_prior,
          learning_rate,
          beta1,
          beta2,
          epsilon,
          snapshot_save_path,
          snapshot_id):

    with tf.Graph().as_default():
        raw = tf.placeholder(tf.float32, shape=(1, 1, 268, 268))
        # Make a reshape in format tf expects with batch as one dim:
        raw_batched = tf.reshape(raw, (1,1,1,268,268))

        f_out_batched = bunet(fmaps_in=raw_batched,
                             num_fmaps=12,
                             fmap_inc_factor=2,
                             downsample_factors=[[1,3,3],[1,3,3],[1,3,3]],
                             drop_rate=drop_rate,
                             kernel_prior=kernel_prior,
                             activation='relu')

        logits = pre_sigmoid_split_conv_layer(f_out_batched, 
                                              drop_rate, 
                                              kernel_prior)

        logits_with_noise = gaussian_noise_layer(logits, mc_samples)

        f_out_shape_batched = f_out_batched.get_shape().as_list()
        f_out_shape = f_out_shape_batched[1:] # strip batch dim
        
        gt_affs_shape = f_out_shape
        gt_affs_shape[0] = 2

        gt_affs = tf.placeholder(tf.float32, shape=gt_affs_shape)
        loss_weights = tf.placeholder(tf.float32, shape=gt_affs_shape)
        
        aff_out_batched = tf.sigmoid(logits[:,0:2,:,:,:])
        aff_out_with_noise = tf.sigmoid(logits_with_noise[0,:,:,:,:])
        aff_out_batched_with_noise = tf.reshape(aff_out_with_noise, tf.shape(aff_out_batched))
        sigma = logits[:,2:4,:,:,:]
        
        aff_out_shape = aff_out_batched.get_shape().as_list()[1:]
        

        loss = stochastic_loss_layer(logits_with_noise,
                                     gt_affs, 
                                     loss_weights,
                                     mc_samples)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_step = optimize(loss, 
                            global_step,
                            learning_rate,
                            beta1,
                            beta2,
                            epsilon)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.run(init)

        for step in range(steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_provider_object=data_provider,
                                       data_path=data_path,
                                       raw_placeholder=raw,
                                       gt_aff_placeholder=gt_affs,
                                       loss_weight_placeholder=loss_weights,
                                       loss_weight_scaling=lw_scaling,
                                       step=step,
                                       f_out_shape=f_out_shape,
                                       group="train")

            _, loss_value = sess.run([train_step, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time
            
            if step%100 == 0:
                print("Step %d: loss %.2f (%3f sec)" % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                
            if (step + 1) % 1000 == 0 or (step + 1) == steps:
                checkpoint_file = os.path.join(log_dir, "unet.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                
                eval_correct = evaluation(affs=aff_out_batched,
                                          affs_noise=aff_out_batched_with_noise,
                                          sigma=sigma,
                                          gt_affs=gt_affs,
                                          threshold=0.1)

                print("Training Data Eval:") 
                do_eval(sess,
                        eval_correct,
                        data_provider,
                        data_path,
                        raw,
                        gt_affs,
                        loss_weights,
                        lw_scaling,
                        aff_out_shape,
                        group="train",
                        snapshot_save_path=snapshot_save_path,
                        snapshot_id=snapshot_id)

                """
                print("Validation Data Eval:")
                do_eval(sess,
                        eval_correct,
                        data_provider,
                        data_path,
                        raw,
                        gt_affs,
                        loss_weights,
                        aff_out_shape,
                        group="validation")

                print("Test Data Eval:")
                do_eval(sess,
                        eval_correct,
                        data_provider,
                        data_path,
                        raw,
                        gt_affs,
                        loss_weights,
                        aff_out_shape,
                        group="test")
                """

if __name__ == "__main__":
    steps = 100000
    drop_rate = 0.2
    data_path = "/media/nilsec/d0/bunet/square_data_noisy/data.hdf5"
    data_provider = SquareDataProvider()
    snapshot_save_path = data_path.replace("data.hdf5", "snaps.hdf5")
    snapshot_id = 9
    run = 9
    log_dir = "./bunet_logs/run_%s" % run
    kernel_prior = tf.truncated_normal_initializer(mean=0.01, 
                                                   stddev=.01,
                                                   seed=13,
                                                   dtype=tf.float32)
    kernel_prior=None
    learning_rate = 0.0005
    beta1 = 0.7
    beta2 = 0.7
    epsilon = 1e-8

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    train(steps=steps,
          drop_rate=drop_rate,
          mc_samples=50,
          lw_scaling=1.0,
          data_path=data_path,
          data_provider=data_provider,
          log_dir=log_dir,
          kernel_prior=kernel_prior,
          learning_rate=learning_rate,
          beta1=beta1,
          beta2=beta2,
          epsilon=1e-8,
          snapshot_save_path=snapshot_save_path,
          snapshot_id=snapshot_id)
     
