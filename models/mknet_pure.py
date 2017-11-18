from bunet import bunet, gaussian_noise_layer, pre_sigmoid_split_conv_layer, conv_drop_pass
import os
from train_bunet import stochastic_loss_layer
import tensorflow as tf
import json
import inspect

def make(x_dim=268, 
         y_dim=268, 
         z_dim=84,
         learning_rate=0.5e-4,
         beta1=0.95,
         beta2=0.999,
         epsilon=1e-8,
         num_fmaps=12,
         fmap_inc_factor=5, 
         downsample_factors=([1,3,3],[1,3,3],[1,3,3]), 
         conv_drop_rate=0.0,
	 up_drop_rate=0.0,
	 pre_sigmoid_drop_rate=0.0,
	 bottom_drop_rate=0.5,
	 final_drop_rate=0.0,
         kernel_prior=None,
	 conv_kernel_regularizer=None,
	 conv_kernel_regularizer_scale=None,
	 up_kernel_regularizer=None,
	 up_kernel_regularizer_scale=None,
	 pre_sigmoid_kernel_regularizer=None,
	 pre_sigmoid_kernel_regularizer_scale=None,
	 run=14):

    if not os.path.exists("./run_{}".format(run)):
	os.makedirs("./run_{}".format(run))

    frame = inspect.currentframe()
    args,_,_,values = inspect.getargvalues(frame)
    with open('run_{}/net_params.txt'.format(run), 'w') as f:
        f.write(str([(i, values[i]) for i in args]))

    if conv_kernel_regularizer is not None:
	if conv_kernel_regularizer == "l2":
	    if conv_kernel_regularizer_scale is None:
		raise ValueError("No regularizer scale provided")
	    else:
	    	conv_kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=conv_kernel_regularizer_scale)

    if up_kernel_regularizer is not None:
	if up_kernel_regularizer == "l2":
	    if up_kernel_regularizer_scale is None:
		raise ValueError("No regularizer scale provided")
	    else:
	        up_kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=up_kernel_regularizer_scale)

    if pre_sigmoid_kernel_regularizer is not None:
	if pre_sigmoid_kernel_regularizer == "l2":
	    if pre_sigmoid_kernel_regularizer_scale is None:
		raise ValueError("No regularizer scale provided")
	    else:
	        pre_sigmoid_kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=pre_sigmoid_kernel_regularizer_scale)
 

    raw = tf.placeholder(tf.float32, shape=(z_dim, y_dim, x_dim))
    # Make a reshape in format tf expects with batch as one dim:
    raw_batched = tf.reshape(raw, (1, 1, z_dim, y_dim, x_dim))

    f_out_batched = bunet(fmaps_in=raw_batched,
                          num_fmaps=num_fmaps,
                          fmap_inc_factor=fmap_inc_factor,
                          downsample_factors=list(downsample_factors),
                          conv_drop_rate=conv_drop_rate,
			  up_drop_rate=up_drop_rate,
			  bottom_drop_rate=bottom_drop_rate,
			  final_drop_rate=final_drop_rate,
                          kernel_prior=kernel_prior,
			  conv_kernel_regularizer=conv_kernel_regularizer,
			  up_kernel_regularizer=up_kernel_regularizer,
                          activation='relu')

    

    f_out_shape_batched = f_out_batched.get_shape().as_list()
    f_out_shape = f_out_shape_batched[1:] # strip batch dim
    f_out = tf.reshape(f_out_batched, f_out_shape)

    """
    affs_batched = conv_drop_pass(f_out_batched, 
				  kernel_size=1,
				  num_fmaps=num_fmaps,
			  	  num_repetitions=1,
			  	  drop_rate=pre_sigmoid_drop_rate,
			  	  kernel_prior=kernel_prior,
			  	  kernel_regularizer=pre_sigmoid_kernel_regularizer,
			  	  activation='sigmoid',
			  	  name="sigmoid_layer")
    

    affs = tf.reshape(affs_batched, f_out_shape)
    """
    affs = tf.sigmoid(f_out)
    gt_affs = tf.placeholder(tf.float32, shape=f_out_shape)
    loss_weights = tf.placeholder(tf.float32, shape=f_out_shape)

    """
    loss = tf.losses.mean_squared_error(gt_affs,
					affs,
					loss_weights)
    """
    loss = tf.losses.sigmoid_cross_entropy(gt_affs,
					    f_out,
					    loss_weights)

    if (conv_kernel_regularizer is not None) or (up_kernel_regularizer is not None) or\
	(pre_sigmoid_kernel_regularizer is not None):
	print "Collect regularizer losses..."
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss = tf.add(tf.cast(loss, dtype=tf.float32), tf.reduce_sum(reg_losses))

    opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon)

    optimizer = opt.minimize(loss)

    tf.train.export_meta_graph(filename='run_{}/bunet.meta'.format(run))
    
    names = {
        'raw': raw.name,
        'affs': affs.name,
        'gt_affs': gt_affs.name,
        'loss_weights': loss_weights.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        }

    with open('run_{}/net_io_names.json'.format(run), 'w') as f:
        json.dump(names, f) 

if __name__ == "__main__":
    make()
