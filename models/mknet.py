from bunet import bunet, gaussian_noise_layer, pre_sigmoid_split_conv_layer
import os
from train_bunet import stochastic_loss_layer
import tensorflow as tf
import json
import inspect

def make(x_dim=268, 
         y_dim=268, 
         z_dim=84,
         mc_samples=25,
         learning_rate=0.5e-4,
         beta1=0.95,
         beta2=0.999,
         epsilon=1e-8,
         num_fmaps=12,
         fmap_inc_factor=6, 
         downsample_factors=([1,3,3],[1,3,3],[1,3,3]), 
         drop_rate=0.2,
	 pre_sigmoid_drop_rate=0.0,
         kernel_prior=None,
	 kernel_regularizer=None,
	 kernel_regularizer_scale=None,
	 run=2):

    if not os.path.exists("./run_{}".format(run)):
	os.makedirs("./run_{}".format(run))

    frame = inspect.currentframe()
    args,_,_,values = inspect.getargvalues(frame)
    with open('run_{}/net_params.txt'.format(run), 'w') as f:
        f.write(str([(i, values[i]) for i in args]))

    if kernel_regularizer is not None:
	if kernel_regularizer == "l2":
	    if kernel_regularizer_scale is None:
		raise ValueError("No regularizer scale provided")
	    else:
	    	kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=kernel_regularizer_scale)

    raw = tf.placeholder(tf.float32, shape=(z_dim, y_dim, x_dim))
    # Make a reshape in format tf expects with batch as one dim:
    raw_batched = tf.reshape(raw, (1, 1, z_dim, y_dim, x_dim))

    f_out_batched = bunet(fmaps_in=raw_batched,
                          num_fmaps=num_fmaps,
                          fmap_inc_factor=fmap_inc_factor,
                          downsample_factors=list(downsample_factors),
                          drop_rate=drop_rate,
                          kernel_prior=kernel_prior,
			  kernel_regularizer=kernel_regularizer,
                          activation='relu')

    logits = pre_sigmoid_split_conv_layer(f_out=f_out_batched,
                                          drop_rate=pre_sigmoid_drop_rate,
                                          kernel_prior=kernel_prior,
					  kernel_regularizer=kernel_regularizer,
					  num_fmaps=num_fmaps)

    logits_with_noise = gaussian_noise_layer(logits=logits, 
					     n_samples=mc_samples,
					     num_fmaps=num_fmaps)

    # Note this is the shape before the split layer i.e.
    # the shape will be (bs, num_fmaps=12, z,y,x)

    f_out_shape_batched = f_out_batched.get_shape().as_list()
    print f_out_shape_batched
    f_out_shape = f_out_shape_batched[1:] # strip batch dim
    f_out_shape[0] = num_fmaps # Set f_maps to output feature maps (3 in case of x,y,z affs) 

    affs = tf.reshape(tf.sigmoid(logits[:,0:num_fmaps,:,:,:]), f_out_shape)

    gt_affs = tf.placeholder(tf.float32, shape=f_out_shape)
    loss_weights = tf.placeholder(tf.float32, shape=f_out_shape)

    sigma = logits[:,num_fmaps:,:,:,:]

    loss = stochastic_loss_layer(logits_with_noise,
                                 gt_affs,
                                 loss_weights,
                                 mc_samples)

    if kernel_regularizer is not None:
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss += tf.reduce_sum(reg_losses)

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
        'logits': logits.name,
        'logits_with_noise': logits_with_noise.name,
        'sigma': sigma.name
        }

    with open('run_{}/net_io_names.json'.format(run), 'w') as f:
        json.dump(names, f) 

if __name__ == "__main__":
    make()
