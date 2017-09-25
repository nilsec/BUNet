import tensorflow as tf
import numpy as np
from train import cross_entropy_loss

def example_loss(f_out, labels):
    """
        Args:
            f_out: Logits tensor, float [batch_size, channels, depth, height, width]
            labels: Labels tensor, int32 [batch_size, channels, depth, height, width]
    """

    labels = tf.to_int64(labels)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=f_out)
    return loss

if __name__ == "__main__":
    
    with tf.Graph().as_default():
        f_out = tf.placeholder(tf.float32, shape=(10, 2,2,2))
        labels = tf.placeholder(tf.int32, shape=(10,2,2))

        f_out_values = np.asarray(np.arange(80), dtype=float)
        f_out_values = f_out_values.reshape([10,2,2,2])


        feed_dict = {f_out: 
                        np.array([[40,40],[40,4],[40,2],[42,1],[38,0],[43,0],\
                                     [2,30],[4,40],[37,1],[50,0],[8,50],[9,32],\
                                     [30,1],[44,2],[60,2],[40, 1]]),
                     labels: 
                        np.array([0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0], dtype=int)}

        feed_dict_2 = {f_out: f_out_values,
                       labels: np.ones([10, 2, 2])}

        loss_pred = cross_entropy_loss(f_out, labels)

        with tf.Session() as sess:
            print(sess.run(loss_pred, feed_dict=feed_dict_2))
