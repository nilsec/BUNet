import numpy as np
import os
#import matplotlib.pyplot as plt
import pdb
import h5py
import json

def generate_square_image(x_dim, y_dim, n_squares, show_image=False):
    """
    Returns an image with n_squares randomly placed squares on a canvas
    of size x_dim, y_dim as well as the corresponding affinities in 
    a np.array of shape (2,1,x_dim,y_dim).
    I.e. 2 channels where channel 0 corresponds to x affinities
    and channel 1 correspinds to y affinities. 
    """

    canvas = np.zeros([x_dim, y_dim])

    # make a border of +1 around affinity maps to take care of edge effects
    # make the convention that the affinities saved in pos x,y
    # always denotes the affinity in positive direction. That means for x 
    # the affinity w.r.t. to the right neighbour and 
    # for y the affinity w.r.t. the down neighbour.
    x_affs = np.zeros([x_dim + 2, y_dim + 2])
    y_affs = np.zeros([x_dim + 2, y_dim + 2])   

    for n in range(n_squares):
        x_mid = np.random.randint(0, x_dim, size=1)[0]
        y_mid = np.random.randint(0, y_dim, size=1)[0]

        size = np.random.randint(1, int(min(x_dim, y_dim)/10.), size=1)[0]
        square = np.ones([2*size + 1, 2*size + 1])
    
        canvas[x_mid - size:x_mid+size + 1, y_mid - size:y_mid+size + 1] =\
            np.ones([x_dim, y_dim])[x_mid - size:x_mid+size + 1, y_mid - size:y_mid+size + 1] *\
            np.random.uniform(0.1,1,1)

        x_affs[1:x_dim + 1, 1:y_dim + 1] = (np.roll(canvas, -1, axis=1) - canvas) == 0
        y_affs[1:x_dim + 1, 1:y_dim + 1] = (np.roll(canvas, -1, axis=0) - canvas) == 0

    x_affs = np.asarray(x_affs[1:x_dim + 1, 1:y_dim + 1], dtype=float)
    y_affs = np.asarray(y_affs[1:x_dim + 1, 1:y_dim + 1], dtype=float)

    if show_image:
        plt.imshow(canvas)
        plt.imshow(x_affs, alpha=0.5)
        plt.imshow(y_affs, alpha=0.5)
        plt.show()

    canvas = np.reshape(canvas, [1, np.shape(canvas)[1], np.shape(canvas)[0]]) # z,y,x
    x_affs = np.reshape(x_affs, [1, np.shape(x_affs)[1], np.shape(x_affs)[0]]) # z,y,x
    y_affs = np.reshape(y_affs, [1, np.shape(y_affs)[1], np.shape(y_affs)[0]]) # z,y,x
    gt_affs = np.stack([x_affs, y_affs])
    
    # Normalization factor in loss weights
    n_pos_aff_x = len(np.transpose(np.nonzero(x_affs)))
    n_pos_aff_y = len(np.transpose(np.nonzero(y_affs)))

    n_neg_aff_x = x_dim * y_dim - n_pos_aff_x
    n_neg_aff_y = x_dim * y_dim - n_pos_aff_y

    x_affs_neg = np.asarray(x_affs == 0, dtype=float)
    y_affs_neg = np.asarray(y_affs == 0, dtype=float)
    x_weights = x_affs + x_affs_neg * (float(n_pos_aff_x)/n_neg_aff_x)
    y_weights = y_affs + y_affs_neg * (float(n_pos_aff_y)/n_neg_aff_y)
    
    loss_weights = np.stack([x_weights, y_weights])
    assert(np.shape(loss_weights) == np.shape(gt_affs))
    
    if show_image:
        plt.imshow(x_weights[0,:,:])
        plt.show()

    if show_image:
        plt.imshow(canvas[0, :,:])
        plt.imshow(gt_affs[0,0,:,:], alpha=0.5) # x affinities
        plt.imshow(gt_affs[1,0,:,:], alpha=0.5) # y affinities
        plt.show()

    return canvas, gt_affs, loss_weights

class SquareDataProvider(object):
    def __init__(self):
        self.x_dim = None
        self.y_dim = None
        self.z_dim = 1
        self.n_squares = None
        self.size = None
        
    def generate_dataset(self, 
                         save_dir, 
                         size, 
                         x_dim, 
                         y_dim, 
                         n_squares,
                         group_name="training"):


        self.x_dim = x_dim
        self.size = size
        self.y_dim = y_dim
        self.n_squares = n_squares

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
 
        f = h5py.File(os.path.join(save_dir, "data.hdf5"), 'a')
        
        group = f.create_group(group_name)
        dset_raw = group.create_dataset("raw", 
                                           (size, 1, self.z_dim, self.y_dim, self.x_dim),
                                           chunks=(1,1,self.z_dim,self.y_dim,self.x_dim))
        
        dset_gt_affs = group.create_dataset("gt_affs",
                                               (size, 2, self.z_dim, self.y_dim, self.x_dim),
                                               chunks=(1, 2, self.z_dim, self.y_dim, self.x_dim))

        dset_loss_weights = group.create_dataset("loss_weights",
                                                    (size, 2, self.z_dim, self.y_dim, self.x_dim),
                                                    chunks=(1, 2, self.z_dim, self.y_dim, self.x_dim))

        for img in xrange(size):
            # create image:
            canvas, gt_affs, loss_weights =\
                generate_square_image(self.x_dim, self.y_dim, self.n_squares)
            
            # write in hdf5 file:
            dset_raw[img, :,:,:,:] = canvas
            dset_gt_affs[img, :,:,:,:] = gt_affs
            dset_loss_weights[img, :,:,:,:] = loss_weights

        json.dump(self.__dict__, open(os.path.join(save_dir, "meta.json"), "w+"))

    def get_batch(self, data_path, n_batch, group):
        f = h5py.File(data_path, "r")
        raw = f[group]["raw"][n_batch,:,:,:,:]
        gt_affs = f[group]["gt_affs"][n_batch, :,:,:,:]
        loss_weights = f[group]["loss_weights"][n_batch, :,:,:,:]

        return raw, gt_affs, loss_weights

def show_predictions(data_path):
    f = h5py.File(data_path, "r")
    prediction = f["predictions"]["affs"][-2, :,:,:,:]

    plt.imshow(prediction[0,0, :,:], alpha=0.5)
    plt.imshow(prediction[1,0, :,:], alpha=0.5)
    plt.show()

def test_data_provider(data_path):
    squares = SquareDataProvider()
    #squares.generate_dataset("./square_data", 10, 268, 268, 50)
    raw, gt_affs, loss_weights = squares.get_batch(data_path, 0, "train")
    
    plt.imshow(raw[0,0, :,:])
    plt.imshow(gt_affs[0,0,:,:], alpha=0.5) # x affinities
    #plt.imshow(gt_affs[1,0,:,:], alpha=0.5) # y affinities
    #plt.imshow(loss_weights[0,0,:,:], alpha=0.5)
    #plt.imshow(loss_weights[1,0,:,:], alpha=0.5)
    plt.show()

def gen_training_data():
    squares = SquareDataProvider()
    squares.generate_dataset("/media/nilsec/d0/bunet/square_data", 1000, 268, 268, 50, "test")
  
        
if __name__ == "__main__":
    #show_predictions("/media/nilsec/d0/bunet/square_data/data.hdf5")
    test_data_provider("/media/nilsec/d0/bunet/square_data_noisy/data.hdf5")
 
 
