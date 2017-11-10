import numpy as np
import h5py
from scipy.special import expit
import os

def online_variance(prediction_paths, dataset="volumes/labels/pred_affinities", reshape=None):
    """
    Welford algorithm, numerically stable, see wikipedia 
    "Algorithms for calculating variance"
    """
    
    n = 0
    mean = M2 = 0.0

    N = len(prediction_paths)
    for path in prediction_paths:
        n += 1
        print("{}/{}".format(n, N))
        f = h5py.File(path)
	try:
            if reshape is not None:
            	x = np.array(np.reshape(f[dataset].value, reshape), dtype=np.float32)
	    else:
	        x = np.array(f[dataset].value, dtype=np.float32)
 
	except KeyError:
	    print "No dset: ", path
        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta * delta2

    if n<2:
        return np.nan
    else:
        return mean, M2/(n-1), M2/n

def combine_predictions(base_dir, output_path):
    dirs = [os.path.join(base_dir, top_dir) for top_dir in os.listdir(base_dir)]
    predictions = [os.path.join(base_dir + "/p_{}".format(n), 
		   "sample_C_padded_20160501.aligned.filled.cropped.62:153.hdf") for n in range(30)]

    f = h5py.File(output_path, "w")
    
    """
    data_dir = '/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data'
    sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153.hdf'
    f_sample = h5py.File(os.path.join(data_dir, sample), "r")
    f.create_dataset("volumes/labels/gt_affinities", data=f_sample["volumes/labels/affinities"])
    """
    mean, unbiased, biased = online_variance(predictions)
    
    f.create_dataset("volumes/labels/mean_pred_affinities", data=mean)
    mean_entropy = -(mean * np.log(mean) + (1. - mean) * np.log(1. - mean))
    f.create_dataset("volumes/labels/mean_entropy", data=mean_entropy)
    f.create_dataset("volumes/labels/var_pred_affinities", data=unbiased)

    mean_ale, unbiased_ale, biased_ale = online_variance(predictions, dataset="volume/labels/sigma", reshape=np.shape(mean))    
    f.create_dataset("volumes/labels/mean_aleatoric", data=mean_ale)
    f.create_dataset("volumes/labels/var_aleatoric", data=unbiased_ale)

    f_0 = h5py.File(predictions[0], "r")
    f.create_dataset("volumes/raw", data=f_0["volumes/raw"])
    f.create_dataset("volumes/labels/0_pred_affinities", data=f_0["volumes/labels/pred_affinities"])

    data_dir = '/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data'
    sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153'
    f_sample = h5py.File(os.path.join(data_dir, sample), "r")
    f.create_dataset("volumes/labels/gt_affinities", data=f_sample["volumes/labels/affinities"])

if __name__ == "__main__":
    combine_predictions("./predictions/140000", "/groups/saalfeld/home/funkej/nils/combined_predictions/"+\
			"sample_C_padded_20160501.aligned.filled.cropped.r0.140000.c30.62:153.hdf")
