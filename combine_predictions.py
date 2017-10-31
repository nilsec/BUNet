import numpy as np
import h5py
from scipy.special import expit
import os

def online_variance(prediction_paths, dataset="volumes/labels/pred_affinities"):
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
        x = np.array(f[dataset].value, dtype=np.float32)
        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta * delta2

    if n<2:
        return np.nan
    else:
        return mean, M2/(n-1), M2/n

def combine_predictions(base_dir):
    dirs = [os.path.join(base_dir, top_dir) for top_dir in os.listdir(base_dir)]
    predictions = [os.path.join(comb_dir, "sample_A.augmented.0.hdf") for comb_dir in dirs]
    mean, unbiased, biased = online_variance(predictions[:10])
    f = h5py.File(os.path.join(base_dir, "combined.hdf"))
    f.create_dataset("volumes/labels/pred_affinities", data=mean)
    f.create_dataset("volumes/labels/aleatoric", data=unbiased)

    f_0 = h5py.File(predictions[0])
    f.create_dataset("volumes/raw", data=f_0["volumes/raw"]) 
    

if __name__ == "__main__":
    combine_predictions("/media/nilsec/Backup/predictions_mc_3")
