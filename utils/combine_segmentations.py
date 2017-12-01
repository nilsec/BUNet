import h5py
import numpy as np
import os

def overlay(fseg_0, 
	    fseg_1, 
	    dset,
	    output_dir):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	

	f0 = h5py.File(fseg_0, "r")
	f1 = h5py.File(fseg_1, "r")
	
	seg0 = f0[dset].value
	seg1 = f1[dset].value

	dt = seg0.dtype

	f0.close()
	f1.close()
	
	max_id0 = np.max(seg0)
	max_id1 = np.max(seg1)

	diff_segs = []

	matches = 1
	for id0 in range(max_id0):
	    print id0,"/",max_id0

	    mask0 = seg0 == id0
	    n0 = np.sum(mask0)
	    if not n0:
		continue

	    nz = np.nonzero(mask0)
	    if max(nz[0]) - min(nz[0]) < 5:
		continue

	    if max(nz[1]) - min(nz[1]) < 10:
		continue
	
	    if max(nz[2]) - min(nz[2]) < 10:
		continue

	    match = False
	    for id1 in range(max_id1):
		if match:
		    continue

		mask1 = seg1 == id1
		intersect = np.logical_and(mask0, mask1)
		n_intersect = np.sum(intersect)
		if n_intersect:
			n1 = np.sum(mask1)
	
			if n_intersect > 0.3 * np.max([n0, n1]):
				# Check if two segments are the same process defined by sharing at least
				# 30% the voxels of the larger segment
				match=True
				print "Found match..."

				if n_intersect < 0.5 * np.max([n0, n1]):
					print "Found difference"
					affix = "." + fseg_0.split(".")[-1]
					f0 = h5py.File(output_dir +  "/{}_seg0".format(matches) + affix, "w")
					f1 = h5py.File(output_dir +  "/{}_seg1".format(matches) + affix, "w")
  
					f0.create_dataset("volumes/labels/segmentation", 
							  data=np.array(mask0 * (2*matches) , dtype=dt))
					f1.create_dataset("volumes/labels/segmentation", 
							  data=np.array(mask1 * (2*matches + 1), dtype=dt))

					f0.close()
					f1.close()
					matches += 1




					break
	

if __name__ == "__main__":
	base_dir = "../segmentations/run_22/60000/"
	overlay(base_dir + "p_0.h5", base_dir + "p_1.h5", dset="volumes/labels/segmentation", output_dir = "../segmentations/run_22/60000/matches_2")

		
	
					
			

	
