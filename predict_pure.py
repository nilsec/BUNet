import sys
import h5py
from gunpowder import *
from gunpowder.tensorflow import Predict
import os
import json
import numpy as np

data_dir = '/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data'
sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153'

#data_dir = "."
#sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153.0:268.0:268'

def predict(checkpoint_file, net_io_file, output_dir):
    
    with open(net_io_file, "r") as f:
        net_io_names = json.load(f)

    voxel_size = (40,4,4)
    input_size = Coordinate((84,268,268))*voxel_size
    output_size = Coordinate((56,56,56))*voxel_size
    context = (input_size - output_size)/2

    register_volume_type('PRED_AFFINITIES')
    
    chunk_request = BatchRequest()
    
    chunk_request.add(VolumeTypes.RAW, input_size, voxel_size=voxel_size)
    chunk_request.add(VolumeTypes.PRED_AFFINITIES, output_size, voxel_size=voxel_size)
    
    #f=h5py.File(os.path.join(data_dir, sample + ".hdf"), "r")
    #raw_roi = Roi(offset=np.array([0,0,0]), shape=np.array(f["volumes/raw"].shape) * np.array([40,4,4]))
    #f.close()
 
    source = (Hdf5Source(os.path.join(data_dir, sample + ".hdf"),
                        datasets={VolumeTypes.RAW: 'volumes/raw'},
                        volume_specs = {VolumeTypes.RAW: VolumeSpec(voxel_size=(40,4,4))}) + # VS(roi=raw_roi)
              Normalize())
              #Pad({VolumeTypes.RAW: (4000, 400, 400)}))

    with build(source):
        raw_spec = source.spec[VolumeTypes.RAW]

    #print "raw spec roi", raw_spec.roi

    pipeline = (source + 
		IntensityScaleShift(2, -1) +
                ZeroOutConstSections() + 
                Predict(checkpoint_file,
                        inputs = {net_io_names['raw']: VolumeTypes.RAW,},
                        outputs= {net_io_names['affs']: VolumeTypes.PRED_AFFINITIES,
                                 },
                        volume_specs = {VolumeTypes.PRED_AFFINITIES: 
                                        VolumeSpec(roi=raw_spec.roi, 
                                                   voxel_size=raw_spec.voxel_size,
                                                   dtype=np.float32)}) + 
                PrintProfilingStats() + 
                Scan(chunk_request) + 
                Snapshot({VolumeTypes.RAW: 'volumes/raw',
                          VolumeTypes.PRED_AFFINITIES: 'volumes/labels/pred_affinities'},
                          dataset_dtypes={VolumeTypes.PRED_AFFINITIES: np.float32,},
                          every=1,
                          output_dir=output_dir,
                          output_filename=sample + '.hdf')
               )

    with build(pipeline):
        raw_spec = source.spec[VolumeTypes.RAW].copy()
        aff_spec = raw_spec.copy()
        aff_spec.roi = raw_spec.roi.grow(-context, -context)
        
        whole_request = BatchRequest({VolumeTypes.RAW: raw_spec,
                                      VolumeTypes.PRED_AFFINITIES: aff_spec})
        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))
        pipeline.request_batch(whole_request)

if __name__ == "__main__":
    #checkpoint_file = sys.argv[1]
    #net_io_file = sys.argv[2]
    #output_dir = sys.argv[3]
    checkpoint_file = "./models/run_11/bunet_checkpoint_96000"
    net_io_file = "./models/run_11/net_io_names.json"
    output_dir = "./predictions/run_11/96000_large/p_{}"
    for n in range(10):
        print("Predict {}/{}".format(n, 10))
        if not os.path.exists(output_dir.format(n)):
            os.makedirs(output_dir.format(n))
        predict(checkpoint_file, net_io_file, output_dir.format(n))
