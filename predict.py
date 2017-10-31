import sys
from gunpowder import *
from gunpowder.tensorflow import Predict
import os
import json
import numpy as np

data_dir = '/media/nilsec/m3/cremi/20170312_mala_v2'
sample = 'sample_A.augmented.0'

def predict(checkpoint_file, net_io_file, output_dir):
    
    with open(net_io_file, "r") as f:
        net_io_names = json.load(f)

    voxel_size = (40,4,4)
    input_size = Coordinate((21,268,268))*voxel_size
    output_size = Coordinate((21,56,56))*voxel_size
    context = (input_size - output_size)/2

    register_volume_type('PRED_AFFINITIES')
    register_volume_type('SIGMA')
    
    chunk_request = BatchRequest()
    
    chunk_request.add(VolumeTypes.RAW, input_size, voxel_size=voxel_size)
    chunk_request.add(VolumeTypes.PRED_AFFINITIES, output_size, voxel_size=voxel_size)
    chunk_request.add(VolumeTypes.SIGMA, output_size, voxel_size=voxel_size)
     
    source = (Hdf5Source(os.path.join(data_dir, sample + ".hdf"),
                        datasets={VolumeTypes.RAW: 'volumes/raw'},
                        volume_specs = {VolumeTypes.RAW: VolumeSpec(voxel_size=(40,4,4))}) +
              Normalize())
              #Pad({VolumeTypes.RAW: (4000, 400, 400)}))

    with build(source):
        raw_spec = source.spec[VolumeTypes.RAW]

    pipeline = (source + 
                ZeroOutConstSections() + 
                Predict(checkpoint_file,
                        inputs = {net_io_names['raw']: VolumeTypes.RAW,},
                        outputs= {net_io_names['affs']: VolumeTypes.PRED_AFFINITIES,
                                  net_io_names['sigma']: VolumeTypes.SIGMA,},
                        volume_specs = {VolumeTypes.PRED_AFFINITIES: 
                                        VolumeSpec(roi=raw_spec.roi, 
                                                   voxel_size=raw_spec.voxel_size,
                                                   dtype=np.float32),
                                        VolumeTypes.SIGMA:
                                        VolumeSpec(roi=raw_spec.roi,
                                                   voxel_size=raw_spec.voxel_size,
                                                   dtype=np.float32)}) + 
                PrintProfilingStats() + 
                Scan(chunk_request) + 
                Snapshot({VolumeTypes.RAW: 'volumes/raw',
                          VolumeTypes.PRED_AFFINITIES: 'volumes/labels/pred_affinities',
                          VolumeTypes.SIGMA: 'volumes/labels/sigma'},
                          dataset_dtypes={VolumeTypes.PRED_AFFINITIES: np.float32,
                                          VolumeTypes.SIGMA: np.float32,},
                          every=1,
                          output_dir=output_dir,
                          output_filename=sample + '.hdf')
               )

    with build(pipeline):
        raw_spec = source.spec[VolumeTypes.RAW].copy()
        aff_spec = raw_spec.copy()
        #aff_spec.roi = raw_spec.roi.grow(-context, -context)
        
        whole_request = BatchRequest({VolumeTypes.RAW: raw_spec,
                                      VolumeTypes.PRED_AFFINITIES: aff_spec,
                                      VolumeTypes.SIGMA: aff_spec})
        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))
        pipeline.request_batch(whole_request)

if __name__ == "__main__":
    #checkpoint_file = sys.argv[1]
    #net_io_file = sys.argv[2]
    #output_dir = sys.argv[3]
    checkpoint_file = "./models/bunet_checkpoint_700000"
    net_io_file = "./models/net_io_names.json"
    output_dir = "/media/nilsec/m3/predictions_mc_5/700000_{}"
    for n in range(7, 50):
        print("Predict {}/{}".format(n, 50))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        predict(checkpoint_file, net_io_file, output_dir.format(n))
