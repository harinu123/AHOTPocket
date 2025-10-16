import tensorflow as tf
import argparse
from models import MQAModel
import numpy as np
from glob import glob
import mdtraj as md
import os

from validate_performance_on_xtals import process_strucs, predict_on_xtals

def make_predictions(pdb_paths, model, nn_path, debug=False, output_basename=None):
    '''
        pdb_paths : list of pdb paths
        model : MQAModel corresponding to network in nn_path
        nn_path : path to checkpoint files
    '''
    strucs = [md.load(s) for s in pdb_paths]
    X, S, mask = process_strucs(strucs)
    if debug:
        np.save(f'{output_basename}_X.npy', X)
        np.save(f'{output_basename}_S.npy', S)
        np.save(f'{output_basename}_mask.npy', mask)
    predictions = predict_on_xtals(model, nn_path, X, S, mask)
    return predictions

# main method
def main(fpaths, outnames, dirnames):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for idx in range(len(fpaths)): 
        strucs = [fpaths[idx]]
        output_name = outnames[idx]
        output_folder = dirnames[idx]

        # debugging mode can be turned on to output protein features and sequence
        debug = False

        # Load MQA Model used for selected NN network
        nn_path = os.path.join(script_dir, "../models/pocketminer")
        DROPOUT_RATE = 0.1
        NUM_LAYERS = 4
        HIDDEN_DIM = 100
        model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                         hidden_dim=(16, HIDDEN_DIM),
                         num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
        
        
        try:
            if debug:
                output_basename = f'{output_folder}/{output_name}'
                predictions = make_predictions(strucs, model, nn_path, debug=True, output_basename=output_basename)
            else:
                predictions = make_predictions(strucs, model, nn_path)
        except ValueError:
            continue

        # output filename can be modified here
        np.save(f'{output_folder}/{output_name}-preds.npy', predictions)
        np.savetxt(os.path.join(output_folder,f'{output_name}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="input pdb file", default='../data/ACE2.pdb')
    parser.add_argument("-o", type=str, help="output name", default="ACE2")
    parser.add_argument("-d", type=str, help="output directory", default=".")
    args = parser.parse_args()
    main(args.f, args.o, args.d)
