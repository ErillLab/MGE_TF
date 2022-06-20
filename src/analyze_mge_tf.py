# -*- coding: utf-8 -*-
"""



"""

import json
import os
from mpi4py import MPI

from mge import MGE
from tf import TF
from mge_tf import MGE_TF


def go():
    
    for tf_name in TF_LIST:
        
        tf_path = '../datasets/' + TF_DIRNAME + "/" + tf_name
        my_tf = TF(tf_path, 'fasta')
        my_tf.set_permuted_motifs(N_PERMUTED_MOTIFS)
        
        for mge_name in MGE_LIST:
            
            mge_path = '../datasets/' + MGE_DIRNAME + "/" + mge_name
            my_mge = MGE(mge_path, 'gb')
            my_mge.set_pseudogenomes(N_PSEUDOGENOMES, kmer_len=KMER_LEN)
            
            # MGE-TF analysis
            my_mge_tf = MGE_TF(my_mge, my_tf)
            
            # Compute values
            my_mge_tf.compute_motif_specific_vals()
            # Analyze hits
            my_mge_tf.analyze_scores()
            my_mge_tf.analyze_positional_distribution()
            my_mge_tf.analyze_intergenicity()
            
            # Save results
            my_mge_tf.save_p_vals(OUT_DIRNAME)
            my_mge_tf.save_motif_specific_vals(OUT_DIRNAME)


def set_up():
    
    global N_PSEUDOGENOMES
    global KMER_LEN
    global N_PERMUTED_MOTIFS
    global MGE_DIRNAME
    global TF_DIRNAME
    global OUT_DIRNAME
    
    config_filename = 'config.json'
    
    with open(config_filename, 'r') as f:
        config = json.load(f)
    
    N_PSEUDOGENOMES = config['n_pseudogenomes']
    KMER_LEN = config['kmer_len']
    N_PERMUTED_MOTIFS = config['n_permuted_motifs']
    MGE_DIRNAME = config['mge_dirname']
    TF_DIRNAME = config['tf_dirname']
    OUT_DIRNAME = config['out_dirname']


def set_MPI():
    
    global MGE_LIST
    global TF_LIST
    
    # Set MPI
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank()  
    p = comm.Get_size()
    
    MGE_LIST = os.listdir('../datasets/' + MGE_DIRNAME)
    TF_LIST = os.listdir('../datasets/' + TF_DIRNAME)
    
    # Assign a subset of TF_LIST to each process
    n = len(TF_LIST)  # total number of TFs
    q = n // p  # quotient
    r = n % p  # remainder
    
    if rank < r:
        n_i = q + 1  # number of TFs assigned
        start = rank * n_i
    else:
        n_i = q  # number of TFs assigned
        start = rank * n_i + r
    
    TF_LIST = TF_LIST[start : start+n_i]
    
    # print('\nHello from process {}/{}'.format(rank, p))
    # print('My TF list is:', TF_LIST)


if __name__ == "__main__":
    # Set parameters
    set_up()
    # Distribute workload over processes
    set_MPI()
    # Run pipeline
    go()
















