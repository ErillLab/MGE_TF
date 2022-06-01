# -*- coding: utf-8 -*-
"""


"""

import numpy as np
from Bio import motifs
from Bio import SeqIO
from Bio.Seq import Seq
import warnings


class TF():
    
    def __init__(self, filepath, fileformat):
        
        self.original = self.load_motif(filepath, fileformat)
        self.permuted = None
        self.source = (filepath, fileformat)
        self.perm_counter = 0
        self.n_instances = 100  # !!! Set from config
    
    
    # Loading motifs
    
    def load_motif(self, filepath, fileformat):
        if fileformat == 'fasta':
            motif = self.read_motif_from_fasta(filepath)
        elif fileformat == 'jaspar':
            motif = self.read_motif_from_jaspar(filepath)
        else:
            raise TypeError('File format should be "fasta" or "jaspar".')
        name = filepath.split('/')[-1]
        motif.name = '.'.join(name.split('.')[:-1])
        return motif
    
    def read_motif_from_fasta(self, filepath):
        records = list(SeqIO.parse(filepath, 'fasta'))
        binding_sites = [rec.seq for rec in records]
        motif = motifs.create(binding_sites)
        return motif

    def read_motif_from_jaspar(self, filepath):
        '''
        Reads motif from jaspar file. It ensures that
        '''
        f = open(filepath)
        jaspar_motifs = []
        for motif in motifs.parse(f, "jaspar"):
            jaspar_motifs.append(motif)
        f.close()
        if len(jaspar_motifs) > 1:
            warnings.warn("More than one motif were found in jaspar file! " +
                          "Only the first one is returned.")
        jaspar_m = jaspar_motifs[0]
        
        # Generate fake instances
        if self.is_counts_matrix(jaspar_m.counts):
            counts_matrix = self.get_2d_array(jaspar_m.counts)
            instances = self.get_fake_instances_from_counts_matrix(counts_matrix)
        else:
            instances = self.get_fake_instances_from_pwm(jaspar_m.pwm)
        return motifs.create(instances)
    
    def is_counts_matrix(self, matrix_from_file):
        '''
        Checks that the matrix is a counts matrix and not a PWM.
        If the sum over the first column is 1 or (close to 1) it means it's a PWM,
        and the function will return False. If the closest integer to the sum is
        larger than 1 the function returns True.
        '''
        a1 = matrix_from_file['A'][0]
        c1 = matrix_from_file['C'][0]
        g1 = matrix_from_file['G'][0]
        t1 = matrix_from_file['T'][0]
        return round(a1 + c1 + g1 + t1) > 1
    
    def get_2d_array(self, motif_matrix):
        ''' Returns a motif 'matrix' of the motif object as a 2D numpy array. '''
        return np.array([motif_matrix[key] for key in motif_matrix.keys()])
    
    def get_fake_instances_from_pwm(self, motif_pwm):
        '''
        Generates fake instances from a pwm. The frequencies in the PWM are
        rounded to become multiples of 1/n_instances.
        The sum of the weights in each column is controlled by applying the
        "Largest Remainder Method". In this way, the sum of the frequencies is
        guaranteed to be 1 even after the rounding. A list of Bio Seq objects
        is returned (a list of "fake instances" for the motif).
        '''
        pwm_array = self.get_2d_array(motif_pwm)  # As 2D numpy array
        matrix = pwm_array * self.n_instances
        counts_matrix = np.apply_along_axis(self.round_column, 0, matrix)
        instances = self.get_fake_instances_from_counts_matrix(counts_matrix)
        return instances
    
    def round_column(self, column):
        '''
        Applies the "Largest Remainder Method" to round scaled frequencies - i.e.
        f*K where f is a frequency and K is a scalar - from a column of a scaled
        PWM, so that the sum remains K (i.e., the sum of the frequencies remains 1).
        For K=100, this is a method to deal with percentages, rounding them to
        integers in a way that minimizes the errors.
        '''
        truncated = np.floor(column).astype(int)
        order = np.flip(np.argsort(column - truncated))
        remainder = self.n_instances - sum(truncated)
        for i in range(remainder):
            truncated[order[i]] += 1
        return truncated
        
    def get_fake_instances_from_counts_matrix(self, counts_array):
        '''
        Generates fake instances from a counts matrix in the form of a 2D array.
        A list of Bio Seq objects is returned.
        '''
        counts_matrix = counts_array.astype(int)
        instances_matrix = np.apply_along_axis(self.get_instances_column, 0, counts_matrix)
        instances = list(np.apply_along_axis(self.row_to_string, 1, instances_matrix))
        instances = [Seq(x) for x in instances]
        return instances
    
    def get_instances_column(self, counts_column):
        '''
        Given a column of the counts matrix it returns the corresponding column
        of the instances matrix. Used to generate fake instances from counts.
        '''
        inst_column = (['A'] * counts_column[0] +
                       ['C'] * counts_column[1] +
                       ['G'] * counts_column[2] +
                       ['T'] * counts_column[3] )
        return inst_column
    
    def row_to_string(self, row):
        ''' Converts a 1D array to a string. '''
        return ''.join(list(row))
    
    
    # Permuting motifs
    
    def set_permuted_motifs(self, n_permuted_motifs):
        '''
        Returns a list of permutations of the input motif. The set_size
        paramter defines the number of permuted motifs to be generated.
        '''
        self.permuted = []
        for i in range(n_permuted_motifs):
            perm_motif = self.get_permuted_motif()
            self.permuted.append(perm_motif)
    
    def get_permuted_motif(self, perm_column_elements=False):
        '''
        Takes as input a Bio motif object and returns a permuted version of the
        input motif (also as a Bio motif object).
        '''
        
        # Apply permutation
        inst_matrix = self.get_instances_array()
        perm_inst_mat = self.get_permuted_columns(inst_matrix)
        if perm_column_elements:
            perm_inst_mat = self.permuted_column_elements(perm_inst_mat)
        perm_instances = list(np.apply_along_axis(self.row_to_string, 1, perm_inst_mat))
        perm_instances = [Seq(x) for x in perm_instances]
        
        # Create Bio motif object
        perm_motif = motifs.create(perm_instances)
        
        # Assign a unique name to the permuted motif
        self.increase_perm_counter()
        perm_motif.name = self.original.name + "_" + str(self.perm_counter)
        return perm_motif
    
    def get_instances_array(self):
        ''' Returns instances as a 2D array of characters. '''
        list_of_strings = self.original.instances
        list_2D = []
        for a_string in list_of_strings:
            list_2D.append([x for x in a_string])
        return np.array(list_2D)
    
    def get_permuted_columns(self, matrix):
        ''' Returns a copy of the matrix where the columns have been randomly
        permuted. '''
        return np.random.permutation(matrix.T).T
    
    def permuted_column_elements(self, matrix):
        '''
        Returns a copy of the matrix where the elements in each column have been
        independently permuted at random.
        '''
        ix_i = np.random.sample(matrix.shape).argsort(axis=0)
        ix_j = np.tile(np.arange(matrix.shape[1]), (matrix.shape[0], 1))
        return matrix[ix_i, ix_j]
    
    def increase_perm_counter(self):
        self.perm_counter += 1












