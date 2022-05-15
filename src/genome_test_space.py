# -*- coding: utf-8 -*-
"""


"""

import numpy as np
from Bio import SeqIO


# Motif related code pasted here for testing purposes -------------------------

from Bio.Seq import Seq
from Bio import motifs
import warnings


N_INSTANCES = 100

def read_motif_from_fasta(filepath):
    records = list(SeqIO.parse(filepath, 'fasta'))
    binding_sites = [rec.seq for rec in records]
    motif = motifs.create(binding_sites)
    return motif

def is_counts_matrix(matrix_from_file):
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

def split_string(a_string):
    return [x for x in a_string]

def get_instances_array(list_of_strings):
    list_2D = [split_string(s) for s in list_of_strings]
    return np.array(list_2D)

def get_2d_array(motif_matrix):
    ''' Returns a motif 'matrix' of the motif object as a 2D numpy array. '''
    return np.array([motif_matrix[key] for key in motif_matrix.keys()])

def permuted_columns(matrix):
    ''' Returns a copy of the matrix where the columns have been randomly
    permuted. '''
    return np.random.permutation(matrix.T).T

def permuted_column_elements(matrix):
    '''
    Returns a copy of the matrix where the elements in each column have been
    independently permuted at random.
    '''
    ix_i = np.random.sample(matrix.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(matrix.shape[1]), (matrix.shape[0], 1))
    return matrix[ix_i, ix_j]

def round_column(column):
    '''
    Applies the "Largest Remainder Method" to round scaled frequencies - i.e.
    f*K where f is a frequency and K is a scalar - from a column of a scaled
    PWM, so that the sum remains K (i.e., the sum of the frequencies remains 1).
    For K=100, this is a method to deal with percentages, rounding them to
    integers in a way that minimizes the errors.
    '''
    truncated = np.floor(column).astype(int)
    order = np.flip(np.argsort(column - truncated))
    remainder = N_INSTANCES - sum(truncated)
    for i in range(remainder):
        truncated[order[i]] += 1
    return truncated

def get_instances_column(counts_column):
    '''
    Given a column of the counts matrix it returns the corresponding column
    of the instances matrix.
    '''
    inst_column = (['A'] * counts_column[0] +
                   ['C'] * counts_column[1] +
                   ['G'] * counts_column[2] +
                   ['T'] * counts_column[3] )
    return inst_column

def row_to_string(row):
    ''' Converts a 1D array to a string. '''
    return ''.join(list(row))

def get_fake_instances_from_counts(counts_array):
    '''
    Generates fake instances from a counts matrix in the form of a 2D array.
    A list of Bio Seq objects is returned.
    '''
    counts_matrix = counts_array.astype(int)
    instances_matrix = np.apply_along_axis(get_instances_column, 0, counts_matrix)
    instances = list(np.apply_along_axis(row_to_string, 1, instances_matrix))
    instances = [Seq(x) for x in instances]
    return instances

def get_fake_instances_from_pwm(pwm_array):
    '''
    Generates fake instances from a pwm. The frequencies in the PWM are rounded.
    The sum of the weights in each column is controlled by applying the
    "Largest Remainder Method". In this way, the sum of the frequencies is
    guaranteed to be 1 even after the rounding. A list of Bio Seq objects is
    returned.
    '''
    matrix = pwm_array * N_INSTANCES
    counts_matrix = np.apply_along_axis(round_column, 0, matrix)
    instances = get_fake_instances_from_counts(counts_matrix)
    return instances

def read_motif_from_jaspar(filepath):
    f = open(filepath)
    jaspar_motifs = []
    for motif in motifs.parse(f, "jaspar"):
        jaspar_motifs.append(motif)
    f.close()
    if len(jaspar_motifs) > 1:
        warnings.warn("More than one motif were found in jaspar file! " +
                      "Only the first one is returned.")
    jaspar_m = jaspar_motifs[0]
    if is_counts_matrix(jaspar_m.counts):
        return jaspar_m
    else:
        pwm_matrix = get_2d_array(jaspar_m.pwm)
        instances = get_fake_instances_from_pwm(pwm_matrix)
        return motifs.create(instances)

def load_motif(filepath):
    if filepath.endswith('.fasta') or filepath.endswith('.fna'):
        return read_motif_from_fasta(filepath)
    elif filepath.endswith('.jaspar'):
        return read_motif_from_jaspar(filepath)
    else:
        raise TypeError('File format should be fasta or jaspar.')



# -----------------------------------------------------------------------------
# Test with KY555145 (Caulobacter phage Ccr29) scanned with CtrA motif
# -----------------------------------------------------------------------------

from genome import Genome

# CtrA motif
motif = load_motif('../datasets/TF_binding_motifs/CtrA.fasta')

# Custom Genome class for KY555145
my_genome = Genome('../datasets/MGE_sequences/KY555145.gb', 'gb')

print('Description:', my_genome.description)
print('ID:', my_genome.id)

# PSSM scan
my_genome.scan(motif, 0.5, 8.25)

# Statistical analysis
my_genome.analyze_scores()
my_genome.analyze_positional_distribution(50)
my_genome.analyze_intergenicity()

print('Average score:', my_genome.avg_score)
print('Normalized Entr:', my_genome.norm_entropy)
print('Intergenic freq:', my_genome.intergenicity)


# Plot positional distribution as histogram
import matplotlib.pyplot as plt

plt.hist(my_genome.hits['positions'], bins=8)
plt.title('CtrA sites distribution for\n' + my_genome.description)
plt.xlabel('genomic positions')
plt.ylabel('CtrA sites counts')
plt.show()

















