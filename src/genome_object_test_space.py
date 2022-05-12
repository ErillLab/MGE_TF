# -*- coding: utf-8 -*-
"""


"""

import numpy as np
from Bio import SeqIO
import positional_distribution_methods as pdm


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


class Genome():
    
    def __init__(self, filepath, fileformat):
        
        SR = SeqIO.read(filepath, fileformat)
        # For traceability
        self.source = filepath
        
        # SeqRecord
        self.seq = SR.seq
        self.id = SR.id
        self.name = SR.name
        self.description = SR.description
        self.dbxrefs = SR.dbxrefs
        self.features = SR.features
        self.annotations = SR.annotations
        self.letter_annotations = SR.letter_annotations
        self.format = SR.format
        
        # Additional attributes
        self.length = len(SR.seq)
        self.pssm_scores = None
        self.hits = None
        self.n_sites = None
        self.site_density = None
        self.avg_score = None
        self.extremeness = None
        self.counts = None
        self.entropy = None
        self.norm_entropy = None
        self.gini = None
        self.norm_gini = None
        self.evenness = None
        self.new_evenness = None
        
    def scan(self, motif, pseudocount, threshold=None):
        # Generate PWM (and reverse complement)
        pwm = motif.counts.normalize(pseudocounts=pseudocount)
        rpwm = pwm.reverse_complement()
        # Generate PSSM (and reverse complement)
        pssm = pwm.log_odds()
        rpssm = rpwm.log_odds()
        # Scan sequence
        f_scores = pssm.calculate(self.seq)  # Scan on forward strand
        r_scores = rpssm.calculate(self.seq)  # Scan on reverse strand
        effective_scores = self.combine_f_and_r_scores(f_scores, r_scores)
        # Store PSSM-scores
        self.pssm_scores = {'forward': f_scores,
                            'reverse': r_scores,
                            'combined': effective_scores}
        
        # Define hits (if a threshold is specified)
        if threshold:
            hits_scores = effective_scores[effective_scores > threshold]
            hits_positions = np.asarray(np.argwhere(effective_scores > threshold))
            self.hits = {'scores': hits_scores,
                         'positions': hits_positions,
                         'threshold': threshold}
        
    def combine_f_and_r_scores(self, f_scores, r_scores):
        '''
        Combines the PSSM scores on the forward and reverse strand into
        'effective scores', according to the
        method developed in:
        
        Hobbs ET, Pereira T, O'Neill PK, Erill I. A Bayesian inference method for
        the analysis of transcriptional regulatory networks in metagenomic data.
        Algorithms Mol Biol. 2016 Jul 8;11:19. doi: 10.1186/s13015-016-0082-8.
        PMID: 27398089; PMCID: PMC4938975.
        '''
        effective_scores = np.log2(2**f_scores + 2**r_scores)
        return effective_scores
    
    def analyze_hits(self, n_bins, use_double_binning=True):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_hits'.")
        
        # Number of sites
        hits_scores = self.hits['scores']
        hits_positions = self.hits['positions']
        genome_length = self.length
        
        n_sites = len(hits_scores)
        # Site density (sites per thousand bp)
        site_density = 1000 * n_sites / genome_length
        # Average score
        avg_score = hits_scores.mean()
        # Extrmeness
        extremeness = (hits_scores - self.hits['threshold']).sum()
        
        # Study positional distribution
        # Counts in each bin (for Entropy and Gini)
        counts, bins = np.histogram(hits_positions, bins=n_bins, range=(0,genome_length))
        # Entropy, Gini, Evenness
        entr = pdm.entropy(counts)  # Positional entropy
        norm_entr = pdm.norm_entropy(counts)  # Normalized positional entropy
        gini = pdm.gini_coeff(counts)  # Gini coefficient
        norm_gini = pdm.norm_gini_coeff(counts)  # Normalized Gini coefficient
        
        if use_double_binning:
            # The coordinate system will be shifted by half the bin size
            half_bin_size = int((bins[1] - bins[0])/2)
            # Change coordinates (the start point moved from 0 to half_bin_size)
            shifted_matches_positions = []
            for m_pos in hits_positions:
                shifted_m_pos = m_pos - half_bin_size
                if shifted_m_pos < 0:
                    shifted_m_pos += genome_length
                shifted_matches_positions.append(shifted_m_pos)
            shifted_matches_positions.sort()   
            # Counts in each shifted bin (for Entropy and Gini)
            counts_sh, bins_sh = np.histogram(
                shifted_matches_positions, bins=n_bins, range=(0,genome_length))
            # Entropy, Gini, Evenness
            entr_sh = pdm.entropy(counts_sh)
            norm_entr_sh = pdm.norm_entropy(counts_sh)
            gini_sh = pdm.gini_coeff(counts_sh)
            norm_gini_sh = pdm.norm_gini_coeff(counts_sh)
            # Chose frame that detects clusters the most
            entr = min(entr, entr_sh)
            norm_entr = min(norm_entr, norm_entr_sh)
            gini = max(gini, gini_sh)
            norm_gini = max(norm_gini, norm_gini_sh)
        
        # Evenness
        even = pdm.original_evenness(hits_positions, genome_length)
        new_even = pdm.new_evenness(hits_positions, genome_length)
        
        # Set results of the analysis
        self.n_sites = n_sites
        self.site_density = site_density
        self.avg_score = avg_score
        self.extremeness = extremeness
        self.counts = counts
        self.entropy = entr
        self.norm_entropy = norm_entr
        self.gini = gini
        self.norm_gini = norm_gini
        self.evenness = even
        self.new_evenness = new_even
        


# -----------------------------------------------------------------------------
# Test with KY555145 (Caulobacter phage Ccr29) scanned with CtrA motif
# -----------------------------------------------------------------------------

# CtrA motif
motif = load_motif('../datasets/TF_binding_motifs/CtrA.fasta')

# Custom Genome class for KY555145
my_genome = Genome('../datasets/MGE_sequences/KY555145.gb', 'gb')

print('Description:', my_genome.description)
print('ID:', my_genome.id)

my_genome.scan(motif, 0.5, 8.25)
my_genome.analyze_hits(50)

# Print Norm Gini
print('Normalized Gini:', my_genome.norm_gini)


# Plot positional distribution as histogram
import matplotlib.pyplot as plt

plt.hist(my_genome.hits['positions'], bins=8)
plt.title('CtrA sites distribution for\n' + my_genome.description)
plt.xlabel('genomic positions')
plt.ylabel('CtrA sites counts')
plt.show()













'''
class TryGenome(SeqRecord):
    
    def new_method(self):
        print('new method')


class OldGenome():
    
    def __init__(self, SR):
        self.annotations = SR.annotations
        self.dbxrefs = SR.dbxrefs
        self.description = SR.description
        self.features = SR.features
        self.format = SR.format
        self.id = SR.id
        self.letter_annotations = SR.letter_annotations
        self.lower = SR.lower
        self.name = SR.name
        self.reverse_complement = SR.reverse_complement
        self.seq = SR.seq
        self.translate = SR.translate
        self.upper = SR.upper
    
    def new_method(self):
        print('new method')
'''











