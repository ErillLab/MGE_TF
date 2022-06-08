# -*- coding: utf-8 -*-
"""


"""

import numpy as np
import random
import copy
from Bio.Seq import Seq
from genome import Genome
import warnings


class MGE():
    
    def __init__(self, filepath, fileformat):
        
        self.original = Genome(filepath, fileformat)
        self.pseudogenomes = []
        self.source = (filepath, fileformat)
        self.pseudo_g_counter = 0
        self.n_bins = 50  # !!! Set from config
        
        # p-values
        self.site_density = None
        self.avg_score = None
        self.extremeness = None
        self.entropy = None
        self.norm_entropy = None
        self.gini = None
        self.norm_gini = None
        self.evenness = None
        self.new_evenness = None
        self.intergenicity = None
    
    def set_pseudogenomes(self, n_pseudogenomes, kmer_len):
        '''
        Sets the  pseudogenomes  attribute: a list of pseudogenomes.
        '''
        for i in range(n_pseudogenomes):
            pseudogenome = self.get_pseudogenome(kmer_len)
            self.pseudogenomes.append(pseudogenome)
    
    def get_pseudogenome(self, kmer_len):
        '''
        It generates a 'pseudogenome'. For each genomic unit in the original
        genome sequence, a k-sampled sequence is generated. The pseudogenome is
        composed of these pseudo-units (k-sampled sequences) joined in the same
        order as their corresponding units appear on the original genome, to
        preserve genomic structure. In other words, each genomic unit is
        independently 'k-sampled' (using the 'get_k_sampled_sequence' method).
        '''
        pseudogenome = copy.deepcopy(self.original)
        self.clear_stats(pseudogenome)
        pseudogenome.seq = Seq("")
        units_bounds = pseudogenome.genomic_units['bounds']
        for i in range(len(units_bounds)-1):
            unit = self.original.seq[units_bounds[i]: units_bounds[i+1]]
            pseudogenome.seq += self.get_k_sampled_sequence(unit, kmer_len)
        
        # The permuted genome is assigned a unique ID
        self.increase_pseudo_g_counter()
        pseudogenome.id = str(pseudogenome.id) + '_' + str(self.pseudo_g_counter)
        pseudogenome.name = str(pseudogenome.name) + ' pseudo_' + str(self.pseudo_g_counter)
        pseudogenome.description = str(pseudogenome.description) + 'pseudo_' + str(self.pseudo_g_counter)
        return pseudogenome
    
    def increase_pseudo_g_counter(self):
        self.pseudo_g_counter += 1
    
    def get_k_sampled_sequence(self, sequence, k):
        '''
        All kmers are stored. Than sampled without replacement.
        Example with k = 3:
        ATCAAAGTCCCCGTACG
        for which 3-mers are
        ATC, TCA, CAA, AAA, AAG, ...
        A new sequence is generated by sampling (without replacement) from that
        complete set of k-mers.
        The nucleotide content (1-mers content) may not be perfectly identical
        because of overlap between k-mers that are then randomly sampled.
        '''
        
        if k > 1:
            n_kmers = len(sequence) // k
            n_nuclotides_rem = len(sequence) % k
            
            all_kmers = self.get_all_kmers(sequence, k)
            sampled_seq_list = random.sample(all_kmers, n_kmers)
            n_nucleotides = random.sample(str(sequence), n_nuclotides_rem)
            sampled_seq_list += n_nucleotides
        
        else:
            sampled_seq_list = random.sample(str(sequence), len(sequence))
        
        sampled_seq = Seq("".join(sampled_seq_list))
        return sampled_seq
    
    def get_all_kmers(self, seq, k):
        '''
        Returns the list of all the k-mers of length k in sequence seq.
        '''
        return [str(seq)[i:i+k] for i in range(len(seq)-k+1)]
    
    def clear_stats(self, genome):
        ''' Ensures all the statistics in the 'stats' list are set to None. '''
        stats = ['n_sites', 'site_density', 'avg_score', 'extremeness',
                 'counts', 'entropy', 'norm_entropy', 'gini', 'norm_gini',
                 'evenness', 'new_evenness', 'intergenicity']
        for stat in stats:
            vars(genome)[stat] = None
    
    def scan(self, motif, pseudocount, threshold=None):
        '''
        Scans the original genome and all the pseudogenomes with the PSSM of a
        given motif.
        '''
        self.original.scan(motif, pseudocount, threshold=threshold)
        for pg in self.pseudogenomes:
            pg.scan(motif, pseudocount, threshold=threshold)
    
    def analyze_scores(self):
        ''' Sets the p-value for the statistics related to the PSSM-scores. '''
        genomes = [self.original] + self.pseudogenomes
        for g in genomes:
            g.analyze_scores()
        # Set p-values
        self.set_pvalue('avg_score', 'greater')
        self.set_pvalue('extremeness', 'greater')
    
    def analyze_positional_distribution(self):
        ''' Sets the p-value for the statistics related to the positional
        distribution. '''
        genomes = [self.original] + self.pseudogenomes
        for g in genomes:
            g.analyze_positional_distribution(self.n_bins)
        # Set p-values
        self.set_pvalue('entropy', 'smaller')
        self.set_pvalue('norm_entropy', 'smaller')
        self.set_pvalue('gini', 'greater')
        self.set_pvalue('norm_gini', 'greater')
        self.set_pvalue('evenness', 'greater')
        self.set_pvalue('new_evenness', 'smaller')
    
    def analyze_intergenicity(self):
        ''' Sets the p-value for the statistics related to the intergenicity. '''
        genomes = [self.original] + self.pseudogenomes
        for g in genomes:
            g.analyze_intergenicity()
        # Set p-values
        self.set_pvalue('intergenicity', 'greater')
    
    def set_pvalue(self, metric, alternative):
        '''
        Estimates the p-value for a given metric, and a given alternative
        hypothesis. The estimate is based on the frequency of pseudogenomes
        that can reproduce the results observed on the original genome.
        '''
        control_values = []
        for genome in self.pseudogenomes:
            control_values.append(vars(genome)[metric])
        
        if None in control_values:
            raise ValueError('The value of ' + str(metric) +
                             ' is not set for all pseudogenomes.')
        
        valid_values = [x for x in control_values if not isinstance(x, str)]
        if len(valid_values) < len(control_values):
            warnings.warn(("Only {}/{} values of {} were valid and used to "
                           "estimate the p-value.").format(len(valid_values),
                          len(control_values), metric))
        
        control = np.array(valid_values)
        obs = vars(self.original)[metric]
        
        if not isinstance(obs, (int, float)):
            p_val = 'no_obs'
        
        elif len(control)==0:
            p_val = 'no_control_vals'
        
        else:
            if alternative == 'greater':
                p_val = (control >= obs).sum()/len(control)
            elif alternative == 'smaller':
                p_val = (control <= obs).sum()/len(control)
            else:
                raise ValueError('alternative should be "greater" or "smaller".')
        
        # Set p_value
        vars(self)[metric] = p_val














