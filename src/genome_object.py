# -*- coding: utf-8 -*-
"""


"""

import numpy as np
from Bio import SeqIO
import positional_distribution_methods as pdm


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
        pwm = motif.counts.normalize(pseudocounts=pseudocount)
        rpwm = pwm.reverse_complement()
        # Generate PSSM (and reverse complement)
        pssm = pwm.log_odds()
        rpssm = rpwm.log_odds()
        f_scores = pssm.calculate(self.seq)  # Scan on forward strand
        r_scores = rpssm.calculate(self.seq)  # Scan on reverse strand
        effective_scores = self.combine_f_and_r_scores(f_scores, r_scores)
        
        self.pssm_scores = {'forward': f_scores,
                            'reverse': r_scores,
                            'combined': effective_scores}
        
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
    
    def analyze_putative_sites(self, n_bins, use_double_binning):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_putative_sites'.")
        
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
        


















