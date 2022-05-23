# -*- coding: utf-8 -*-
"""


"""

import numpy as np
from Bio import SeqIO


class Genome():
    
    def __init__(self, filepath, fileformat):
        
        SR = SeqIO.read(filepath, fileformat)
        # For traceability
        self.source = (filepath, fileformat)
        
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
        self.intergenicity = None
    
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
            hits_positions = np.argwhere(effective_scores > threshold).flatten()
            self.hits = {'scores': hits_scores,
                         'positions': hits_positions,
                         'threshold': threshold,
                         'motif_length': pssm.length}
        
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
    
    def get_entropy(self, counts):
        counts_vector = np.array(counts)
        frequencies = counts_vector / counts_vector.sum()
        H = 0
        for p in frequencies:
            if p != 0:
                H -= p * np.log(p)
        return H
    
    def get_norm_entropy(self, counts):
        '''
        Entropy divided by the maximum entropy possible with that number of counts
        and that number of bins.
        
        Parameters
        ----------
        counts : array-like object
            Counts associated to each class.

        Returns
        -------
        rel_possible_ent : float
            Ranges from 0, when entropy is 0, to 1, when entropy is the maximum
            possible entropy. The maximum possible entropy depends on the number of
            counts and bins, and it's achieved when the counts are distributed as
            evenly as possible among the bins. Example: with 10 bins and 12 counts,
            maximum possible entropy is the entropy of the distribution where 2
            bins contain 2 counts, and 8 bins contain 1 count.
        '''
        
        counts_vector = np.array(counts)
        n_obs = counts_vector.sum()
        n_bins = len(counts_vector)
        if n_obs == 1:
            rel_possible_ent = 1
        else:
            # Compute max entropy possible with that number of obs and bins
            quotient = n_obs // n_bins
            remainder = n_obs % n_bins
            chunk_1 = np.repeat(quotient, n_bins - remainder)
            chunk_2 = np.repeat(quotient + 1, remainder)
            values = np.hstack((chunk_1, chunk_2))  # values distr as evenly as possible
            max_possible_entropy = self.get_entropy(values)
            # Compute relative entropy
            rel_possible_ent = self.get_entropy(counts) / max_possible_entropy
        return rel_possible_ent
    
    def get_gini_coeff(self, counts):
        '''
        Gini coefficient measures distribution inequality.
    
        Parameters
        ----------
        counts : array-like object
            Values associated to each class.
            They don't need to be already sorted and/or normalized.
    
        Returns
        -------
        gini_coeff : float
            Ranges from 0 (perfect equality) to 1 (maximal inequality).
        '''
        
        values = np.array(counts)
        norm_values = values / values.sum()  # normalize
        
        # Generate Lorenz curve
        norm_values.sort()
        cum_distr = np.cumsum(norm_values)
        cum_distr = list(cum_distr)
        cum_distr.insert(0, 0)
        
        # Get area under Lorenz curve
        n_classes = len(cum_distr)-1
        under_lorenz = np.trapz(y = cum_distr, dx = 1/n_classes)
        
        # Area under Perfect Equality curve
        # It's the area of a triangle with base = 1 and height = 1
        under_PE = 0.5
        
        # Compute Gini coefficient
        gini_coeff = (under_PE - under_lorenz) / under_PE
        
        return gini_coeff
    
    def get_norm_gini_coeff(self, counts):
        '''
        Normalized Gini coefficient.
        The minimum and maximum possible Gini coefficient with that number of
        bins and observations are computed. Then, norm_Gini_coefficient is
        defined as
        norm_Gini_coefficient := (Gini - min_Gini) / (max_Gini - min_Gini)
    
        Parameters
        ----------
        counts : array-like object
            Values associated to each class.
            They don't need to be already sorted and/or normalized.
    
        Returns
        -------
        norm_gini_coeff : float
            Ranges from 0 (minimal inequality possible) to 1 (maximal
            inequality possible).
        '''
    
        # Compute Gini coefficient
        nuber_of_bins = len(counts)
        number_of_obs = np.array(counts).sum()
        Gini = self.get_gini_coeff(counts)
        
        # Compute minimum possible Gini coefficient
        quotient = number_of_obs // nuber_of_bins
        remainder = number_of_obs % nuber_of_bins
        chunk_1 = np.repeat(quotient, nuber_of_bins - remainder)
        chunk_2 = np.repeat(quotient + 1, remainder)
        vect = np.hstack((chunk_1, chunk_2))  # values distr as evenly as possible
        min_Gini = self.get_gini_coeff(vect)
        
        # Compute maximum possible Gini coefficient
        chunk_1 = np.repeat(0, nuber_of_bins - 1)
        chunk_2 = np.repeat(number_of_obs, 1)
        vect = np.hstack((chunk_1, chunk_2))  # values distr as unevenly as possible
        vect = [int(v) for v in vect]
        max_Gini = self.get_gini_coeff(vect)
        
        # Compute normalized Gini coefficient
        if max_Gini - min_Gini == 0:
            norm_gini = 0
        else:
            norm_gini = (Gini - min_Gini) / (max_Gini - min_Gini)
        
        return norm_gini
    
    def get_hits_distances(self, circular=True):
        
        distances = []
        hits_positions = self.hits['positions']
        for i in range(len(hits_positions)):
            if i == 0:
                distance = self.length - hits_positions[-1] + hits_positions[i]
            else:
                distance = hits_positions[i] - hits_positions[i-1]
            distances.append(distance)
        return distances

    def get_original_evenness(self):
        '''
        Evenness as defined in Philip and Freeland (2011).
        It's the variance of the distances between consecutive (sorted)
        datapoints.
        '''
        if len(self.hits['positions']) < 2:
            return 'Not enough sites'
        
        intervals = self.get_hits_distances()
        return np.var(intervals)

    def get_norm_evenness(self):
        '''
        Normalize evenness.
        Norm_Evenness = Evenness / Max_Evenness
        '''
        if len(self.hits['positions']) < 2:
            return 'Not enough sites'
        
        intervals = self.get_hits_distances()
        var = np.var(intervals)
        
        n_intervals = len(intervals)
        mean = self.length/n_intervals
        max_var = ((n_intervals - 1) * mean**2 + (self.length - mean)**2)/n_intervals
        norm_var = var / max_var
        return norm_var

    def get_new_evenness(self):
        
        if len(self.hits['positions']) < 2:
            return 'Not enough sites'
        
        norm_var = self.get_norm_evenness()
        # Transform so that large evenness values mean very even distribution
        # (it's the opposite in the original evenness definition)
        new_evenness = 1 - norm_var
        return new_evenness
    
    def analyze_scores(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_positional_distribution'.")
                
        # Set average score
        self.avg_score = self.hits['scores'].mean()
        # Set extrmeness
        self.extremeness = (self.hits['scores'] - self.hits['threshold']).sum()
    
    def set_counts(self, n_bins, use_double_binning):
        
        # Counts in each bin (for Entropy and Gini)
        counts, bins = np.histogram(
            self.hits['positions'], bins=n_bins, range=(0, self.length))
        counts_shifted = None
        
        if use_double_binning:
            # The coordinate system will be shifted by half the bin size
            half_bin_size = int((bins[1] - bins[0])/2)
            # Change coordinates (start point moved from 0 to half_bin_size)
            shifted_matches_positions = []
            for m_pos in self.hits['positions']:
                shifted_m_pos = m_pos - half_bin_size
                if shifted_m_pos < 0:
                    shifted_m_pos += self.length
                shifted_matches_positions.append(shifted_m_pos)
            shifted_matches_positions.sort()   
            # Counts in each shifted bin (for Entropy and Gini)
            counts_shifted, bins_shifted = np.histogram(
                shifted_matches_positions, bins=n_bins, range=(0, self.length))
        
        self.counts = {'regular_binning': counts,
                       'shifted_binning': counts_shifted}
    
    def analyze_positional_distribution(self, n_bins, use_double_binning=True):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_positional_distribution'.")
        
        # Number of predicted binding sites
        self.n_sites = len(self.hits['scores'])
        # Site density (sites per thousand bp)
        self.site_density = 1000 * self.n_sites / self.length
        
        # Set counts (regular binning and shifted binning)
        self.set_counts(n_bins, use_double_binning)
        counts_regular, counts_shifted = self.counts.values()
        
        # Entropy, Normalized entropy, Gini, Normalized Gini (regular frame)
        entr = self.get_entropy(counts_regular)
        norm_entr = self.get_norm_entropy(counts_regular)
        gini = self.get_gini_coeff(counts_regular)
        norm_gini = self.get_norm_gini_coeff(counts_regular)
        
        if use_double_binning:
            # Entropy, Normalized entropy, Gini, Normalized Gini (shifted frame)
            entr_sh = self.get_entropy(counts_shifted)
            norm_entr_sh = self.get_norm_entropy(counts_shifted)
            gini_sh = self.get_gini_coeff(counts_shifted)
            norm_gini_sh = self.get_norm_gini_coeff(counts_shifted)
            
            # Chose frame that detects clusters the most
            entr = min(entr, entr_sh)
            norm_entr = min(norm_entr, norm_entr_sh)
            gini = max(gini, gini_sh)
            norm_gini = max(norm_gini, norm_gini_sh)
        
        # Set entropy, normalized entropy, Gini and normalized Gini
        self.entropy = entr
        self.norm_entropy = norm_entr
        self.gini = gini
        self.norm_gini = norm_gini
        
        # Set original evenness and new evenness
        self.evenness = self.get_original_evenness()
        self.new_evenness = self.get_new_evenness()
    
    def overlaps_with_feature(self, site_pos, feat):
        '''
        Given a genome feature and a TF binding site position, it returns True
        if there is overlap, False otherwise.
        '''
        
        site_start = site_pos
        site_end = site_start + self.hits['motif_length']
        
        feat_start = int(feat.location.start)
        feat_end = int(feat.location.end)
        
        if site_start < feat_end and feat_start < site_end:
            return True
        else:
            return False
    
    def gene_to_site_distance(self, feat, site_pos, circular_genome=False):
        '''
        
        Returns
        -------
        result : DICT
            The function returns a dictionary with two keys: 'distance' and 'location'.
            The 'distance' key provides an integer. Its absolute value is the distance
            between the gene start position and the closest edge of the TFBS. It's
            going to be 0 if the gene start is contained into the TFBS; it's going
            to be negative for TFBS that are upstream of the gene start; it's going
            to be positive for TFBS that are downstream of the gene start.
            The 'location' key is redundant, and it provides a string indicating
            whether the TFBS is located upstream or downstream of the gene start.
            
            
            EXAMPLE 1:
                A 23 bp TFBS located at position 1000, will be reported to be +3 bp
                from a gene start at position 997, -5 bp from a gene start at
                position 1028, and 0 bp from a gene start at position 1016 (because
                the TFBS would contain the gene start).
            
            EXAMPLE 2:
                In a circular genome of 1,000,000 bp a TFBS located at position
                1000 would be reported to be at +1030 bp from a gene start located
                at position 999,970.
            
        '''
        
        # Define site center "position" (it can be non-integer)
        edge_to_center = (self.hits['motif_length'] - 1)/2
        site_center = site_pos + edge_to_center
        
        # Feature coordinates
        coord = np.array([int(feat.location.start), int(feat.location.end)])
        
        # Three pairs of coordinates for circular genomes. One pair otherwise.
        coordinates = [coord]
        if circular_genome == True:
            coordinates.append(coord + self.length)
            coordinates.append(coord - self.length)
        
        # In this list, a single distance will be appended for non circular
        # genomes. For circular genomes three distances will be recorded (for
        # the three coordinate systems)
        tmp_distances = []
        
        for coord in coordinates:
            # Identify gene start position and compute distance from site_center
            
            # If gene is on forward strand
            if feat.location.strand in [1, '1', '+']:
                gene_start = coord[0]
                tmp_distances.append(site_center - gene_start)
            
            # If gene is on reverse strand
            elif feat.location.strand in [-1, '-1', '-']:
                gene_start = coord[1]
                tmp_distances.append(gene_start - site_center)
            
            else:
                raise ValueError("Unknown 'location.strand' value: " +
                                 str(feat.location.strand))
        
        # Choose the distance with the lowest absolute value.
        tmp_absolute_distances = [abs(x) for x in tmp_distances]
        gene_to_site_center = tmp_distances[np.argmin(tmp_absolute_distances)]
        
        # Define distance
        if abs(gene_to_site_center) < edge_to_center:
            # Overlapping
            distance = 0
        else:
            # Reduce the absoulte value of the distance by  edge_to_center
            if gene_to_site_center > 0:
                # Downstream
                distance = round(gene_to_site_center - edge_to_center)
            else:
                # Upstream
                distance = round(gene_to_site_center + edge_to_center)
        
        return distance
    
    def get_genetic_context(self, site_pos):
        
        intergenic = True
        
        genes = []
        distances = []
        for feat in self.features:
            # Ignore the feature if it's not 'CDS' nor 'tRNA' nor 'rRNA'
            if feat.type not in ['CDS', 'tRNA', 'rRNA']:  # !!! other feat types ...
                continue
            
            # Distance (from gene start to site)
            distance = self.gene_to_site_distance(
                feat, site_pos, circular_genome=True)  # !!! circular genome?
            
            # As soon as one gene is found to overlap with the hit, the
            # variable  intergenic  is set to False.
            if self.overlaps_with_feature(site_pos, feat):
                intergenic = False
            
            genes.append(feat)
            distances.append(distance)
        
        if len(genes) > 0:
            # Get closest gene record
            abs_distances = [abs(x) for x in distances]
            j = np.argmin(abs_distances)  # j-th element was the closest gene
            closest_gene = genes[j]  # j-th gene was the closest gene
        else:
            intergenic = 'no_genes'
            closest_gene = 'no_genes'
        
        return intergenic, closest_gene
    
    def set_hits_genetic_context(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        intergenic_list = []
        closest_gene_list = []
        for hit_pos in self.hits['positions']:
            intergenic, closest_gene = self.get_genetic_context(hit_pos)
            intergenic_list.append(intergenic)
            closest_gene_list.append(closest_gene)
        self.hits['intergenic'] = intergenic_list
        self.hits['closest_genes'] = closest_gene_list
    
    def analyze_intergenicity(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        # Identify genetic context for each hit
        self.set_hits_genetic_context()
        
        if self.n_sites == 0:
            return 'no_hits'
        
        if self.hits['intergenic'][0] == 'no_genes':
            return 'no_genes'
        
        # Count number of intergenic sites
        n_intergenic = sum(self.hits['intergenic'])
        # Intergenic frequency
        intergenic_freq = n_intergenic / self.n_sites
        # Set intergenicity attribute
        self.intergenicity = intergenic_freq








