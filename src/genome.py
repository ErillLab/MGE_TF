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
        self.genomic_units = {'bounds': None, 'coding': None}
        self.pssm_scores = None
        self.hits = {'scores': None,
                     'positions': None,
                     'threshold': None,
                     'motif_length': None,
                     'intergenic': None,
                     'closest_genes': None}
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
        
        # Features considered as 'coding' => When a site falls within one of
        # those features it is considered 'intergenic'.
        self.coding_feat = ['CDS','rRNA','tRNA','ncRNA','preRNA','tmRNA','misc']
        
        # Set genomic units, used to generate psudogenomes and to define
        # intergenic sites
        self.set_genomic_units()
    
    def set_genomic_units(self):
        '''
        Sets the  genomic_units  attribute.
        genomic_units  has two keys:
            "bounds": list of the bounds that can be used to split the genome
                      into units. The first and last bounds are the start and
                      the end of the genome.
            "coding": array of booleans of length n, where n is the number of
                      units. The i-th element is True when the i-th unit is
                      a coding unit, False otherwise.
        '''
        
        # Define bounds of units
        units_bounds = [0, self.length]
        coding_regions = []
        for feat in self.features:
            start, end = int(feat.location.start), int(feat.location.end)
            units_bounds.append(start)
            units_bounds.append(end)
            if feat.type in self.coding_feat:
                coding_regions.append([start, end])
        units_bounds = list(set(units_bounds))
        units_bounds.sort()
        
        # Check what units are 'coding'
        n_units = len(units_bounds) - 1
        coding_units = np.array([False] * n_units)
        for cod_reg in coding_regions:
            start, end = cod_reg
            unit_idx_start = units_bounds.index(start)
            unit_idx_end = units_bounds.index(end)
            coding_units[unit_idx_start:unit_idx_end] = True
        
        self.genomic_units['bounds'] = units_bounds
        self.genomic_units['coding'] = coding_units
    
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
            self.hits['scores'] = hits_scores
            self.hits['positions'] = hits_positions
            self.hits['threshold'] = threshold
            self.hits['motif_length'] = pssm.length
            self.n_sites = len(hits_scores)
        
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
        gini = self.get_gini_coeff(counts)
        
        # Compute minimum possible Gini coefficient
        quotient = number_of_obs // nuber_of_bins
        remainder = number_of_obs % nuber_of_bins
        chunk_1 = np.repeat(quotient, nuber_of_bins - remainder)
        chunk_2 = np.repeat(quotient + 1, remainder)
        vect = np.hstack((chunk_1, chunk_2))  # values distr as evenly as possible
        min_gini = self.get_gini_coeff(vect)
        
        # Compute maximum possible Gini coefficient
        chunk_1 = np.repeat(0, nuber_of_bins - 1)
        chunk_2 = np.repeat(number_of_obs, 1)
        vect = np.hstack((chunk_1, chunk_2))  # values distr as unevenly as possible
        vect = [int(v) for v in vect]
        max_gini = self.get_gini_coeff(vect)
        
        # Compute normalized Gini coefficient
        if max_gini - min_gini == 0:
            norm_gini = 0
        else:
            norm_gini = (gini - min_gini) / (max_gini - min_gini)
        
        return norm_gini
    
    def get_hits_distances(self):
        '''
        Returns the distance (in bp) between consecutive hits on the genome.
        '''
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
        
        intervals = self.get_hits_distances()
        return np.var(intervals)

    def get_norm_evenness(self):
        '''
        Normalized evenness.
        Norm_Evenness = Evenness / Max_Evenness
        '''
        
        intervals = self.get_hits_distances()
        var = np.var(intervals)
        
        n_intervals = len(intervals)
        mean = self.length/n_intervals
        max_var = ((n_intervals - 1) * mean**2 + (self.length - mean)**2)/n_intervals
        norm_var = var / max_var
        return norm_var

    def get_new_evenness(self):
        '''
        A transformation is applied so that large evenness values imply a very
        even distribution (it's the opposite in the original definition of
        evenness by Philip and Freeland).
        '''
        
        norm_var = self.get_norm_evenness()
        new_evenness = 1 - norm_var
        return new_evenness
    
    def analyze_scores(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'analyze_positional_distribution'.")
        
        if self.n_sites == 0:
            self.avg_score = 'no_sites'
            self.extremeness = 0
        else:
            self.avg_score = self.hits['scores'].mean()
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
        
        # Site density (sites per thousand bp)
        self.site_density = 1000 * self.n_sites / self.length
        
        if self.n_sites < 3:  # !!! Make it a parameter (from config file?)
            self.entropy = 'not_enough_sites'
            self.norm_entropy = 'not_enough_sites'
            self.gini = 'not_enough_sites'
            self.norm_gini = 'not_enough_sites'
            self.evenness = 'not_enough_sites'
            self.new_evenness = 'not_enough_sites'
        
        else:
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
    
    def get_closest_gene(self, site_pos):
        
        genes = []
        distances = []
        for feat in self.features:
            # Ignore the feature if it's not 'coding'.
            if feat.type not in self.coding_feat:
                continue
            
            # Distance (from gene start to site)
            distance = self.gene_to_site_distance(feat, site_pos, circular_genome=True)  # !!! circular genome?
            genes.append(feat)
            distances.append(distance)
        
        if len(genes) > 0:
            # Get closest gene record
            abs_distances = [abs(x) for x in distances]
            j = np.argmin(abs_distances)  # j-th element was the closest gene
            return genes[j]  # j-th gene was the closest gene
        else:
            return 'no_genes'
    
    def set_hits_closest_genes(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        if sum(self.genomic_units['coding']) == 0:
            self.hits['closest_genes'] = 'no_genes'
        
        else:
            self.hits['closest_genes'] = []
            for hit_pos in self.hits['positions']:
                closest_gene = self.get_closest_gene(hit_pos)
                self.hits['closest_genes'].append(closest_gene)
    
    def is_intergenic(self, hit_pos):
        idx_right_bound = np.searchsorted(self.genomic_units['bounds'], hit_pos)
        idx_unit = idx_right_bound - 1
        return not self.genomic_units['coding'][idx_unit]
    
    def set_hits_intergenic(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        if sum(self.genomic_units['coding']) == 0:
            self.hits['intergenic'] = 'no_genes'
        
        else:
            self.hits['intergenic'] = []
            for hit_pos in self.hits['positions']:
                self.hits['intergenic'].append(self.is_intergenic(hit_pos))
    
    def analyze_intergenicity(self):
        
        if not self.hits:
            raise TypeError(
                "The 'hits' attribute is 'NoneType'. Make sure you call the\
                'scan' method specifying a threshold to get PSSM-hits before\
                calling 'get_intergenicity'.")
        
        # Identify genetic context for each hit
        self.set_hits_intergenic()
        
        if self.n_sites == 0:
            self.intergenicity = 'no_sites'
            
        elif self.hits['intergenic'] == 'no_genes':
            self.intergenicity = 'no_genes'
            
        else:
            # Count number of intergenic sites
            n_intergenic = sum(self.hits['intergenic'])
            # Intergenic frequency
            intergenic_freq = n_intergenic / self.n_sites
            # Set intergenicity attribute
            self.intergenicity = intergenic_freq
    
    
    








