# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import warnings


class MGE_TF():
    
    def __init__(self, mge, tf):
        
        self.mge = mge
        self.tf = tf
        self.motif_specific_vals = None
        
        # p-values
        self.avg_score = None
        self.extremeness = None
        self.entropy = None
        self.norm_entropy = None
        self.gini = None
        self.norm_gini = None
        self.evenness = None
        self.new_evenness = None
        self.intergenicity = None
    
    def compute_motif_specific_vals(self):
        '''
        Each motif in the TF object (the original motif and the permuted motifs)
        is used to analyze the MGE object. The results for each motif are
        stored in the motif_specific_vals attribute.
        motif_specific_vals is a dictionary where the keys are the different
        metrics, and the values are lists with the results from all the motifs.
        '''
        
        stats = ['avg_score', 'extremeness', 'entropy', 'norm_entropy', 'gini',
                 'norm_gini', 'evenness', 'new_evenness', 'intergenicity']
        
        self.init_motif_specific_vals(stats)
        
        all_motifs = [self.tf.original] + self.tf.permuted
        for m in all_motifs:
            self.mge.scan(m, self.tf.pseudocount, self.tf.patser_threshold)
            self.mge.analyze_scores()
            self.mge.analyze_positional_distribution()
            self.mge.analyze_intergenicity()
            self.compile_motif_specific_vals(stats)
    
    def init_motif_specific_vals(self, stats):
        ''' Initialize motif_specific_vals attribute as a dictionary where the
        values are empty lists. '''
        self.motif_specific_vals = dict(zip(stats, [[] for x in range(len(stats))]))
    
    def compile_motif_specific_vals(self, stats):
        ''' Compile motif_specific_vals attribute. The current statistic values
        from the MGE object are appended to the corresponding values of the
        motif_specific_vals dictionary. '''
        for stat in stats:
            val = vars(self.mge)[stat]
            self.motif_specific_vals[stat].append(val)
    
    def analyze_scores(self):
        ''' Sets the p-value for the statistics related to the PSSM-scores.
        The alternative Hp is always 'smaller' because the values used for this
        test are already p-values. '''
        self.set_pvalue('avg_score', 'smaller')
        self.set_pvalue('extremeness', 'smaller')
    
    def analyze_positional_distribution(self):
        ''' Sets the p-value for the statistics related to the positional distribution.
        The alternative Hp is always 'smaller' because the values used for this
        test are already p-values. '''
        self.set_pvalue('entropy', 'smaller')
        self.set_pvalue('norm_entropy', 'smaller')
        self.set_pvalue('gini', 'smaller')
        self.set_pvalue('norm_gini', 'smaller')
        self.set_pvalue('evenness', 'smaller')
        self.set_pvalue('new_evenness', 'smaller')
    
    def analyze_intergenicity(self):
        ''' Sets the p-value for the statistics related to the intergenicity.
        The alternative Hp is always 'smaller' because the values used for this
        test are already p-values. '''
        self.set_pvalue('intergenicity', 'smaller')
    
    def set_pvalue(self, metric, alternative):
        '''
        Estimates the p-value for a given metric, and a given alternative
        hypothesis. The estimate is based on the frequency of pseudogenomes
        that can reproduce the results observed on the original genome.
        '''
        obs = self.motif_specific_vals[metric][0]
        control_values = self.motif_specific_vals[metric][1:]
        
        valid_values = [x for x in control_values if not isinstance(x, str)]
        if len(valid_values) < len(control_values):
            warnings.warn("Only {}/{} values of {} were valid and used to \
                          estimate the p-value.".format(len(valid_values),
                          len(control_values), metric))
        
        control = np.array(valid_values)
        
        if isinstance(obs, (str, type(None))):
            p_val = 'no_valid_obs'
        
        elif len(control) == 0:
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
    
    
    
    
    
    











































