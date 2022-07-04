# -*- coding: utf-8 -*-
"""



"""

import numpy as np
import pandas as pd
import json
import os
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
        self.ripleyl = None
        self.intergenicity = None
    
    def compute_motif_specific_vals(self, save_original_data=True, outdir=None):
        '''
        Each motif in the TF object (the original motif and the permuted motifs)
        is used to analyze the MGE object. The results for each motif are
        stored in the motif_specific_vals attribute.
        motif_specific_vals is a dictionary where the keys are the different
        metrics, and the values are lists with the results from all the motifs.
        
        
        !!! save_original_data:
        
            
        '''
        
        stats = ['avg_score', 'extremeness', 'entropy', 'norm_entropy', 'gini',
                 'norm_gini', 'evenness', 'new_evenness', 'ripleyl', 'intergenicity']
        
        self.init_motif_specific_vals(stats)
        all_motifs = [self.tf.original] + self.tf.permuted
        for m in all_motifs:
            self.mge.scan(m, self.tf.pseudocount, self.tf.patser_threshold)
            self.mge.analyze_scores()
            self.mge.analyze_positional_distribution()
            self.mge.analyze_intergenicity()
            self.compile_motif_specific_vals(stats)
            if save_original_data:
                filename_tag = m.name + '_' + self.mge.original.id
                self.mge.original.save_report(filename_tag, outdir)
                save_original_data = False
    
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
        self.set_pvalue('ripleyl', 'greater')
    
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
    
    def save_p_vals(self, outdir=None):
        '''
        !!! Docstring here
        '''
        
        # Save p-values to CSV file
        stats = ['avg_score', 'extremeness', 'entropy', 'norm_entropy', 'gini',
                  'norm_gini', 'evenness', 'new_evenness', 'intergenicity']
        first_pval = []
        second_pval = []
        for stat in stats:
            first_pval.append(self.motif_specific_vals[stat][0])
            second_pval.append(vars(self)[stat])

        res = pd.DataFrame({'stat_name': stats,
                            'pval': first_pval,
                            'corrected_pval': second_pval})
        filename = self.tf.original.name + "_" + self.mge.original.id + '.csv'
        if outdir != None:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            filename = outdir + "/" + filename
        res.to_csv(filename, index=False)
    
    def save_motif_specific_vals(self, outdir=None):
        '''
        !!! Docstring here
        '''
        
        # Save motif-specific results to JSON file
        filename = self.tf.original.name + "_" + self.mge.original.id + '_motif_specific_values.json'
        if outdir != None:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            filename = outdir + "/" + filename
        with open(filename, 'w') as f:
            json.dump(self.motif_specific_vals, f)
    
    
    
    
    
    











































