import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ProtparamExtractor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def transform(self, seq_df, y=None):
        return seq_df[["heavy", "light"]].apply(descriptors_for_ab, axis=1, result_type="expand")
    
    def get_all_params(self, sequence):
        analysed_seq = ProteinAnalysis(seq)
        aa_per = analysed_seq.get_amino_acids_percent().values()
        aromacity = analysed_seq.aromaticity()
        instability = analysed_seq.instability_index()
        flexibility = np.average(analysed_seq.flexibility())
        isoelectric = analysed_seq.isoelectric_point()
        mol_extinct1, mol_extinct2 = analysed_seq.molar_extinction_coefficient()
        mw = analysed_seq.molecular_weight()
        gravy = analysed_seq.gravy()
        ss_faction = analysed_seq.secondary_structure_fraction()
        feature = list(aa_per) + [aromacity, instability, flexibility, isoelectric, mol_extinct1, mol_extinct2, mw, gravy] + list(ss_faction)
        return feature
    
    def descriptors_for_ab(self, seq):
        desc_heavy = calculate_all_descriptors(seq["heavy"])
        desc_light = calculate_all_descriptors(seq["light"])
        all_desc = desc_heavy + desc_light
        return np.asarray(all_desc)
        


# from https://github.com/yemilyz/bioviaclinic1920/blob/aa69b4d98c1d98f810f45145ad09a5b4cae5e9a2/source/protparam_features.py
def get_properties_for_sequences(seqs):
    feature_set = {}
    colNames = ['aa_percent{}'.format(i) for i in range(20)] + ['aromacity', 'instability',
                'flexibility', 'isoelectric', 'mol_extinct1',
                'mol_extinct2', 'mw', 'gravy', 'ss_faction1', 'ss_faction2',
                'ss_faction3']
    for name, seq in seqs.items():
        analysed_seq = ProteinAnalysis(seq)
        aa_per = analysed_seq.get_amino_acids_percent().values()
        aromacity = analysed_seq.aromaticity()
        instability = analysed_seq.instability_index()
        flexibility = np.average(analysed_seq.flexibility())
        isoelectric = analysed_seq.isoelectric_point()
        mol_extinct1, mol_extinct2 = analysed_seq.molar_extinction_coefficient()
        mw = analysed_seq.molecular_weight()
        gravy = analysed_seq.gravy()
        ss_faction = analysed_seq.secondary_structure_fraction()
        feature = list(aa_per) + [aromacity, instability, flexibility, isoelectric, mol_extinct1, mol_extinct2, mw, gravy] + list(ss_faction)
        feature_set[name] = feature
    feature_set = pd.DataFrame.from_dict(feature_set, orient='index', columns=colNames)
    return feature_set