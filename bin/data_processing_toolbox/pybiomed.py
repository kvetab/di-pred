import pandas as pd
import numpy as np
from PyBioMed import Pyprotein


class PyBioMedExtractor:
    """
    
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def calculate_all_descriptors(self, sequence):
        protein = Pyprotein.PyProtein(sequence)
        desc = list(protein.GetALL().values())
        tripept = list(protein.GetTPComp().values())
        all_desc = desc[:420] + tripept + desc[8420:]
        return all_desc
    
    def descriptors_for_ab(self, seqs):
        desc_heavy = calculate_all_descriptors(seqs["heavy"])
        desc_light = calculate_all_descriptors(seqs["light"])
        all_desc = desc_heavy + desc_light
        return np.asarray(all_desc)
    
    def transform(self, seq_df, y=None):
        return seq_df[["heavy", "light"]].apply(descriptors_for_ab, axis=1, result_type="expand")