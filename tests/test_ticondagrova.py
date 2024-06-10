import sys; sys.path.append("../")
import pytest
import tara.distributions as dists
import tara.tests.data.ticondagrova as tara_test
import numpy as np

def check_dictionary_keys(dictionary_one:dict, dictionary_two:dict):
    for key in dictionary_one.keys():
        assert key in dictionary_two.keys()
    for key in dictionary_two.keys():
        assert key in dictionary_one.keys()
def check_dictionary_values(dictionary_one:dict, dictionary_two:dict):
    for key in dictionary_one.keys():
        assert dictionary_one[key] == dictionary_two[key]

class TestDistributions:
    
    def test_TSP(self):
        l = 10
        m = 18
        h = 23
        n = 2
        TSP = dists.TwoSidedPower()
        dict_TSP = TSP.moments(l,m,h,n)
        dict_test = tara_test.TEST_TSP
        check_dictionary_keys(dict_TSP, dict_test)
        check_dictionary_values(dict_TSP, dict_test)
    
    
    def test_generateTSP(self):
        sample = 0.3305
        lst_params = [10, 18, 23,2]
        TSP = dists.TwoSidedPower()
        tsp_sample = TSP.generateTSP(lst_params,sample)
        tsp_expected = tara_test.TSP_SAMPLE_EXPECTED
        assert round(tsp_sample,4) == round(tsp_expected,4)
        


