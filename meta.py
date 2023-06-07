import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def createDictionaryFromList(list:list):
    dictionary = dict
    for index, value in enumerate(list):
        dictionary[value] = index
        
    return dictionary

def createFeatureDictionary(data_frame:pd.DataFrame, column_name:str):
    unique_items = list(data_frame[column_name].unique())
