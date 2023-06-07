import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Variables
data_location = 'C:/stellar-grove/tara/data/'
excel_types = ['.xlsx','.xls']
csv_types = ['.csv','.txt']
file_name = "Book1 - DMC"

# Functions

def process_data():
    data_frame = loadData(file_name)
    remove_blank_columns(data_frame)
    data_frame.drop(0,axis=0,inplace=True)
    return data_frame

def loadData(file_name:str, file_extension:str='.xlsx'):
    file_location = f'{data_location}{file_name}{file_extension}'
    # Stuck to two common style of data storage. 
    # Could add more as more data types are encountered.
    if file_extension in (excel_types):
        data_frame = pd.read_excel(file_location)
    if file_extension in (csv_types):
        # Doesn't account for headers - can be processed later
        data_frame = pd.read_csv(file_location)
    return data_frame

def remove_blank_columns(data_frame:pd.DataFrame):
    sub_string = "Unnamed"
    for column in data_frame.columns:
        if sub_string in column:
            data_frame.drop(column,axis=1,inplace=True)
    return data_frame

def createDictionaryFromList(list:list):
    dictionary = dict
    for index, value in enumerate(list):
        dictionary[value] = index
    return dictionary

def createFeatureDictionary(data_frame:pd.DataFrame, column_name:str):
    unique_items = list(data_frame[column_name].unique())
