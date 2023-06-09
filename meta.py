import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Variables
#data_location = '/content/sample_data/'
data_location = 'C:/stellar-grove/tara/data/'
excel_types = ['.xlsx','.xls']
csv_types = ['.csv','.txt']
file_name = "Book1 - DMC"
blank_column_search = 'Unnamed'
date_extraction_column = 'story'
entity_roll_up = {
                    '"Apple"':'Apple',
                    '"Gmail"':'Google',
                    'Google+':'Google',
                    'Sony Online Entertainment':'Sony',
                    'Sony Pictures':'Sony',
                    'Sony PSN':'Sony',
                    'T-Mobile':'T-Mobile',
                    'T-Mobile, Deutsche Telecom':'T-Mobile'
                }



# Functions
# ----------------------------------------------------------
## Helper Functions
def createDictionaryFromList(list:list):
    dictionary = dict
    for index, value in enumerate(list):
        dictionary[value] = index
    return dictionary

def createFeatureDictionary(data_frame:pd.DataFrame, column_name:str):
    unique_items = list(data_frame[column_name].unique())
    return unique_items

def get_column_type(data_frame:pd.DataFrame,column_name='*'):
    column_types = data_frame.dtypes
    if column_name != "*":
        column_types = column_types[column_name]
    return column_types

def locate_text_rows(data_frame:pd.DataFrame, column_name=None):
    idx = pd.to_numeric(data_frame[column_name],errors="coerce").isna()
    return idx

def create_abbreviation(word_list:list):
    if len(word_list) == 1: return word_list[0]
    if len(word_list) > 1:
        abbr = ""
        for word in word_list:
            abbr += word[0]
    return abbr

# ----------------------------------------------------------
# Processing Functions

def process_data():
    data_frame = loadData(file_name)
    remove_blank_columns(data_frame)
    data_frame.drop(0,axis=0,inplace=True)
    # Process Columns names to be all lower case with 
    # underscores for spaces.
    data_frame.columns = process_column_names(data_frame)
    process_entity_rollup(data_frame)
    process_date_columns(data_frame)
    data_frame = process_records_lost_outlier(data_frame)
    data_frame = process_entity_names_for_labels(data_frame)
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
    for column in data_frame.columns:
        if blank_column_search in column:
            data_frame.drop(column,axis=1,inplace=True)
    return data_frame

def process_column_names(data_frame:pd.DataFrame):
    new_columns = data_frame.columns.str.lower().str.replace(' ', '_')
    return new_columns

def process_entity_rollup(data_frame:pd.DataFrame):
    data_frame["entity_rollup"] = data_frame["entity"].map(entity_roll_up)
    data_frame['entity_rollup'] = data_frame['entity_rollup'].where(~data_frame['entity_rollup'].isna(), data_frame['entity'])
    if data_frame["entity_rollup"].loc[:].isna:
        data_frame["entity"].loc[:] = data_frame['entity'].loc[:]
    else:
        data_frame["entity"].loc[:] = data_frame["entity_rollup"].loc[:]
    return data_frame

def process_date_columns(data_frame:pd.DataFrame):
    data_frame["month"] = data_frame["story"].str.split(" ").apply(lambda x: x[0])
    data_frame["month"] = data_frame["month"].apply(lambda x: x[0:3])
    data_frame["date_words"] = data_frame["month"] + "-" + data_frame["year"].astype(str)
    data_frame["date"] = pd.to_datetime(data_frame["date_words"],format="%b-%Y")
    data_frame["month"] = pd.to_datetime(data_frame["month"],format="%b").dt.month
    return data_frame
    

def find_error_rows(data_frame:pd.DataFrame,column_name:None,expected_type=float):
    idx = locate_text_rows(data_frame,column_name="records_lost")
    errors = pd.DataFrame(data_frame["records_lost"][idx])
    errors["entity_rollup"] = data_frame["entity_rollup"][idx]
    errors["1st_source_link"] = data_frame["1st_source_link"][idx]
    return errors



# ---------------------------------------------------------------------------------------------------------

# Process entity names for good labeling
def process_entity_names_for_labels(data_frame:pd.DataFrame, column_name:str="entity_rollup"):
    data_frame["entity_split"] = data_frame[column_name].apply(lambda row: list(row.split(" ")))
    data_frame["entity_label"] = data_frame["entity_split"].apply(lambda row: create_abbreviation(row))
    data_frame.drop("entity_split",axis=1,inplace=True)
    return data_frame 

# Data Functions
def process_records_lost_outlier(data_frame:pd.DataFrame):
    file_location = f'{data_location}error_corrections.csv'
    error_corrections = pd.read_csv(file_location)
    ec_mapper = error_corrections[["id","new_records"]].set_index("id").to_dict()
    ec_mapper = ec_mapper["new_records"]
    for row in ec_mapper:
        data_frame.loc[row,"records_lost"] = ec_mapper[row]
    data_frame["records_lost"] = pd.to_numeric(data_frame["records_lost"])
    idx = data_frame["records_lost"].notnull()
    data_frame = data_frame[idx]
    return data_frame