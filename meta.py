import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
# -------------------------------------------------------------------------------------------------------
# Variables
# -------------------------------------------------------------------------------------------------------
# Use these variables to reference down in the functions.  The idea is 
# to put the hard coded stuff at the top of the page so that anything 
# that needs to be changed can be done up here, once and not need to go 
# through a bunch of lines of code to fix something.
# -------------------------------------------------------------------------------------------------------
# Use these variables to specify which directory to look at. This was just 
# for my own ease of use as I tested going from my local machine to CoLab.  
# Idea behind these functions are to say that if you're not using my 
# StarFighter533 machine then use the sample_data directory where 
# everything is uploaded.
# -------------------------------------------------------------------------------------------------------
google = '/content/sample_data/';laptop = 'C:/stellar-grove/tara/data/'
data_location = laptop if os.environ["COMPUTERNAME"] != "Starfighter33" else google

# -------------------------------------------------------------------------------------------------------
# Simple Variables
# -------------------------------------------------------------------------------------------------------
file_name = "Book1 - DMC"
blank_column_search = 'Unnamed'
date_extraction_column = 'story'

# -------------------------------------------------------------------------------------------------------
# Various Dictionarys and lists
# -------------------------------------------------------------------------------------------------------
excel_types = ['.xlsx','.xls']
csv_types = ['.csv','.txt']

# This is the entity rollup dictionary used when transforming the entity 
# name to something that is easy to read.  In practice this could be a 
# table in a database that could be updated as new outliers are found.
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


# Columns used when factoring string data.  
#   Factorization is the concept of assigning a numerical 
#   values to all the different unique values of a set.
cols_to_factorize = ["source_name","method","entity_rollup","sector"]

# Lists out the column names we use in the analysis.
# -------------------------------------------------------------------------------------------------------
#   Idea here is to do all the work on the initial dataframe, i.e. add transformed columns to the
#   end of the data frame.  Then, once you have all the data you need, and the source columns that 
#   transformed data came from in one data frame, pare that down to only the numeric values you 
#   are going to use to run all the analysis on.  If you ever need to retrieve the original value
#   for output purposes you can reference the original dataframe instead of the transformed one. 

analysis_data_cols = ["entity_rollup","year","records_lost","sector",
                      "method","interesting_story","displayed_records",
                      'data_sensitivity', "month",'is_web','is_healthcare',
                      'is_app', 'is_retail','is_gaming', 'is_transport',
                      'is_financial', 'is_tech','is_government', 'is_telecoms',
                      'is_legal','is_media','is_academic', 'is_energy', 'is_military'
                      ]

# List of sectors to determine whether or not the entity resided within one of the sectors.  In the sectors
# there were multiple lines that had multiple sectors. My assumption here is that this is OK - if it's not
# obviously we have some issues, but I would think it would larger than the scope of this project as it would
# mean that some how multiple lines for only the sector column would have been written incorrectly.  Assuming
# its OK this is a way to count the entity in multiple sectors' counts, while still maintaing the original 
# row count.
sector_list = ['web','healthcare','app','retail','gaming','transport',
               'financial','tech','government','telecoms','legal',
               'media','academic','energy','military']

# -------------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------------
# The functions below are broken out into different types of functions corresponding to the different 
# aspects of the analysis.
# The first type of function are helper functions, i.e. micro type functions that are typically pretty 
# generalized and do the "grunt" work for other functions.
# -------------------------------------------------------------------------------------------------------
## Helper Functions
# -------------------------------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------------------------------
# Extract Transform & Load (ETL) Functions
# -------------------------------------------------------------------------------------------------------
# These functions are the meat and potatoes for the data gathering process.  Through these functions we 
# are able to gather all the data from the locations where it saved, clean it up and then get it ready 
# to be used in the analysis functions written out in the next section.
# -------------------------------------------------------------------------------------------------------

# On the initial analysis of the data it was noticed that there was a data_sensitivity type of 7, which 
# in all likelilhood was a manual error entry (7, should have been 4).  Since there was only one data 
# point the data point was removed.  In practice however, a process could be set up so that any 
# outliers that are not in the specified paramaters (from the first row of the data which looked 
# to be a description row) to flag, it - store it and put it on a list for an analyst / AI'ish powered 
# web searching bot to find out why the error occurred and how to solve it.

def check_data_sensitivity(data_frame:pd.DataFrame):
    passed = True if data_frame["data_sensitivity"].max() <= 5 else False
    data_frame["sensitivity_error"] = data_frame["data_sensitivity"].apply(lambda row: True if row > 5 else False)
    data_frame.drop(data_frame[data_frame['sensitivity_error']].index, inplace=True)
    return data_frame

# Creating weights to find a data_sensivity weighted records count.  This is by no means the only weight 
# you can create, but was a good example to show you can look at things in a different way other 
# than just how it's given.  The idea behind this paricular weight is that the data sensitivities seemed to be 
# in somewhat of a ranked order with 1 being the least troublesome and 5 being the most troublesome.  The 
# weights were calculated by taking the sensitivity of the row and dividing it by the max possible value.  
# This says that for data sensitivity of 1 we are really only going to count it at 20% because its online 
# information that could be found otherwise.  However if it was ALL the data, it's max amount, or 1 because 
# it's really bad.  This again assumes that ALL is the worst.  If a company doesn't have personal information 
# about people in their databases than losing ALL of it is the same as losing online information.
def create_weights(data_frame:pd.DataFrame, column_name:str="data_sensitivity"):
    columns = ["records_lost",column_name]
    weight_total = data_frame[columns].groupby(by=column_name).sum()
    return weight_total

# This function simply finds all the unique elements of a column of a dataframe and then assigns a numerical 
# value starting at 1 and going to the total number of elements. 
def create_factors(data_frame:pd.DataFrame, column_name:str):
    uniques = list(data_frame[column_name].unique())
    dict_factor = {}
    for index, factor in enumerate(uniques):
        dict_factor[factor] = (index+1)
    data_frame[column_name] = data_frame[column_name].map(dict_factor)
    return data_frame
# This function just takes all the columns listed in the cols_to_factorize list and factorizes them.  It saves
# down on having the to run the create_factors function multiple times by being able to simply add and take 
# away column names in the list to add / take away columns to turn into factors.
def factorize_columns(data_frame:pd.DataFrame):
    for column in cols_to_factorize:
        data_frame[column] = pd.factorize(data_frame[column])[0]+1
    return data_frame 

# -------------------------------------------------------------------------------------------------------
# Processing Functions
# -------------------------------------------------------------------------------------------------------
# Processes all the data.  This function runs a series of steps to take the data from the file that was 
# given and turns it into the a usable Data Frame to be used in the analysis phase.
def process_data():
    data_frame = loadData(file_name)
    remove_blank_columns(data_frame)
    # remove data description column
    data_frame.drop(0,axis=0,inplace=True) 
    # Process Columns names to be all lower case with underscores for spaces.
    data_frame.columns = process_column_names(data_frame)
    data_frame = process_entity_rollup(data_frame)
    data_frame = process_date_columns(data_frame)
    data_frame = process_records_lost_outlier(data_frame)
    data_frame = process_entity_names_for_labels(data_frame)
    data_frame = process_data_sensitivity(data_frame)
    return data_frame

# Gets the data from the location storage and returns the data frame associated with it.
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

# Seeks out any columns that are blank (tipped off by being named Unnamed. This assumes headers in the 
# first row are the actual factor names.)
def remove_blank_columns(data_frame:pd.DataFrame):
    for column in data_frame.columns:
        if blank_column_search in column:
            data_frame.drop(column,axis=1,inplace=True)
    return data_frame

# Takes all the column names and turns them into lower cases and turns any spaces into underscores for 
# easy reference.
def process_column_names(data_frame:pd.DataFrame):
    new_columns = data_frame.columns.str.lower().str.replace(' ', '_')
    return new_columns

# Takes the entity roll up dictionary creates a new column in the data frame with "cleaner" entity names.
def process_entity_rollup(data_frame:pd.DataFrame):
    data_frame["entity_rollup"] = data_frame["entity"].map(entity_roll_up)
    data_frame['entity_rollup'] = data_frame['entity_rollup'].where(~data_frame['entity_rollup'].isna(), data_frame['entity'])
    if data_frame["entity_rollup"].loc[:].isna:
        data_frame["entity"].loc[:] = data_frame['entity'].loc[:]
    else:
        data_frame["entity"].loc[:] = data_frame["entity_rollup"].loc[:]
    return data_frame

# Adds columns needed for dates / time series analysis
def process_date_columns(data_frame:pd.DataFrame):
    data_frame["month"] = data_frame["story"].str.split(" ").apply(lambda x: x[0])
    data_frame["month"] = data_frame["month"].apply(lambda x: x[0:3])
    data_frame["date_words"] = data_frame["month"] + "-" + data_frame["year"].astype(str)
    data_frame["date"] = pd.to_datetime(data_frame["date_words"],format="%b-%Y")
    data_frame["month"] = pd.to_datetime(data_frame["month"],format="%b").dt.month
    return data_frame
    
# Helper function to go through all the rows of a data frame and return the indices of any rows that are 
# not the expected type.  Typeically used to find string values in numerical columns, but has other 
# applications as well.
def find_error_rows(data_frame:pd.DataFrame,column_name:None,expected_type=float):
    idx = locate_text_rows(data_frame,column_name="records_lost")
    errors = pd.DataFrame(data_frame["records_lost"][idx])
    errors["entity_rollup"] = data_frame["entity_rollup"][idx]
    errors["1st_source_link"] = data_frame["1st_source_link"][idx]
    return errors

# # Process entity names for good labeling
def process_entity_names_for_labels(data_frame:pd.DataFrame, column_name:str="entity_rollup"):
    data_frame["entity_split"] = data_frame[column_name].apply(lambda row: list(row.split(" ")))
    data_frame["entity_label"] = data_frame["entity_split"].apply(lambda row: create_abbreviation(row))
    data_frame.drop("entity_split",axis=1,inplace=True)
    return data_frame 

# Turns the interesting story column from Y / null to 1 for Y and 0 for null
def process_interesting_story(data_frame:pd.DataFrame):
    data_frame["interesting_story"] = data_frame["interesting_story"].apply(lambda row: 0 if pd.isna(row) else 1)
    return data_frame

# From the assumed data description row, this column displays a 1 if the records leaked were over 
# One Hundred Million.
def process_large_records(data_frame:pd.DataFrame):
    data_frame["displayed_records"] = data_frame["displayed_records"].apply(lambda row: 1 if row > 100000000 else 0)
    return data_frame

# Creates a column that tells whether or not the data leak type was of seemingly sensitive information 
# (not online info, etc.)
def process_data_sensitivity(data_frame:pd.DataFrame):
    data_frame["is_sensitive"] = data_frame["data_sensitivity"].apply(lambda row: 1 if row > 1 else 0)
    return data_frame
# Creates a column named is_active_penetration for those instances where it seemed like more an active 
# attempt as opposed to a passive attempt / accident.
def process_security_issue(data_frame:pd.DataFrame):
    data_frame["is_active_penetration"] =  data_frame["method"].apply(lambda row: 1 if row in ["hacked","Inside Job"] else 0)
    return data_frame

# Runs through all the different sectors and creates a binary variable each sector the incidents occurred in.
def process_is_sector(data_frame:pd.DataFrame):
    data_frame["sector_list"] = list(data_frame["sector"].apply(lambda row: row.split(",")))
    for sector in sector_list:
        data_frame[f'is_{sector}'] = data_frame["sector_list"].apply(lambda row: 1 if sector in row else 0)
    data_frame.drop("sector_list",axis=1,inplace=True)
    return data_frame

# Processes the string values found in the records_lost column. Some of these values were filled in by 
# going to the article / searching for the incident and then manually adding the records lost / data type.  
# A more efficient way would be to use some NLP and web driving to go to each of the articles and seek it out.  
# The results could then be written to a database where the data could be uploaded and confirmed by a human 
# until the process occurs successfully at a desired rate of success (95%, 99%, 99.99%, etc.)
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


# -------------------------------------------------------------------------------------------------------
# Dataset Functions
# -------------------------------------------------------------------------------------------------------
# Creates the datasets needed to graph all the different looks we wanted to show.  The datasets are 
# returned into a dictionary and can be referenced by their column name.
def generate_graph_data(data_frame:pd.DataFrame):
    data_dictionary = {}
    cols_year = ["records_lost","year"]
    rcds_year = data_frame[cols_year].groupby(by=cols_year[1]).sum()
    data_dictionary["year"] = rcds_year
    cols_entity = ["records_lost","entity_label"]
    rcds_entity = data_frame[cols_entity].groupby(by=cols_entity[1]).sum().sort_values(by="entity_label",ascending = False).head(25)
    data_dictionary["entity"] = rcds_entity
    cols_method = ["records_lost","method"]
    rcds_method = data_frame[cols_method].groupby(by=cols_method[1]).sum()
    data_dictionary["method"] = rcds_method    
    cols_source = ["records_lost","source_name"]
    rcds_source = data_frame[cols_source].groupby(by=cols_source[1]).sum()
    data_dictionary["source"] = rcds_source
    cols_sensitivity = ["records_lost","data_sensitivity"]
    rcds_sensitivity = data_frame[cols_sensitivity].groupby(by=cols_sensitivity[1]).sum()
    data_dictionary["sensativity"] = rcds_sensitivity
    return data_dictionary

# Goes through and gets all the data into a form where it can be used in the analysis.  The result is a 
# data frame that contains only numeric values.  From there the rest of the functions can simply call on 
# column names.
def create_analysis_data(data_frame:pd.DataFrame):
    df = data_frame.copy()
    log = {}
    df["year"] = pd.to_numeric(df["year"])
    df = process_interesting_story(df)
    df = process_large_records(df)
    df = process_data_sensitivity(df)
    df = process_is_sector(df)
    df = check_data_sensitivity(df)
    df = process_security_issue(df)

    df["data_sensitity"] = pd.to_numeric(df["data_sensitivity"])
    analysis_data = df[analysis_data_cols]
    analysis_data = create_weights(df,column_name="data_sensitivity")
    return analysis_data

# -------------------------------------------------------------------------------------------------------
# Analysis Functions
# -------------------------------------------------------------------------------------------------------
# Use these functions to run the different analyses functions
# -------------------------------------------------------------------------------------------------------
## This subsection is for analysis helper functions, technically these should
## probably be up further in the helper functions, but that might just be
## my OCD coming out.
# -------------------------------------------------------------------------------------------------------

def split_data(data_frame:pd.DataFrame, features_list:list):
    # Split data into training and validation sets
    # First determine the independent variables.
    X = data_frame[features_list[:-1]]
    y = data_frame[features_list[-1:]]
    # Next run the test / train split algorithm
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=33)
    # return the outputs of the train test split algorithm
    return X_train, X_val, y_train, y_val

def function_narrative(d_var:list, d_coef:list, d_intercept:list, decimals:int):
    function_list = []
    for i in range(len(d_var)):
        var_name = d_var[i]
        var_coef = str(round(d_coef[i],decimals))
        intercept = str(round(d_intercept[0],decimals))
        function_list.append(var_coef + " * " + var_name)
    function_string = " ".join(function_list)
    function_string = f"f(x) = {function_string} {intercept}"
    return function_string

# Build the confusion matrix needed for the narrative output
def confusion_matrix_output(y_train, predictions, labels:list) -> tuple:
    confusion = pd.DataFrame(metrics.confusion_matrix(y_train, 
                                                      predictions),
                                                      index=labels,
                                                      columns=labels)
    confusion_pct = confusion / sum(confusion.sum())
    return confusion, confusion_pct

# Build the narrative we like to output for the confusion matrix
def confusion_matrix_narrative(confusion_matrices:tuple):
    confusion_matrix = confusion_matrices[0]
    confusion_pct = confusion_matrices[1]

    total_right = confusion_matrix["No"]["No"] + confusion_matrix["Yes"]["Yes"]
    pct_right = confusion_pct["No"]["No"] + confusion_pct["Yes"]["Yes"]

    pred_right_but_wrong = confusion_matrix["Yes"]["No"]
    pct_right_but_wrong = confusion_pct["Yes"]["No"]
    pred_wrong_but_right = confusion_matrix["No"]["Yes"]
    pct_wrong_but_right = confusion_pct["No"]["Yes"]


    confusion_narrative = f"""Per the confusion matrix, the amount of items that were correctly predicted were {total_right} or {pct_right}.
    The amount of predictions were predicted to be Yes but were actually No was: {pred_right_but_wrong} or {pct_right_but_wrong}.
    The amount of predictions were predicted to be No but were actually Yes was: {pred_wrong_but_right} or {pct_wrong_but_right}"""

    return confusion_narrative

# -------------------------------------------------------------------------------------------------------
# Build Logit Models for anything we are trying to calculate 
# probability of something happening
# -------------------------------------------------------------------------------------------------------
def build_logit(split_data:tuple):
    # Set variables for the data sets
    y_train = np.ravel(split_data[2])
    y_val = np.ravel(split_data[3])
    X_train = split_data[0]
    X_val = split_data[1]

    dict_model = {}
    dict_data = {}
    # Train model
    model = LogisticRegression()
    fitted_model = model.fit(X_train, y_train)
    dict_model["model"] = fitted_model
    dict_model["d_var"] = list(X_train.columns)
    dict_model["d_coef"] = [c for c in model.coef_[0]]
    dict_model["intercept"] = model.intercept_
    dict_model["model_words"] = function_narrative(dict_model["d_var"],
                                                dict_model["d_coef"],
                                                dict_model["intercept"],
                                                4)
    dict_model["accuracy_score"] = model.score(X_val,y_val)
    predictions = model.predict(X_train)
    dict_data["predictions"] = pd.DataFrame(predictions,columns=["Prediction"])
    confusion_matricies = confusion_matrix_output(y_train,
                                                dict_data["predictions"],
                                                ["No","Yes"])
    dict_model["confusion_matrix"] = confusion_matricies[0]
    dict_model["confusion_pct"] = confusion_matricies[1]
    dict_model["confusion_narrative"] = confusion_matrix_narrative(confusion_matricies)
    return dict_model, dict_data



# -------------------------------------------------------------------------------------------------------
#                         Jupyter Functions
# -------------------------------------------------------------------------------------------------------
# Use these functions to run the functionality of the Jupyter Notebook
def print_out_logit_results(model_output):
    for key in model_output[0]:
        print(key,model_output[0][key])