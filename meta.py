import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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
analysis_data_cols = ["entity_rollup","year","records_lost","sector",
                      "method","interesting_story","displayed_records",'data_sensitivity',
                      "month",'is_web','is_healthcare', 'is_app', 'is_retail',
                      'is_gaming', 'is_transport','is_financial', 'is_tech',
                      'is_government', 'is_telecoms', 'is_legal','is_media',
                      'is_academic', 'is_energy', 'is_military'
                      ]

sector_list = ['web','healthcare','app','retail','gaming','transport',
               'financial','tech','government','telecoms','legal',
               'media','academic','energy','military']

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

def check_data_sensitivity(data_frame:pd.DataFrame):
    passed = True if data_frame["data_sensitivity"].max() <= 5 else False
    data_frame["sensitivity_error"] = data_frame["data_sensitivity"].apply(lambda row: True if row > 5 else False)
    data_frame.drop(data_frame[data_frame['sensitivity_error']].index, inplace=True)

    return data_frame

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

def process_interesting_story(data_frame:pd.DataFrame):
    data_frame["interesting_story"] = data_frame["interesting_story"].apply(lambda row: 0 if pd.isna(row) else 1)
    return data_frame

def process_large_records(data_frame:pd.DataFrame):
    data_frame["displayed_records"] = data_frame["displayed_records"].apply(lambda row: 1 if row > 100000000 else 0)
    return data_frame

def process_data_sensitivity(data_frame:pd.DataFrame):
    data_frame["is_sensitive"] = data_frame["data_sensitivity"].apply(lambda row: 1 if row > 1 else 0)
    return data_frame

def process_security_issue(data_frame:pd.DataFrame):
    data_frame["is_active_penetration"] =  data_frame["method"].apply(lambda row: 1 if row in ["hacked","Inside Job"] else 0)
    return data_frame

def process_is_sector(data_frame:pd.DataFrame):
    data_frame["sector_list"] = list(data_frame["sector"].apply(lambda row: row.split(",")))
    for sector in sector_list:
        data_frame[f'is_{sector}'] = data_frame["sector_list"].apply(lambda row: 1 if sector in row else 0)
    data_frame.drop("sector_list",axis=1,inplace=True)
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
    cols_sensativity = ["records_lost","data_sensitivity"]
    rcds_sensativity = data_frame[cols_sensativity].groupby(by=cols_sensativity[1]).sum()
    data_dictionary["sensativity"] = rcds_sensativity
    return data_dictionary


def create_analysis_data(data_frame:pd.DataFrame):
    df = data_frame.copy()
    log = {}
    df["year"] = pd.to_numeric(df["year"])
    df = process_interesting_story(df)
    df = process_large_records(df)
    df = process_data_sensitivity(df)
    df = process_is_sector(df)
    df = check_data_sensitivity(df)
    df["data_sensitity"] = pd.to_numeric(df["data_sensitivity"])
    factorize_columns = ["source_name","method","entity_rollup","sector","data_sensitivity"]
    for column in factorize_columns:
        df[column] = pd.factorize(df[column])[0]
    analysis_data = df[analysis_data_cols]
    return analysis_data


# -----------------------------------------------------------------
# Analysis Functions

def split_data(data_frame:pd.DataFrame, features_list:list):
    # Split data into training and validation sets
    # First determine the independent variables.
    X = data_frame[features_list[:-1]]
    y = data_frame[features_list[-1:]]
    # Next run the test / train split algorithm
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=33)
    # return the outputs of the train test split algorithm
    return X_train, X_val, y_train, y_val

# The build_logit for
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

def confusion_matrix_output(y_train, predictions, labels:list) -> tuple:
    confusion = pd.DataFrame(metrics.confusion_matrix(y_train, 
                                                      predictions),
                                                      index=labels,
                                                      columns=labels)
    confusion_pct = confusion / sum(confusion.sum())
    return confusion, confusion_pct

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

