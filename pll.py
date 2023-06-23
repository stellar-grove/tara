import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys;sys.path.append("./")
import sqlalchemy
import pyodbc as db 
import os
import re

# ------------------------------------------------------------------------------------------------------------------ #
# Variables
# ------------------------------------------------------------------------------------------------------------------ #
season_list = [2021,2022,2023]

data_location = "C:/stellar-grove/tara/data/pll/"
computerName = os.environ['COMPUTERNAME']
peepName = os.environ['USERNAME']
server = f'{computerName}\SQLEXPRESS'
DB = {'servername': server,
        'database': 'tara',
        'driver': 'driver=SQL Server Native Client 11.0'
        ,'tgtSchema':'pll'
        ,'tgtTbl':'player_stats'
}

tgtSchema = DB['tgtSchema']
tgtTbl = DB['tgtTbl']

dict_gender = {1:"female",2:"male",3:"other"}


#sqlcon = create_engine('mssql://' + servername + '/' + dbname + '?trusted_connection=yes')
cnxn = db.connect('DRIVER={SQL Server};SERVER='+DB['servername']+';DATABASE='+DB['database'])
engine = sqlalchemy.create_engine('mssql+pyodbc://' + DB['servername'] + '/' + DB['database'] + "?" + DB['driver'],echo=False)

# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# Helper Functions
# ------------------------------------------------------------------------------------------------------------------ #

def get_database(db_name:str):
    file_location = f'{data_location}{db_name}.csv'
    df_data = pd.read_csv(file_location)
    return df_data



class Data(object):

    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','Credit Score Class Initiated']]} 
        self.data = {}
        self.config = {}
        self.model = {}

    def removeCharacters(text):
        text = re.sub(r'\W+',"",text)
        return text

    def CompareDataFrames(dfSource: pd.DataFrame, dfTarget: pd.DataFrame, lstIdCompare: list):
        sourceId = lstIdCompare[0]
        targetId = lstIdCompare[1]
        dfCompare = dfSource[~dfSource[sourceId].isin(dfTarget[targetId])]
        return dfCompare

    def CheckDataType(DataInput):
        rtn = type(DataInput)
        return rtn

    def process_player_stats(self):
        df = pd.DataFrame()
        for season in season_list:
            file_location = f'{data_location}pll-player-stats-{season}.csv'
            pll_playerStats = pd.read_csv(file_location)
            pll_playerStats["season"] = season
            df = pd.concat([pll_playerStats,df],ignore_index=True)
        last_col = df.iloc[:, -1]
        df = pd.concat([last_col, df.iloc[:, :-1]], axis=1)
        return df

    def update_schedule(self):
        qry = f"""
        select * from pll.schedule
        """
        schedule = pd.read_sql()

    def load_to_database(self, data_frame:pd.DataFrame(), table:str, schema:str='pll', chunksize:int=250, csv=False):
        # First divert if using csv so that you don't need to go through the hassle of connecting to the database, whichever
        # one you are using.
        if csv:
            schema = DB["tgtSchema"]
            file_name = f"{schema}_{table}.csv"
            file_location = DB["fileRoot"]
            file_location = f"{file_location}{file_name}"
            data_frame.to_csv(file_location)
            records = data_frame.shape[0]

        else:
            data_frame.to_sql(table,
                              con=engine,                              
                              schema=schema,
                              if_exists='append',
                              index=False,
                              chunksize=chunksize)
            records = data_frame.shape[0]
        return records
    
    def load_file(self,file_name:str,file_extension:str=".csv",sheet_name:str=None):
        file_location = f"{data_location}{file_name}{file_extension}"
        if file_extension in [".csv"]: data_frame = pd.read_csv(file_location)
        if file_extension in [".txt"]: data_frame = pd.read_table(file_location)
        if file_extension in ["excel"]: data_frame = pd.read_excel(file_extension,sheet_name=sheet_name)
        return data_frame
    
    


    

    #--------------------------------------------------------
    # Data Cleaner
    # -------------------------------------------------------
    
    # -- Point of this section is to group together a series of functions that takes a file and runs several checks on it
    # -- to attempt to find commonly encountered things. Technically this could probably be a seperate class, but I figured
    # -- it would be easier to follow by just grouping the functions together and running a "run_" function, which is just
    # -- all the other functions that are cleaning the data stacked together to run in a pipeline of sorts.

    def run_data_cleaner(self,file_name:str,file_extension:str=".csv",sheet_name:str=None):
        data_frame = self.load_file(file_name=file_name,file_extension=file_extension,sheet_name=sheet_name)
        self.data["initial_shape"] = data_frame.shape
        unnamed_count = self.check_first_rows_for_blanks(data_frame)
        self.data["unnamed_count"] = unnamed_count
        return self.data
        


        
    def check_first_rows_for_blanks(self,data_frame:pd.DataFrame):
        columns = data_frame.columns
        unnamed_columns = []
        for column in columns:
            if "unnamed" in column:
                unnamed_columns.append(column)
        unnamed_count = len(unnamed_columns)
        return unnamed_count
        
    def check_numbers_as_string(self,data_frame:pd.DataFrame):
        columns = data_frame.columns
        for column in columns:
            idx = data_frame[column].str.isnumeric()
            rows = idx.shape[0]
            print(rows)
    





    def clear_log(self, dictionary_name:str):
        self.log[dictionary_name].clear()
    
class simulator(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','Simulator Initiated']]} 
        self.data = {}
        self.config = {}
        self.customer = {}
    
    def generate_age(self):
        age = max(min(np.random.normal(40,20), 100),0)
        self.customer["age"] = age
        
    def generate_gender(self):
        gender_value = np.random.randint(1,4)
        gender = dict_gender[gender_value]
        self.customer["gender_value"] = gender_value
        self.customer["gender"] = gender
        

    def generate_geographic_region(self):
        geography = np.random.randint(1,9)
        self.customer["geography"] = geography
    
    def generate_bought_merchandise(self):
        bought_merch = np.random.rand()
        if bought_merch < 0.65:
            bought_merch = 1
        else:
            bought_merch = 0
        self.customer["bought_merch"] = bought_merch
        
    def generate_amount_of_merchandise(self):
        if self.customer["bought_merch"] == 1:
            merch_count = np.random.randint(1,100)
        else:
            merch_count = 0
        self.customer["merch_count"] = merch_count

    def generate_customer_name(self):
        gender = self.customer["gender_value"]
        if gender == 1:
            name = "sally samsonite"
        if gender == 2:
            name = "keanu reeves"
        if gender == 3:
            name = "Jone Doe"
        self.customer["name"] = name

    def load_mappers(self):
        file_location = f"{data_location}unstructured.xlsx"
        mapper = pd.read_excel(file_location,sheet_name="mappers",skiprows=1)
        mapper = mapper.to_dict(orient="index")
        self.data["spend_mapper"] = mapper
        

    def generate_merch_spend(self):
        if self.customer["merch_count"] == 0:
            self.customer["total_spend"] = 0
            self.customer["average_item_spend"] = 0
        else:
            self.load_mappers()
            merch_count = self.customer["merch_count"]
            geography = self.customer["geography"]-1 # Account for 0 index
            spending_mapper = self.data["spend_mapper"][geography]
            total_merch_spend = (spending_mapper["AvgTotalSpend"]) * (1+np.random.random())
            average_item_price = total_merch_spend / merch_count
            self.customer["total_spend"] = total_merch_spend
            self.customer["average_item_spend"] = average_item_price

    def generate_games_attended(self):
        if self.customer["merch_count"] == 0:
            games_attended = np.random.randint(0,2)
            self.customer["games_attended"] = games_attended
        if self.customer["merch_count"] > 0 and self.customer["merch_count"]<5:
            games_attended = np.random.randint(0,4)
            self.customer["games_attended"] = games_attended
        if self.customer["merch_count"] > 5:
            games_attended = np.random.randint(1,10)
            self.customer["games_attended"] = games_attended


    def create_customer(self):
        self.generate_age()
        self.generate_gender()
        self.generate_geographic_region()
        self.generate_bought_merchandise()
        self.generate_amount_of_merchandise()
        self.generate_merch_spend()
        self.generate_customer_name()
        self.generate_games_attended()
        return self.customer

    def create_customer_base(self,base_size:int):
        customers = pd.DataFrame()
        for i in range(1,base_size+1):
            customer = self.create_customer()
            customer = pd.DataFrame().from_dict(customer,orient="index").T
            customer["customer_id"] = i
            customers = pd.concat([customers,customer])
        return customers
        