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
    

    #--------------------------------------------------------
    # Data Cleaner
    # -------------------------------------------------------
    
    # -- Point of this section is to group together a series of functions that takes a file and runs several checks on it
    # -- to attempt to find commonly encountered things. Technically this could probably be a seperate class, but I figured
    # -- it would be easier to follow by just grouping the functions together and running a "process_" function, which is just
    # -- all the other functions that are cleaning the data stacked together to run in a pipeline of sorts.

    def clear_log(self, dictionary_name:str):
        self.log[dictionary_name].clear()

    
    


    
class simulator(object):
    def __init__(self, log={"status":[]})->None:
        self.log = {"status":[['C0','Simulator Initiated']]} 
        self.data = {}
        self.config = {}
        self.customer = {}
    
    def create_age(self):
        self.customer["age"] = np.random.random_integers(0,100,1)

    def generate_geographic_region(self):
        geography = np.random(1,8,1)
        self.customer["geography"] = geography
    
    def generate_bought_merchandise(self):
        bought_merch = np.random.randint(0,1)
        self.customer["bought_merch"] = bought_merch
        
    def generate_amount_of_merchandise(self):
        if self.customer["bought_merch"] == 1:
            merch_count = np.random.random_integers(1,100)
        else:
            merch_count = 0
        self.customer["merch_count"] = merch_count

    def generate_customer_name(self, gender:int):
        if gender == 1:
            name = "sally samsonite"
        if gender == 2:
            name = "keanu reeves"
        if gender == 3:
            name = "Jone Doe"
        self.customer["name"] = name

    def load_mappers(self):
        file_location = f"{data_location}unstructured.xlxs"
        mapper = pd.read_excel(file_location,sheet_name="mapper")
        mapper = mapper.to_dict(orient="index")
        self.data["spend_mapper"] = mapper

    def generate_merch_spend(self,merch_count:int):
        total_merch_spend = (self.data["spend_mapper"]["AvgTotalSpend"]) * (1+np.random.random())
        average_item_price = total_merch_spend / merch_count
        self.customer["total_spend"] = total_merch_spend
        self.customer["average_item_spend"] = average_item_price
        

    
        