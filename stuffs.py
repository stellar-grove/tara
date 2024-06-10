import os


# ------ Begin Folders ------ #
DATA_FOLDER = "C:/stellar-grove/tara/data/"
DATA_FOLDER_SONGROVE = "C:/stellar-grove/tara/SongroveBotanicals/data/"

# ------ Begin Constants ----- #
peep = os.environ["USERNAME"]
robot = os.environ["COMPUTERNAME"]
homeD = f'{os.environ["HOMEDRIVE"]}{os.environ["HOMEPATH"]}'.replace('\\','/')

SGWD = f'{homeD}/Stellar Grove/'
bitsWD = f'{SGWD}bitstaemr - Documents/'
taraWD = f'{SGWD}ticondagrova - Documents/'

kaggleWD = f'{taraWD}Kaggle/'
udemyWD = f'{bitsWD}/Development/Udemy/'
server = f'{robot}\SQLEXPRESS'
sniffnet = 'sniffnet.database.windows.net'
DB_tara = {'servername': server,
        'database': 'tara',
        'driver': 'driver=SQL Server Native Client 11.0'
        ,'tgtSchema':'pll'
        ,'tgtTbl':'player_stats'
        ,'fileRoot':'C:/stellar-grove/tara/data/pll/'}

dbAzureTARA = {'servername': sniffnet,
            'database': 'tara',
            'driver': 'driver=SQL Server Native Client 11.0'
                }

chemistry_constants = {

    "avagadro_constant":602214076000000000000000,
    "gas_constant":8.31446261815324,
    "boltzman_constant":1.389649e-23


}

spyder_text = '''

repo = "C:/stellar-grove/"
import sys; sys.path.append(repo)
from bitstaemr import tools

import tara.SongroveBotanicals.research as rsh
import pandas as pd
import re
import os

irr = rsh.Irrigation()
data = irr.loadData()
fert = rsh.Fertilizer()
data_fert = fert.loadData()
fert.data["FertilizerData"]
pd.DataFrame().from_dict(fert.data['FertilizerData'],'index')


import bitstaemr.utils as utils
import tara.distributions as d

tools = utils.tools()

text = "the function f(x)x2 is a thing"
text = re.sub(r'\W+',"",text)


llaves = os.getenv('StellarGrove')
llaves.split(';')


stones = tools.get_stones()
stones.keys()


point1 = (0,1)
point2 = (1,4)

tools.calculate_trendline_angle(point2, point1)

from bitstaemr import CREAM as cream

av = cream.AlphaVantage()

av.llave
av.ticker

tkr = ['PLTR']
av.set_ticker(tkr)

av.get_data(tkr,"BALANCE_SHEET")
av.transform_data(av.data['quarterlyReports'])


'''