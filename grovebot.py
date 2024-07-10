import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy import text


class CreepyCrawlers(object):

    def __init__(self)->None:
        self.data = {}
        self.stats = {}


class ChittyChat(object):

    def __init__(self)->None:
        self.data={}
        self.stats={}

    sql_prompt_params = f"""
    ### sqlite SQL tables, with their properties:
    #
    # Employee(id, name, department_id)
    # Department(id, name, address)
    # Salary_Payments(id, employee_id, amount, date)
    #
    ### A quey to list the names of the departments which employed more than 10 employees.
    # Select

    """
    def create_table_definition(df):
        prompt = """### sqlite SQL table, with its properties:
        #
        # Sales({})
        #
        """.format(",".join(str(col) for col in df.columns))
        return prompt
    

    def dataframe_to_database(df, table_name):
        """Convert a pandas dataframe to a database.
            Args:
                df (dataframe): pd.DataFrame which is to be converted to a database
                table_name (string): Name of the table within the database
            Returns:
                engine: SQLAlchemy engine object
        """
        engine = create_engine(f'sqlite:///:memory:', echo=False)
        df.to_sql(name=table_name, con=engine, index=False)
        return engine
    
    def handle_response(response):
        """Handles the response from OpenAI.

        Args:
            response (openAi response): Response json from OpenAI

        Returns:
            string: Proposed SQL query
        """
        query = response["choices"][0]["text"]
        if query.startswith(" "):
            query = "Select"+ query
        return query
    
    def execute_query(engine, query):
        """Execute a query on a database.

        Args:
            engine (SQLAlchemy engine object): database engine
            query (string): SQL query

        Returns:
            list: List of tuples containing the result of the query
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchall()
        
    