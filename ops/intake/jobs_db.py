import sqlite3
import pandas as pd

class JobsDB:
    """Class for parsing JOBS database files from NIS Elements. By default these are encrypted
    SQLite databases. To decrypt, open the desired job in NIS Elements. Run the “Jobs_DecryptDb” 
    function as a macro. This may not be listed as a saved macro in the macro panel. If not, 
    click “Add Command…” and search for this function.

    Useful tables:
    `jobdef` : all saved JOB definitions
    `jobrun` : all previous acquisitions
        `iJobdefKeyRef` references `iKey` in `jobdef` table
    `jobrunpropstat` : summary of saved properties for all jobruns
        `iJobrunKeyRef` references `iKey` in `jobrun` table
        `iPropdefKeyRef` references `propdef` table for property definitions/names
    `frame` : all acquired frames and acuisition times (`dCTime`)
        `iJobrunKeyRef` references ``iKey` in `jobrun` table
        Wells can be deduced from referencing `framewellinfo` and `wellinfo` tables
    `frameprop` : all saved properties for all acquired frames
        `iJobrunKeyRef` & `iFrameIndex` are matched to `frame` table
        `iPropdefKeyRef` references `propdef` table for property definitions/names
    """
    def __init__(self,path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.property_mapper = {row['iKey']:row['wszFeatureName'] for _,row in self.get_table('propdef').iterrows()}
        
    def get_tables_list(self):
        return sorted([table[0] 
                for table 
                in self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()])
    
    def get_tables_dataframe(self):
        return pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table';", self.conn)
        
    def get_index_list(self):
        return [index[0] 
                for index
                in self.cursor.execute("SELECT name FROM sqlite_master WHERE type='index';").fetchall()]
    
    def query_to_pandas(self,query):
        return pd.read_sql_query(query, self.conn)
    
    def get_table(self,table):
        return self.query_to_pandas(f"SELECT * FROM {table}")

    def get_parsed_table(self,table):
        df = self.get_table(table)
        if 'iPropdefKeyRef' in df.columns:
            df = df.rename(columns={'iPropdefKeyRef':'property'})
            df['property'] = df['property'].map(self.property_mapper)
        return df
    
    def get_table_sizes(self):
        return self.get_table('sqlite_sequence').rename(columns={'seq':'rows'})
        
    def close(self):
        self.cursor.close()
        self.conn.close()