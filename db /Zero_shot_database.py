# Databricks notebook source
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms.openai import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import pyodbc
from sqlalchemy import create_engine

import os
#import dotenv
#from dotenv import load_dotenv


server = 'LAPTOP-TMG2FQQS\MSSQLSERVER01'
database = 'omopcdm_synthea'
driver = '{ODBC Driver 17 for SQL Server}'

#conn_str = f'Driver={driver};Server={server};Database={database};integratedSecurity=true;trusted_Connection=yes'
# db = pyodbc.connect(conn_str)

odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+ driver + \
                ';Server=' + server + \
                ';DATABASE=' + database + \
                ';integratedSecurity=true;trusted_Connection=yes;'

db_engine = create_engine(odbc_str )
db = SQLDatabase(db_engine)

llm = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"], model_name='gpt-3.5-turbo')
#db_chain = SQLDatabaseSequentialChain(llm=llm, database=connection , verbose=True, top_k=3)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
          You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about Categories, Products and Orders.
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)

# Now initialize the create_sql_agent which is designed to interact with SQL Database as below. The agent is equipped with
# toolkit to connect to your SQL database and read both the metadata and content of the tables.
sql_toolkit.get_tools()
sqldb_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


sqldb_agent.run(final_prompt.format(
        #question="count of records in table medications ?"
        question="What is the most used medications description?"
  ))
# cursor = connection.cursor()
#
# cursor.execute("SELECT TOP 10 * from omopcdm_synthea.dbo.allergies")
# print(cursor.fetchall())
#
# cursor.close()
# connection.close()
####################


#toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=sql_toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
