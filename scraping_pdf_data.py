# Databricks notebook source
import pandas as pd

# COMMAND ----------

#Adding the langchaing 
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# COMMAND ----------

# import OpenAI and pandas 

from langchain.llms import OpenAI
import pandas as pd


# COMMAND ----------

#Installed tabula for transforming pdfs in csv. 
#!pip install 'PyPDF2<3.0'

!pip install --upgrade PyPDF2==2.12.1


# COMMAND ----------



import PyPDF2
f = open('/Workspace/Repos/gb_nuro_case_study/genai_case_study_one/data_cv/2028464_CV_PaoloCristini_DataScientist.pdf', "rb")
pdf_readr = PyPDF2.PdfReader(f)

numpages =  pdf_readr.numPages



# COMMAND ----------

check_csv = pd.read_csv('/Workspace/Repos/gb_nuro_case_study/genai_case_study_one/data_cv/2028464_CV_PaoloCristini_DataScientist.csv', delimiter='\t')

# COMMAND ----------

import csv
with open('/Workspace/Repos/gb_nuro_case_study/genai_case_study_one/data_cv/2028464_CV_PaoloCristini_DataScientist.csv', 'r') as f:

reader = csv.reader(f)
