# Databricks notebook source
#pip install pdfquery pandas

# COMMAND ----------

import os
import PyPDF2
import pandas as pd

# Path to the folder containing the PDF files
folder_path = 




# COMMAND ----------

# List all the PDF files in the folder
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]



# COMMAND ----------

import os
import PyPDF2
import pandas as pd

# Path to the folder containing the PDF files
folder_path = '/Workspace/Repos/gb_nuro_case_study/genai_case_study_one/data_cv'

# List all the PDF files in the folder
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]


pdf_files


# COMMAND ----------


# Loop through each PDF file and extract the data
data = {'Name': [], 'Email': [], 'Profile Summary':[], 'Areas of expertise':[], 'Work Experience': []}

for pdf_file in pdf_files:
    # Load the PDF file into a PdfReader object
    pdf = PyPDF2.PdfReader(pdf_file)

    # Loop through each page of the PDF file and extract the text
    for page in pdf.pages:
        text = page.extract_text()
        # Extract name, email, and work experience from text and append to data dictionary
        if 'Name: ' in text:
            name = text.split('Name: ')[1].split('\n')[0]
            data['Name'].append(name)
        if 'Email: ' in text:
            email = text.split('Email: ')[1].split('\n')[0]
            data['Email'].append(email)
        if 'Work Experience:\n' in text:
            work_experience = text.split('Work Experience:\n')[1].strip()
            data['Work Experience'].append(work_experience)

# Create a pandas DataFrame from the data dictionary
df = pd.DataFrame(data)


# COMMAND ----------

df

# COMMAND ----------

# Loop through each PDF file and extract the data
data = {'Name': [], 'Email': [], 'Work Experience': []}

for pdf_file in pdf_files:
    # Load the PDF file into a PdfReader object
    pdf = PyPDF2.PdfReader(pdf_file)

    # Loop through each page of the PDF file and extract the text
    for page in pdf.pages:
        text = page.extract_text()
        # Extract name, email, and work experience from text and append to data dictionary
        name = text.split('\n')[0]
        data['Name'].append(name)
        if 'Email: ' in text:
            email = text.split('Email: ')[1].split('\n')[0]
            data['Email'].append(email)
        else:
            data['Email'].append(None)
        if 'Work Experience:\n' in text:
            work_experience = text.split('Work Experience:\n')[1].strip()
            data['Work Experience'].append(work_experience)
        else:
            data['Work Experience'].append(None)

# Create a pandas DataFrame from the data dictionary
df = pd.DataFrame(data)

df

