# Databricks notebook source
# MAGIC %md
# MAGIC # OpenAI Function Calling In LangChain

# COMMAND ----------

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# COMMAND ----------

from typing import List
from pydantic import BaseModel, Field

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pydantic Syntax
# MAGIC
# MAGIC Pydantic data classes are a blend of Python's data classes with the validation power of Pydantic. 
# MAGIC
# MAGIC They offer a concise way to define data structures while ensuring that the data adheres to specified types and constraints.
# MAGIC
# MAGIC In standard python you would create a class like this:

# COMMAND ----------

class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

# COMMAND ----------

foo = User(name="Joe",age=32, email="joe@gmail.com")

# COMMAND ----------

foo.name

# COMMAND ----------

foo = User(name="Joe",age="bar", email="joe@gmail.com")

# COMMAND ----------

foo.age

# COMMAND ----------

foo = User(name="Joe",age="bar", email="joe@gmail.com")

# COMMAND ----------

foo.age

# COMMAND ----------

class pUser(BaseModel):
    name: str
    age: int
    email: str

# COMMAND ----------

foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")

# COMMAND ----------

foo_p.name

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: The next cell is expected to fail.

# COMMAND ----------



# COMMAND ----------

foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")

# COMMAND ----------

class Class(BaseModel):
    students: List[pUser]

# COMMAND ----------

obj = Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com")]
)

# COMMAND ----------

obj

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pydantic to OpenAI function definition

# COMMAND ----------

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

# COMMAND ----------

from langchain.utils.openai_functions import convert_pydantic_to_openai_function

# COMMAND ----------

weather_function = convert_pydantic_to_openai_function(WeatherSearch)

# COMMAND ----------

weather_function

# COMMAND ----------

class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: The next cell is expected to generate an error.

# COMMAND ----------

convert_pydantic_to_openai_function(WeatherSearch1)

# COMMAND ----------

class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str

# COMMAND ----------

convert_pydantic_to_openai_function(WeatherSearch2)

# COMMAND ----------

from langchain.chat_models import ChatOpenAI

# COMMAND ----------

model = ChatOpenAI()

# COMMAND ----------

model.invoke("what is the weather in SF today?", functions=[weather_function])

# COMMAND ----------

model_with_function = model.bind(functions=[weather_function])

# COMMAND ----------

model_with_function = model.bind(functions=[weather_function])

# COMMAND ----------

model_with_function = model.bind(functions=[weather_function])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forcing it to use a function
# MAGIC
# MAGIC We can force the model to use a function

# COMMAND ----------

model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})

# COMMAND ----------

model_with_forced_function.invoke("what is the weather in sf?")

# COMMAND ----------

model_with_forced_function.invoke("hi!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using in a chain
# MAGIC
# MAGIC We can use this model bound to function in a chain as we normally would

# COMMAND ----------

from langchain.prompts import ChatPromptTemplate

# COMMAND ----------


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])

# COMMAND ----------

chain = prompt | model_with_function

# COMMAND ----------

chain.invoke({"input": "what is the weather in sf?"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using multiple functions
# MAGIC
# MAGIC Even better, we can pass a set of function and let the LLM decide which to use based on the question context.

# COMMAND ----------

class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")

# COMMAND ----------

functions = [
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch),
]

# COMMAND ----------

model_with_functions = model.bind(functions=functions)

# COMMAND ----------

model_with_functions.invoke("what is the weather in sf?")

# COMMAND ----------

model_with_functions.invoke("what are three songs by taylor swift?")

# COMMAND ----------

model_with_functions.invoke("hi!")
