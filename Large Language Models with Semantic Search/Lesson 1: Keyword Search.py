# Databricks notebook source
""Setup
Load needed API keys and relevant Python libaries."""

# COMMAND ----------

# !pip install cohere
# !pip install weaviate-client

# COMMAND ----------

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# COMMAND ----------

"""Let's start by imporing Weaviate to access the Wikipedia database."""

# COMMAND ----------

import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])



# COMMAND ----------

client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)

# COMMAND ----------

client.is_ready() 

# COMMAND ----------

"""Keyword Search"""

# COMMAND ----------

def keyword_search(query,
                   results_lang='en',
                   properties = ["title","url","text"],
                   num_results=3):

    where_filter = {
    "path": ["lang"],
    "operator": "Equal",
    "valueString": results_lang
    }
    
    response = (
        client.query.get("Articles", properties)
        .with_bm25(
            query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
        )

    result = response['data']['Get']['Articles']
    return result

# COMMAND ----------

query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query)
print(keyword_search_results)

# COMMAND ----------

"""Try modifying the search options
Other languages to try: en, de, fr, es, it, ja, ar, zh, ko, hi"""


properties = ["text", "title", "url", 
             "views", "lang"]

# COMMAND ----------

print_result(keyword_search_results)

# COMMAND ----------

query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query, results_lang='de')
print_result(keyword_search_results)

# COMMAND ----------

"""

Welcome to Lesson 1. 
In this lesson, you will learn how to use keyword search 
to answer questions using a database. 
Search is crucial to how we navigate the world. 
This includes search engines, but it also includes search within 
apps, so when you search Spotify or YouTube or 
Google Maps for example. 
Companies and organizations also need to use keyword search 
or various methods of search to search their internal documents. 
Keyword search is the method most commonly used 
to build search systems. 
Let's look at how we can use a keyword search system and then 
look at how language models can improve those 
systems. 
Now, in this code example, we'll connect to a database and do 
some keyword search on it. 
The first cell installs "weaviate=client". 
You don't need to run this if you're running this from inside the classroom. 
 
But if you want to download this code and run 
it on your own environment, client, you would want to install "weaviate=client". 
 
The first code cell we need to run loads the API keys we'll 
need later in the lesson. 
And then now we can import Weaviate. 
This will allow us to connect to an online database. 
We'll talk about this database. 
Let's talk about what Weaviate is. 
Weaviate is an open source database. 
It has keyword search capabilities, but also vector 
search capabilities that rely on language models. 
The API key we'll be using here is public, this 
is part of a public demo, so the actual key is not 
a secret and you can use it and we encourage 
you to use it to access this demo database. 
Now that we've set the configurations for authentication, let's 
look at this code that connects the client to 
the actual database. 
Now this database is a public database and 
it contains 10 million records. 
These are records from Wikipedia. 
Each cell, each record, a row in this database, 
is a passage, is a paragraph from Wikipedia. 
These 10 million records are from 10 different languages. 
So one million of them are in English 
and the other nine million are in different languages. 
And we can choose and filter which language 
we want to query, and we'll see that later in this lab. 
After we run this line of code, we make sure that the 
client is ready and connected. 
And if we get true, then that means that 
our local Weaviate client is able to connect 
to the remote Weaviate database. 
And then now we're able to do a keyword search on this data set. 
Let's talk a little bit about keyword search before 
we look at the code. 
So let's say you have the query, what color is the grass? 
And you're searching a very small archive that has these five texts, these 
five sentences. 
One says tomorrow is Saturday, one says the grass is green, 
the capital of Canada is Ottawa, the sky is blue, 
and a whale is a mammal. 
So this is a simple example of search. 
A high-level look at how keyword search works is to 
compare how many words are in common between 
the query and the documents. 
So if we compare how many words are in common between the 
query and the first sentence, they only share the word is. 
And so that's one word they have in common. 
And we can see the counts of every sentence in this archive. 
We can see that the second sentence has the most 
words in common with the query, and so keyword search might 
retrieve that as the answer. 
So now that we're connected to the database, let's build 
out the function that will query it. 
Let's call it "keyword_search" and we'll be building this and going back 
and forth. 
So the simplest thing we'll need to do here is to say "response 
= (" and then "client.query.get". 
Now everything we're doing here, this is Weaviate. 
So this is all defined by the Weaviate API. 
And it tells us the kind of data, I think the collection 
we need to add here is called articles. 
So that's defined in that database. 
And since we want to do keyword search, let's 
say before we go into keyword search, let's copy 
what these properties are like. 
So these will be, let's say, a list defined with this data set, like 
this. 
Every article in this database has a number of properties. 
What we're saying here is that the results for this search, we 
want you to return to us the title, the URL, and 
the text for each result. 
There are other properties in here, but we don't want the database to return 
them to us now. 
Now to do the keyword search part, Weaviate has us type ".with_bm25", and 
bm25 is this keyword search or lexical search 
algorithm commonly used, 
and it scores the documents in the archive 
versus the query based on a specific formula 
that looks at the count of the shared 
words between the query and each document and 
the way we need to do this is to say "query=query" we will pass to you 
and the query we need to add to this function so it 
is a parameter here as well. 
A couple more lines we need to pass to alleviate our ".with_where", 
so we can have a where clause here that is formatted 
in a specific way. 
So what we want to do here is limit this to only English results. 
And results slang will be something we also 
add to this definition. 
So let's say "en". 
By default, we'll filter by the English language and only look at 
the English language articles. 
So that's why we're adding it here as a default, but it's 
something we can change whenever we call the keyword search 
method. 
Now one more line we need to add is to say ".with_limit". 
So how many results do we want the search engine to retrieve to us? 
So, we say "num_results" and then we define that here as well, so 
"num_results". 
And let's set that by default to say 3. 
And with that, our query is complete and 
we just say do and then we close that request. 
And once we've completed this, we can now get the response 
and return the result. 
With this, that is our keyword search function. 
Now let's use this keyword search function and pass it one query. 
Say we say, what is the most viewed televised event? 
We pass the query to the function and then we print it. 
So let's see what happens when we run this. 
It goes and comes back and these are 
the search results that we have. 
It's a lot of text, we'll go through it, but we 
can see that it's a list of dictionaries. 
So let's define a function that prints it 
in maybe a better way. 
And this function can look like this "print_result". 
And with this, we can say, okay, now print it and let me 
see exactly what the results were. 
So the first result that we got back is this is the text. 
This is the passage or paragraph of the text. 
This is the title, and remember, we're trying to look for what 
is the most televised event. 
This does not really look like the correct result very much, 
but it contains a lot of the keywords. 
Now, we have another article here about the Super Bowl. 
This is a better result, so the Super Bowl could probably 
be a highly televised event. and then there's 
another result here that kind of mentions the 
World Cup but it's not exactly the World Cup result. 
 
With each of these you see the URL of that 
article we can click on it and it will lead 
us to a Wikipedia page. 
You can try to edit this query so you can 
see what else is in this data set but this is a high-level 
example of the query connecting to 
the database and then seeing the results. 
A few things you can try here as well is 
you need to look at the properties. 
This is the list of properties that this 
data set was built using and so these 
are all columns that are stored within the database. 
So you can say you're gonna look at how many 
views a Wikipedia page received. 
You can use that to filter or sort. 
This is an estimated figure but then this 
is the language column that we use to filter, 
and you can use other values for language. 
The codes for the other languages look like this. 
So we have English, German, French, Spanish, Italian, Japanese, Arabic, 
Chinese, Korean, and Hindi, I believe. 
So just input one of these and pass it to the keyword search, 
and it will give you results in that language. 
Let's see how we can query the database with 
a different language. 
So let's say we copy this code. 
Notice that now we're printing the result here. 
Let's specify the language to a different language here. 
I'm going to be using, let's say, German. 
And we did German here because some words 
might be shared and we can see here some results. 
So this result for the most televised event 
is for Busta Rhymes, the musician. 
But you can see why it brought this as a result, right? 
Because the word event is here. 
And then the name of the album mentioned here is event. 
So the text here and the query that we have shared, 
they don't have to share all of the keywords but 
at least some of them. 
BM25 only needs one word to be shared for it 
to score that as somewhat relevant. And the more words the 
query and the document share, the more it's 
repeated in the document, the higher the score is. 
But we can see in general, while these results are returned, 
this is maybe not the best, 
most relevant answer to this question or document that 
is most relevant to this query. 
We'll see how language models help with this. 
So at the end of the first lesson, let's 
look back at search at a high level. 
The major components are the query, the search system, 
the search system has access to a document archive 
that it processed beforehand, and then in response 
to the query the system gives us a list of results ordered 
by the most relevant to the query. 
If we look a little bit more closely, we can think of search 
systems as having multiple stages. 
The first stage is often a retrieval or a search stage, 
and there's another stage after it called re-ranking. 
 
Re-ranking is often needed because we want to involve or 
include additional signals rather than just 
text relevance. 
The first stage, the retrieval, commonly uses the BM25 algorithm to 
score the documents in the archive versus the query. 
The implementation of the first stage 
retrieval often contains this idea of an inverted index. 
Notice that this table is a little bit different than the table 
we showed you before of the documents. 
The inverted index is this kind of table that has 
kind of these two columns. 
One is the keyword, and then next to the 
keyword is the documents that this keyword is present in. 
This is done to optimize the speed of the search. 
When you enter a query into a search engine, 
you expect the results in a few milliseconds. 
This is how it's done. 
In practice, in addition to the document ID, 
the frequency of how many times this keyword appears is also added 
to this call. 
With this, you now have a good high-level overview of keyword search. 
Now, notice for this query, what color is the sky, 
when we look at the inverted index, the word color has the document 804, 
and the word sky also has the document 804. 
So 804 will be highly rated from the 
results that are retrieved in the first stage. 
From our understanding of keyword search, 
we can see some of the limitations. 
So, let's say we have this query, strong pain in the side of the head. 
If we search a document archive that has this other document that 
answers it exactly, but it uses different keywords, 
so it doesn't use them exactly, it says sharp temple headache, keyword 
search is not going to be able to 
retrieve this document. 
This is an area where language models can help, 
because they're not comparing keywords simply. 
They can look at the general meaning and they're 
able to retrieve a document like this for 
a query like this. 
Language models can improve both search stages and 
in the next lessons, we'll look at how to do that. 
We'll look at how language models can improve the retrieval or first stage 
using embeddings, which are the topic of the next lesson. 
And then we'll look at how re-ranking works and how it can improve 
the second stage. 
And at the end of this course, we'll look at how large language models 
can generate responses as informed by a search step 
that happens beforehand. 
So let's go to the next lesson and learn about embeddings. 
"""
