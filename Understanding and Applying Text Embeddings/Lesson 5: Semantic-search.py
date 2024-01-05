# Databricks notebook source
# MAGIC %md
# MAGIC ## Lesson 6: Semantic Search, Building a Q&A System

# COMMAND ----------

# MAGIC %md
# MAGIC #### Project environment setup
# MAGIC
# MAGIC - Load credentials and relevant Python Libraries

# COMMAND ----------

from utils import authenticate
credentials, PROJECT_ID = authenticate()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enter project details

# COMMAND ----------

REGION = 'us-central1'

# COMMAND ----------

import vertexai
vertexai.init(project=PROJECT_ID, location=REGION, credentials = credentials)

# COMMAND ----------

## Load Stack Overflow questions and answers from BigQuery

# COMMAND ----------

import pandas as pd

# COMMAND ----------

so_database = pd.read_csv('so_database_app.csv')

# COMMAND ----------

print("Shape: " + str(so_database.shape))
print(so_database)

# COMMAND ----------

## Load the question embeddings

# COMMAND ----------

from vertexai.language_models import TextEmbeddingModel

# COMMAND ----------

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

# COMMAND ----------

import numpy as np
from utils import encode_text_to_embedding_batched

# COMMAND ----------

# MAGIC %md
# MAGIC - Here is the code that embeds the text.  You can adapt it for use in your own projects.  
# MAGIC - To save on API calls, we've embedded the text already, so you can load it from the saved file in the next cell.
# MAGIC
# MAGIC ```Python
# MAGIC # Encode the stack overflow data
# MAGIC
# MAGIC so_questions = so_database.input_text.tolist()
# MAGIC question_embeddings = encode_text_to_embedding_batched(
# MAGIC             sentences = so_questions,
# MAGIC             api_calls_per_second = 20/60, 
# MAGIC             batch_size = 5)
# MAGIC ```

# COMMAND ----------

import pickle
with open('question_embeddings_app.pkl', 'rb') as file:
      
    # Call load method to deserialze
    question_embeddings = pickle.load(file)
  
    print(question_embeddings)

# COMMAND ----------

so_database['embeddings'] = question_embeddings.tolist()

# COMMAND ----------

so_database

# COMMAND ----------

## Semantic Search

When a user asks a question, we can embed their query on the fly and search over all of the Stack Overflow question embeddings to find the most simliar datapoint.

# COMMAND ----------

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin as distances_argmin

# COMMAND ----------

query = ['How to concat dataframes pandas']

# COMMAND ----------

query_embedding = embedding_model.get_embeddings(query)[0].values

# COMMAND ----------

cos_sim_array = cosine_similarity([query_embedding],
                                  list(so_database.embeddings.values))

# COMMAND ----------

cos_sim_array.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have a similarity value between our query embedding and each of the database embeddings, we can extract the index with the highest value. This embedding corresponds to the Stack Overflow post that is most similiar to the question "How to concat dataframes pandas".

# COMMAND ----------

index_doc_cosine = np.argmax(cos_sim_array)

# COMMAND ----------

index_doc_distances = distances_argmin([query_embedding], 
                                       list(so_database.embeddings.values))[0]

# COMMAND ----------

so_database.input_text[index_doc_cosine]

# COMMAND ----------

so_database.output_text[index_doc_cosine]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question answering with relevant context
# MAGIC
# MAGIC Now that we have found the most simliar Stack Overflow question, we can take the corresponding answer and use an LLM to produce a more conversational response.

# COMMAND ----------

from vertexai.language_models import TextGenerationModel

# COMMAND ----------

generation_model = TextGenerationModel.from_pretrained(
    "text-bison@001")

# COMMAND ----------

context = "Question: " + so_database.input_text[index_doc_cosine] +\
"\n Answer: " + so_database.output_text[index_doc_cosine]

# COMMAND ----------

prompt = f"""Here is the context: {context}
             Using the relevant information from the context,
             provide an answer to the query: {query}."
             If the context doesn't provide \
             any relevant information, \
             answer with \
             [I couldn't find a good match in the \
             document database for your query]
             """

# COMMAND ----------

from IPython.display import Markdown, display

t_value = 0.2
response = generation_model.predict(prompt = prompt,
                                    temperature = t_value,
                                    max_output_tokens = 1024)

display(Markdown(response.text))

# COMMAND ----------

# MAGIC %md
# MAGIC ## When the documents don't provide useful information
# MAGIC
# MAGIC Our current workflow returns the most similar question from our embeddings database. But what do we do when that question isn't actually relevant when answering the user query? In other words, we don't have a good match in our database.
# MAGIC
# MAGIC In addition to providing a more conversational response, LLMs can help us handle these cases where the most similiar document isn't actually a reasonable answer to the user's query.

# COMMAND ----------

query = ['How to make the perfect lasagna']

# COMMAND ----------

query = ['How to make the perfect lasagna']

# COMMAND ----------

cos_sim_array = cosine_similarity([query_embedding], 
                                  list(so_database.embeddings.values))

# COMMAND ----------

cos_sim_array

# COMMAND ----------

index_doc = np.argmax(cos_sim_array)

# COMMAND ----------

context = so_database.input_text[index_doc] + \
"\n Answer: " + so_database.output_text[index_doc]

# COMMAND ----------

prompt = f"""Here is the context: {context}
             Using the relevant information from the context,
             provide an answer to the query: {query}."
             If the context doesn't provide \
             any relevant information, answer with 
             [I couldn't find a good match in the \
             document database for your query]
             """

# COMMAND ----------

t_value = 0.2
response = generation_model.predict(prompt = prompt,
                                    temperature = t_value,
                                    max_output_tokens = 1024)
display(Markdown(response.text))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale with approximate nearest neighbor search
# MAGIC
# MAGIC When dealing with a large dataset, computing the similarity between the query and each original embedded document in the database might be too expensive. Instead of doing that, you can use approximate nearest neighbor algorithms that find the most similar documents in a more efficient way.
# MAGIC
# MAGIC These algorithms usually work by creating an index for your data, and using that index to find the most similar documents for your queries. In this notebook, we will use ScaNN to demonstrate the benefits of efficient vector similarity search. First, you have to create an index for your embedded dataset.

# COMMAND ----------

import scann
from utils import create_index

#Create index using scann
index = create_index(embedded_dataset = question_embeddings, 
                     num_leaves = 25,
                     num_leaves_to_search = 10,
                     training_sample_size = 2000)

# COMMAND ----------

query = "how to concat dataframes pandas"

# COMMAND ----------

import time 

start = time.time()
query_embedding = embedding_model.get_embeddings([query])[0].values
neighbors, distances = index.search(query_embedding, final_num_neighbors = 1)
end = time.time()

for id, dist in zip(neighbors, distances):
    print(f"[docid:{id}] [{dist}] -- {so_database.input_text[int(id)][:125]}...")

print("Latency (ms):", 1000 * (end - start))

# COMMAND ----------

start = time.time()
query_embedding = embedding_model.get_embeddings([query])[0].values
cos_sim_array = cosine_similarity([query_embedding], list(so_database.embeddings.values))
index_doc = np.argmax(cos_sim_array)
end = time.time()

print(f"[docid:{index_doc}] [{np.max(cos_sim_array)}] -- {so_database.input_text[int(index_doc)][:125]}...")

print("Latency (ms):", 1000 * (end - start))

# COMMAND ----------

# MAGIC %md
# MAGIC In this final lesson, we're going to take everything we've learned 
# MAGIC about embeddings, semantic similarity, and text 
# MAGIC generation, and put them all together to build a question answering system. 
# MAGIC Specifically, the system will take in a user question 
# MAGIC about Python programming and return an 
# MAGIC answer based on a database of stack overflow posts. So, let's 
# MAGIC get to building. Let's say we wanted to use a large language model to answer 
# MAGIC a question like how to concatenate data frames 
# MAGIC in pandas. We could ask this 
# MAGIC model directly, but out of the box, large language models aren't 
# MAGIC connected to the external world. So, this means that they 
# MAGIC don't have access to information that's outside of their training data. So, 
# MAGIC this question about data frames and pandas, it's not that 
# MAGIC specific. But, imagine if you wanted to answer questions about 
# MAGIC an organization you work for or some kind 
# MAGIC of really specialized domain. 
# MAGIC  
# MAGIC In these examples, you'll probably need to give the LLM 
# MAGIC access to data that wasn't in its training data. So for 
# MAGIC example, you might want to connect it to some external database 
# MAGIC of documents. But in reality, you can't take all 
# MAGIC those documents and stuff them into a prompt. You're going to run 
# MAGIC out of space pretty quickly. Now, another reason you might 
# MAGIC want to connect a large language model to 
# MAGIC an external database is if you want to 
# MAGIC be able to trace the lineage of a response. So, you might've heard the 
# MAGIC term hallucinations. This is where sometimes large 
# MAGIC language models produce responses that seem 
# MAGIC plausible but aren't actually grounded in reality or 
# MAGIC factually accurate. If we connect a large language 
# MAGIC model to an external database, then we can base an answer 
# MAGIC or a response in a particular document and have a way of 
# MAGIC tracing the origins of that answer. This is 
# MAGIC often known as grounding in LLM. 
# MAGIC  
# MAGIC Now, you might think that, well, if we have all these documents, 
# MAGIC we should probably fine-tune a model on all of 
# MAGIC this new text data. But actually, we can do all of this 
# MAGIC without having to do any kind of specialized tuning. Instead, 
# MAGIC we'll use what we've learned about embeddings and a little bit 
# MAGIC of prompting. So, let's check this out in a notebook. We'll start off by doing 
# MAGIC our usual authentication step and then setting the region 
# MAGIC and making sure that we import and initialize the Vertex AI Python 
# MAGIC SDK. Once we've done that setup, we can go ahead and start 
# MAGIC getting our data set created. So, as we did in an 
# MAGIC earlier lab, we're going to use the Stack Overflow data set from BigQuery. But 
# MAGIC this time, we won't be running any BigQuery code. We've got 
# MAGIC a CSV file prepared for you already. So, we will just use Pandas to import 
# MAGIC this data. So, we're going to call this DataFrame our SO database for 
# MAGIC Stack Overflow database, and we can print out 
# MAGIC the shape and the first few rows of this DataFrame. 
# MAGIC So, this data should look familiar. 
# MAGIC  
# MAGIC It's very similar to a DataFrame we used in 
# MAGIC the previous lesson. It's got 2,000 rows, so 2,000 different Stack Overflow 
# MAGIC posts, and there are three columns. The input text, again, 
# MAGIC is the question and title of the Stack Overflow post 
# MAGIC concatenated. The output text is 
# MAGIC the accepted response from the community to that 
# MAGIC question, and the category is the programming language that 
# MAGIC the post was tagged with. So, now that we have 
# MAGIC our data, we can go ahead and embed it. We will import our 
# MAGIC text embedding model, and then we will load the text embedding Gecko 
# MAGIC model. Now, earlier we talked about how if you are trying 
# MAGIC to embed a large amount of data, you'll need to be aware 
# MAGIC of batching the data and also managing rate limits and all that 
# MAGIC is taken care of for you in this encode text to embedding 
# MAGIC batched utility function. 
# MAGIC  
# MAGIC So, just as a reminder if you wanted to 
# MAGIC actually use this in your own projects this 
# MAGIC is the code you would run, you'd call this encode text to embedding 
# MAGIC batched function passing in your data frame, but 
# MAGIC we aren't again going to actually run this model just because 
# MAGIC we want to save on API calls. So, we've already embedded 
# MAGIC the data, and we're just going to 
# MAGIC load in that pre-created embedding data. These pre-computed embeddings are saved as a 
# MAGIC pickle file. 
# MAGIC So, first we'll import pickle and then we can open this file and we will 
# MAGIC load it and then print out the result just to see what 
# MAGIC it looks like. So here, we've got our array of embeddings, and these 
# MAGIC are the embeddings we'll use for this particular lesson. So now, that 
# MAGIC we have all these embeddings, we're going to 
# MAGIC add these embeddings as a column to our data 
# MAGIC frame. And this will just make a few things 
# MAGIC easier a little bit later when we go 
# MAGIC to build our question answering system. So, we 
# MAGIC can take a look at our data frame 
# MAGIC and we've just added this additional column, which is the embeddings vector 
# MAGIC for each of these Stack Overflow posts. So, 
# MAGIC why did we embed all of that data? 
# MAGIC Well, our Stack Overflow dataset is comprised of questions 
# MAGIC and the accepted answers. And what we'd like to do is we'd like 
# MAGIC to take a query from our user of this system 
# MAGIC and look at all of the Stack Overflow questions and 
# MAGIC see if there's a similar question in this database. And if there 
# MAGIC is a similar question, then that's great news because that means we have 
# MAGIC an answer to that question and therefore we 
# MAGIC have an answer for our user. Now, earlier we talked about 
# MAGIC how we can use embeddings to help us 
# MAGIC find similar data points. We can actually quantify 
# MAGIC how similar two embeddings are by using some 
# MAGIC sort of distance metric. And there are a 
# MAGIC few different common distance metrics you might use. 
# MAGIC The first is Euclidean distance, which is the distance 
# MAGIC between the ends of the two vectors. This is also L2 
# MAGIC distance. 
# MAGIC We also have cosine similarity, which is what we 
# MAGIC used in the previous lessons, which calculates the cosine of the 
# MAGIC angle between the two vectors. And then, there's the dot product, which 
# MAGIC is the cosine multiplied by the lengths of both vectors. Note 
# MAGIC that the dot product does take into account both the angle and 
# MAGIC the magnitude of the two vectors. The magnitude can be useful 
# MAGIC in certain use cases, like recommendation systems. It's not 
# MAGIC as important in our particular example. So, we'll be using 
# MAGIC cosine similarity, but the cosine similarity and dot product are actually 
# MAGIC both equivalent when your vectors are 
# MAGIC normalized to a magnitude of 1. So, what we're going to do 
# MAGIC next is we will take our user query and we 
# MAGIC will embed it, and then we'll compute the similarity between this embedded user 
# MAGIC query and every single embedded Stack 
# MAGIC Overflow question in our database. 
# MAGIC  
# MAGIC Once we've done that, we can identify which 
# MAGIC embedded questions are the most similar, and 
# MAGIC these are known as the nearest neighbors. So, let's go 
# MAGIC ahead and try this out in the notebook. We'll start by 
# MAGIC importing a few libraries. We'll need to import NumPy, and then we will 
# MAGIC also import the cosine similarity metric. We will also import 
# MAGIC another metric as well, so that we can 
# MAGIC just see what it looks like to use Euclidean distance. But in 
# MAGIC this example, we'll be sticking with cosine 
# MAGIC similarity. So, let's say that we have a user that asks a question like, 
# MAGIC how to concatenate data frames in pandas? We'll start off by embedding 
# MAGIC this query. We will call getEmbeddings again on our 
# MAGIC embeddings model, and then we will extract the values. 
# MAGIC  
# MAGIC The next thing we're going to do is calculate the 
# MAGIC cosine similarity between our query embedding 
# MAGIC and every embedding in our database. We'll start 
# MAGIC by using the cosine similarity function from scikit-learn and then 
# MAGIC we'll pass in our query embedding. and this is the 
# MAGIC embedding for the input text, how to concatenate data 
# MAGIC frames and pandas. And we are wrapping this in a list and 
# MAGIC that's just because it's a list, but we need to send a 
# MAGIC list of lists to these cosines similarity function. So, that's what it 
# MAGIC looks like. Next, we will pass in the database of 
# MAGIC stack overflow posts. And we're converting the stack overflow database 
# MAGIC into a list, and that's just because it's currently an array, 
# MAGIC but we need this 2D list to pass to this 
# MAGIC cosine similarity function. So, we print this out, we should see that list 
# MAGIC of lists. There you go. Great. 
# MAGIC  
# MAGIC Now, that we've done that, we can compute the cosine similarity. So, 
# MAGIC let's take a look at the shape of this array. It's 1 by 
# MAGIC 2,000 and that is one distance metric calculated for every 
# MAGIC single stack overflow embedding we have 
# MAGIC in our database. So again, just to recap, we took our input 
# MAGIC query and we calculated the cosine similarity between 
# MAGIC that query and every single one of these 
# MAGIC 2,000 stack overflow embeddings. So, from this array, we want 
# MAGIC to figure out what is the most similar stack overflow embedding 
# MAGIC to our question embedding. So, we're going to extract 
# MAGIC the index with the highest value. So, just as a quick aside, if 
# MAGIC you wanted to use a different distance metric and try out this whole use 
# MAGIC case with something other than cosine similarity, you could 
# MAGIC use this distance argmin function from scikit-learn, and 
# MAGIC this would compute the Euclidean distance. But we're going 
# MAGIC to stick with using the cosine similarity for this use 
# MAGIC case. 
# MAGIC So now, that we have computed the cosine similarity, 
# MAGIC and we've extracted the index with the highest similarity, we 
# MAGIC can go and see which question this actually corresponds 
# MAGIC to. So, this question is about concatenating 
# MAGIC objects in pandas. It's not exactly the same, but it 
# MAGIC is pretty similar to our input question. And we 
# MAGIC can go and grab the corresponding answer as well. So, 
# MAGIC if we were to just return this answer to a user, I think 
# MAGIC it would be a pretty unsatisfying user experience. There's some 
# MAGIC strange formatting. It doesn't really sound like it's in context. So, what 
# MAGIC we're going to do now is we're going to use a large language 
# MAGIC model and use all this information as relevant 
# MAGIC context to format a much better and more 
# MAGIC user-friendly response for our question answering system. 
# MAGIC  
# MAGIC To do this, we'll start by importing the text generation 
# MAGIC model that we used in the previous lesson. And then, we 
# MAGIC will load in the text bison model. And now, that 
# MAGIC we have our model loaded, we can go ahead and format a prompt. So, 
# MAGIC we'll start by creating some context that will go into 
# MAGIC our prompt. So, here's the context. We've got the text 
# MAGIC question, and then we've got the actual question 
# MAGIC for the Stack Overflow post, followed by the answer, and 
# MAGIC that is the answer, the accepted answer for the 
# MAGIC Stack. Overflow post. And again, we are selecting this Stack 
# MAGIC Overflow post based on the cosine similarity that we 
# MAGIC calculated in the previous section. So, we're going to 
# MAGIC create a prompt, and we will include this context 
# MAGIC as part of our prompt. So, here's the prompt we're writing. We 
# MAGIC say, here's the context, and then we include all of this context, which 
# MAGIC is the question answer pair from our Stack Overflow database. And 
# MAGIC we say, using the relevant information from this context, provide an 
# MAGIC answer to the query. 
# MAGIC  
# MAGIC Here, we input the user's question, which was about concatenating data 
# MAGIC frames in pandas. We also instruct the model 
# MAGIC to provide a different answer if there isn't relevant 
# MAGIC information in this context. We say that it should respond with, I couldn't 
# MAGIC find a good match in the document database for your query. And we'll see 
# MAGIC why this comes in handy in just a little bit. So now, that we've defined this 
# MAGIC prompt, we can go ahead and call the model, we will call predict and 
# MAGIC we will pass in the prompt as well as a temperature value 
# MAGIC and set some maximum output tokens. And this 
# MAGIC argument here just limits how many tokens 
# MAGIC the model outputs. So, let's display this response and we'll 
# MAGIC display it in Markdown just to make it a little bit 
# MAGIC easier and nicer to read. So here, we've got an answer from 
# MAGIC our model about concatenating data frames using the concat 
# MAGIC function and as well as that 
# MAGIC an example. So, basically we took the answer from our Stack 
# MAGIC Overflow database, and we passed that into our text generation 
# MAGIC model with a little bit of additional context and had it formulate 
# MAGIC a more user-friendly and conversational response. 
# MAGIC  
# MAGIC Now, our current workflow returns the most 
# MAGIC similar question from our embeddings database. 
# MAGIC But what do we do if our user query doesn't 
# MAGIC really have anything to do with the information 
# MAGIC in our database? In addition to providing a 
# MAGIC more conversational response, we can use a text 
# MAGIC generation model to help us handle these cases 
# MAGIC where the most similar document in our database 
# MAGIC isn't actually a reasonable answer to 
# MAGIC our user's query. So, let's start with a different user query. This time, instead of 
# MAGIC asking about pandas, we'll say our user is 
# MAGIC asking how to make the perfect lasagna. While an interesting question, the 
# MAGIC answer is definitely not in this database of Stack Overflow 
# MAGIC posts about Python. 
# MAGIC First, we'll embed this query using the getEmbeddings function and 
# MAGIC our embedding model. And once we've done that, we will 
# MAGIC compute the cosine similarity between this 
# MAGIC embedding and every single embedding in our Stack Overflow 
# MAGIC database. So, that's exactly what we did in the 
# MAGIC previous section, but just with a different query. And we can 
# MAGIC take a look at this array. And again, this is the cosine similarity value computed 
# MAGIC between our query and all 2,000 of our Stack 
# MAGIC Overflow embeddings. From this array, we will extract the 
# MAGIC index with the highest value, and then we can 
# MAGIC use the same prompt we used before. So, we'll define our context, 
# MAGIC which is the document that is most similar to 
# MAGIC our input query. We'll have the question as well 
# MAGIC as the answer. And then, we will also put this into 
# MAGIC our prompt. 
# MAGIC So, let's take a look at this prompt again. We've 
# MAGIC got the context, which is the Stack Overflow question 
# MAGIC and answer that was most similar to our 
# MAGIC user query. We also provide the model with 
# MAGIC the user query about how to make the 
# MAGIC perfect lasagna, and then we instruct it to 
# MAGIC respond with, I couldn't find a match in the document 
# MAGIC database for your query if the stack overflow 
# MAGIC information isn't actually relevant. So, hopefully if we run 
# MAGIC this, we should get back a response that there 
# MAGIC was no good match because we definitely don't 
# MAGIC have any information about lasagna in our database. 
# MAGIC So, we'll lastly call predict with our generation model, we'll pass 
# MAGIC in this prompt, and we will print out the response. 
# MAGIC And here we go. We've got this response from our model, which 
# MAGIC says that there wasn't a good match in the database for this 
# MAGIC query. 
# MAGIC So, I encourage you to try out maybe some different user queries. 
# MAGIC You can also try experimenting with slightly different prompts, 
# MAGIC see how that impacts the results from the model, 
# MAGIC and maybe you can even get an even better response. 
# MAGIC Before we wrap up today, I wanted to add a note that we computed 
# MAGIC the cosine similarity between our query 
# MAGIC embedding and every single embedding in 
# MAGIC our Stack Overflow database. But this exhaustive search isn't 
# MAGIC really feasible when your database is hundreds of millions 
# MAGIC or even billions of vectors. Instead, for 
# MAGIC production use cases, you'll often end up using an 
# MAGIC algorithm that performs an approximate match. 
# MAGIC Now, if you're using a vector database, this 
# MAGIC will probably be taken care of for you. But if you want 
# MAGIC to try out one of these approximate nearest neighbor 
# MAGIC algorithms, one that you can use is called scan or 
# MAGIC scalable nearest neighbors. This is an approximate 
# MAGIC nearest neighbor algorithm and there's an open-source library that 
# MAGIC you can try out which performs efficient vector similarity 
# MAGIC search at scale. So, let's quickly try this 
# MAGIC out in the code. So, I'll first import the scan library and you can 
# MAGIC pip install this with pip install scan, but 
# MAGIC it's already installed in this environment, so we'll just import it. And 
# MAGIC then, we'll also need to import a utility function, which you can go 
# MAGIC check out later if you're curious to learn a little bit more 
# MAGIC about how this works. 
# MAGIC But this function is just going to create 
# MAGIC an index and you can think of an index as being this 
# MAGIC collection of all of our vectors. So, it will be all 
# MAGIC of our question embeddings. This is all of 
# MAGIC our embedded stack overflow questions. So, we can create this 
# MAGIC index and there are a few other parameters 
# MAGIC here that you can set like the number of leaves, 
# MAGIC the number of leaves to search. And we're just keeping these 
# MAGIC at some simple defaults for this example. Now, we'll take 
# MAGIC our input query again, our users asking, how do 
# MAGIC we concatenate data frames and pandas? And then, we can 
# MAGIC use this index to find the most similar document. But just 
# MAGIC to see if we get any speed benefits by doing this approximate 
# MAGIC search instead of the exhaustive search, we will 
# MAGIC import the time library and record how long 
# MAGIC this takes. 
# MAGIC So first, we are going to start the timer, 
# MAGIC and then we will embed this query, we will embed how to 
# MAGIC concatenate data frames and pandas. And once we've done that, we 
# MAGIC will pass this query embedding to our index 
# MAGIC and call the search function. And then, once that's done, we will 
# MAGIC record the time. So, let's also go ahead and print out the responses. So, this 
# MAGIC will print out the nearest neighbor. And then, we will also 
# MAGIC print out the latency. So, that'll be how long this took. So, let's see. That 
# MAGIC was pretty fast. And here's the ID of our document on our Stack Overflow database, 
# MAGIC as well as the similarity metric. 
# MAGIC And you can see that this is the same Stack Overflow post that we 
# MAGIC identified earlier when we did that exhaustive search 
# MAGIC about adding a column to a panda's data 
# MAGIC frame. 
# MAGIC So now, we'll compare this scan algorithm to doing 
# MAGIC an exhaustive search, which is what we did 
# MAGIC earlier when we computed the cosine similarity between 
# MAGIC each of our vectors in our database. So here, we'll 
# MAGIC set a timer again, and then we will call the 
# MAGIC getEmbeddings function on our embeddings model, passing in 
# MAGIC our query. And then, we will call this cosine similarity 
# MAGIC function again, passing in our query and all the embeddings 
# MAGIC in our database. Then, we will also go ahead and 
# MAGIC print out the most similar document like we 
# MAGIC did previously, and we will also calculate the time 
# MAGIC that it took. So let's run this cell. And we can see that we have the 
# MAGIC same document identified. 
# MAGIC This is the same stack overflow post about adding a columns name 
# MAGIC in pandas and concatenating these objects, 
# MAGIC but it just took a decent amount more time. 
# MAGIC You can see here, we've got 80 milliseconds when we use 
# MAGIC the approximate nearest neighbor algorithm and 
# MAGIC then 182 when we used this exhaustive search. Now, this 
# MAGIC was all pretty quick because we had a 
# MAGIC pretty tiny database, but the speed gains would be 
# MAGIC a lot more noticeable if you had a large data set. So now, you've 
# MAGIC seen how you can build a small scale question 
# MAGIC answering system using Stack Overflow data, but everything 
# MAGIC you used in this lesson, you could apply to a data set of your own 
# MAGIC to build your own custom question answering system. 
# MAGIC  
# MAGIC So, to wrap up, let's summarize everything we did 
# MAGIC in this lesson. We took in all of our 
# MAGIC database of stack overflow questions, and 
# MAGIC we also took in a user query. And we passed all of this 
# MAGIC to our embeddings model. Once we had all of 
# MAGIC our embedded questions and our embedded query, we 
# MAGIC could compute a nearest neighbor search where we computed the cosine similarity 
# MAGIC between the embedded query and all of the embedded 
# MAGIC questions in our database. From that, we found the most 
# MAGIC similar question and we extracted its answer. And 
# MAGIC we passed that along with the user query into our text generation 
# MAGIC model to produce a nice conversational system answer for 
# MAGIC our end user. 
# MAGIC

# COMMAND ----------


