# Databricks notebook source
# MAGIC %md
# MAGIC ## Lesson 4: Applications of Embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC #### Project environment setup
# MAGIC
# MAGIC - Load credentials and relevant Python Libraries

# COMMAND ----------

from utils import authenticate
credentials, PROJECT_ID = authenticate()

# COMMAND ----------

REGION = 'us-central1'

# COMMAND ----------

import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)

# COMMAND ----------

#### Load Stack Overflow questions and answers from BigQuery
- BigQuery is Google Cloud's serverless data warehouse.
- We'll get the first 500 posts (questions and answers) for each programming language: Python, HTML, R, and CSS.

# COMMAND ----------

from google.cloud import bigquery
import pandas as pd

# COMMAND ----------

from google.cloud import bigquery
import pandas as pd

# COMMAND ----------

# define list of programming language tags we want to query

language_list = ["python", "html", "r", "css"]

# COMMAND ----------

so_df = pd.DataFrame()

for language in language_list:
    
    print(f"generating {language} dataframe")
    
    query = f"""
    SELECT
        CONCAT(q.title, q.body) as input_text,
        a.body AS output_text
    FROM
        `bigquery-public-data.stackoverflow.posts_questions` q
    JOIN
        `bigquery-public-data.stackoverflow.posts_answers` a
    ON
        q.accepted_answer_id = a.id
    WHERE 
        q.accepted_answer_id IS NOT NULL AND 
        REGEXP_CONTAINS(q.tags, "{language}") AND
        a.creation_date >= "2020-01-01"
    LIMIT 
        500
    """

    
    language_df = run_bq_query(query)
    language_df["category"] = language
    so_df = pd.concat([so_df, language_df], 
                      ignore_index = True) 

# COMMAND ----------

# MAGIC %md
# MAGIC - You can reuse the above code to run your own queries if you are using Google Cloud's BigQuery service.
# MAGIC - In this classroom, if you run into any issues, you can load the same data from a csv file.

# COMMAND ----------

# Run this cell if you get any errors or you don't want to wait for the query to be completed
# so_df = pd.read_csv('so_database_app.csv')

# COMMAND ----------

so_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate text embeddings
# MAGIC - To generate embeddings for a dataset of texts, we'll need to group the sentences together in batches and send batches of texts to the model.
# MAGIC - The API currently can take batches of up to 5 pieces of text per API call.

# COMMAND ----------

from vertexai.language_models import TextEmbeddingModel

# COMMAND ----------

model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

# COMMAND ----------

import time
import numpy as np

# COMMAND ----------

# Generator function to yield batches of sentences

def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

# COMMAND ----------

so_questions = so_df[0:200].input_text.tolist() 
batches = generate_batches(sentences = so_questions)

# COMMAND ----------

batch = next(batches)
len(batch)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get embeddings on a batch of data
# MAGIC - This helper function calls `model.get_embeddings()` on the batch of data, and returns a list containing the embeddings for each text in that batch.

# COMMAND ----------

def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

# COMMAND ----------

batch_embeddings = encode_texts_to_embeddings(batch)

# COMMAND ----------

f"{len(batch_embeddings)} embeddings of size \
{len(batch_embeddings[0])}"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Code for getting data on an entire data set
# MAGIC - Most API services have rate limits, so we've provided a helper function (in utils.py) that you could use to wait in-between API calls.
# MAGIC - If the code was not designed to wait in-between API calls, you may not receive embeddings for all batches of text.
# MAGIC - This particular service can handle 20 calls per minute.  In calls per second, that's 20 calls divided by 60 seconds, or `20/60`.
# MAGIC
# MAGIC ```Python
# MAGIC from utils import encode_text_to_embedding_batched
# MAGIC
# MAGIC so_questions = so_df.input_text.tolist()
# MAGIC question_embeddings = encode_text_to_embedding_batched(
# MAGIC                             sentences=so_questions,
# MAGIC                             api_calls_per_second = 20/60, 
# MAGIC                             batch_size = 5)
# MAGIC ```
# MAGIC
# MAGIC In order to handle limits of this classroom environment, we're not going to run this code to embed all of the data. But you can adapt this code for your own projects and datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the data from file
# MAGIC - We'll load the stack overflow questions, answers, and category labels (Python, HTML, R, CSS) from a .csv file.
# MAGIC - We'll load the embeddings of the questions (which we've precomputed with batched calls to `model.get_embeddings()`), from a pickle file.

# COMMAND ----------

so_df = pd.read_csv('so_database_app.csv')
so_df.head()

# COMMAND ----------

import pickle

# COMMAND ----------

with open('question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)

# COMMAND ----------

print("Shape: " + str(question_embeddings.shape))
print(question_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cluster the embeddings of the Stack Overflow questions

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# COMMAND ----------

clustering_dataset = question_embeddings[:1000]

# COMMAND ----------

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0, 
                n_init = 'auto').fit(clustering_dataset)

# COMMAND ----------

kmeans_labels = kmeans.labels_

# COMMAND ----------

PCA_model = PCA(n_components=2)
PCA_model.fit(clustering_dataset)
new_values = PCA_model.transform(clustering_dataset)

# COMMAND ----------

import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl

# COMMAND ----------

from utils import clusters_2D
clusters_2D(x_values = new_values[:,0], y_values = new_values[:,1], 
            labels = so_df[:1000], kmeans_labels = kmeans_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC - Clustering is able to identify two distinct clusters of HTML or Python related questions, without being given the category labels (HTML or Python).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Anomaly / Outlier detection
# MAGIC
# MAGIC - We can add an anomalous piece of text and check if the outlier (anomaly) detection algorithm (Isolation Forest) can identify it as an outlier (anomaly), based on its embedding.

# COMMAND ----------

from sklearn.ensemble import IsolationForest

# COMMAND ----------

input_text = """I am making cookies but don't 
                remember the correct ingredient proportions. 
                I have been unable to find 
                anything on the web."""

# COMMAND ----------

emb = model.get_embeddings([input_text])[0].values

# COMMAND ----------

embeddings_l = question_embeddings.tolist()
embeddings_l.append(emb)

# COMMAND ----------

embeddings_array = np.array(embeddings_l)

# COMMAND ----------

print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)

# COMMAND ----------

# Add the outlier text to the end of the stack overflow dataframe
so_df = pd.read_csv('so_database_app.csv')
new_row = pd.Series([input_text, None, "baking"], 
                    index=so_df.columns)
so_df.loc[len(so_df)+1] = new_row
so_df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use Isolation Forest to identify potential outliers
# MAGIC
# MAGIC - `IsolationForest` classifier will predict `-1` for potential outliers, and `1` for non-outliers.
# MAGIC - You can inspect the rows that were predicted to be potential outliers and verify that the question about baking is predicted to be an outlier.

# COMMAND ----------

clf = IsolationForest(contamination=0.005, 
                      random_state = 2) 

# COMMAND ----------

preds = clf.fit_predict(embeddings_array)

print(f"{len(preds)} predictions. Set of possible values: {set(preds)}")

# COMMAND ----------

so_df.loc[preds == -1]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remove the outlier about baking

# COMMAND ----------

so_df = so_df.drop(so_df.index[-1])

# COMMAND ----------

so_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification
# MAGIC - Train a random forest model to classify the category of a Stack Overflow question (as either Python, R, HTML or CSS).

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

# re-load the dataset from file
so_df = pd.read_csv('so_database_app.csv')
X = question_embeddings
X.shape

# COMMAND ----------

y = so_df['category'].values
y.shape

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 2)

# COMMAND ----------

clf = RandomForestClassifier(n_estimators=200)

# COMMAND ----------

clf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### You can check the predictions on a few questions from the test set

# COMMAND ----------

y_pred = clf.predict(X_test)

# COMMAND ----------

accuracy = accuracy_score(y_test, y_pred) # compute accuracy
print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Try out the classifier on some questions

# COMMAND ----------

# choose a number between 0 and 1999
i = 2
label = so_df.loc[i,'category']
question = so_df.loc[i,'input_text']

# get the embedding of this question and predict its category
question_embedding = model.get_embeddings([question])[0].values
pred = clf.predict([question_embedding])

print(f"For question {i}, the prediction is `{pred[0]}`")
print(f"The actual label is `{label}`")
print("The question text is:")
print("-"*50)
print(question)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that you've had a chance to get some intuition about embeddings, let's 
# MAGIC check out some different applications. As we've 
# MAGIC done before, we'll start by setting up our credentials and 
# MAGIC authenticating. We'll also need to specify 
# MAGIC the region we'll be running this service in, and then we can import the 
# MAGIC Vertex AI Python SDK and initialize it. Now 
# MAGIC that we've done that setup, we're ready to start loading in our data. 
# MAGIC For this tutorial, we're going to use the Stack Overflow dataset of 
# MAGIC question and answers, which is hosted in BigQuery, 
# MAGIC a serverless data warehouse. So, we'll start by importing 
# MAGIC the BigQuery Python client. And once we've done that, we 
# MAGIC can write a function that will take in a SQL 
# MAGIC query as a string and execute this query 
# MAGIC in BigQuery. 
# MAGIC So, let's go ahead and paste in that function. So 
# MAGIC again, this function is going to take in some SQL, and 
# MAGIC it will run that query in BigQuery, and it will return the results 
# MAGIC as a pandas data frame, which we can then use 
# MAGIC in our notebook for some different applications. Now, 
# MAGIC you don't really need to know a whole lot about how 
# MAGIC this BigQuery function works, so don't worry if 
# MAGIC you don't understand the details, we're just going to use 
# MAGIC this to get the data set for this lesson. So, the next thing we'll do 
# MAGIC is we'll create a list of the language tags that we want to query, 
# MAGIC and that's just because this data set is very 
# MAGIC large and it won't fit in memory, So, we don't want to 
# MAGIC actually pull in all the data. We just want to pull in a small subset of 
# MAGIC Stack Overflow posts for a few different programming languages. So, to get 
# MAGIC our data, we'll start by creating an 
# MAGIC empty data frame. And then, we're going to loop over 
# MAGIC this list of languages. And we will execute a SQL query in BigQuery. So here 
# MAGIC is the query we'll be running. And if you're not 
# MAGIC familiar with SQL, again, don't worry too much about it, but this 
# MAGIC will basically pull the first 500 posts for a particular language tag from 
# MAGIC the Stack Overflow data set. 
# MAGIC  
# MAGIC  
# MAGIC And then, once we've done that, we want to concatenate all 
# MAGIC the results into one single data frame, which 
# MAGIC we will use in this notebook. So, let's run the cell, and this 
# MAGIC might take a minute or so. You can see that it's pulling the 
# MAGIC data for each of the four languages that we're interested in. Now, 
# MAGIC note that if you ran into any errors while 
# MAGIC trying to execute the BigQuery code, or you just don't 
# MAGIC want to use BigQuery at all, you can just run this 
# MAGIC cell right here, which will just pull in the data for you from a 
# MAGIC CSV file, so you don't have to worry about using 
# MAGIC BigQuery. But, that's an optional step if you ran into any errors or 
# MAGIC you just didn't want to execute that code. So now, 
# MAGIC that we have all of our data, let's go ahead and examine it and see what 
# MAGIC it actually looks like. 
# MAGIC Here's our data frame. We can see at the bottom here that it's 2,000 
# MAGIC rows by three columns. So that's 2,000 different Stack Overflow posts and it's 500 
# MAGIC posts for each of the languages that we queried for. 
# MAGIC And these three columns here we have first the input text which 
# MAGIC is the title of the Stack Overflow post concatenated with the question 
# MAGIC of the Stack Overflow post. And then, the output 
# MAGIC text column is the accepted answer from the 
# MAGIC community for that Stack Overflow post. And then 
# MAGIC finally, we've got a column for the category 
# MAGIC which is the programming language. So now, that 
# MAGIC we've got our data, we can now start embedding it and using 
# MAGIC it for some different applications. So, we'll first 
# MAGIC load in our text embedding model, which we've done before. This 
# MAGIC is going to be the text embedding gecko model. And now, in the past 
# MAGIC labs, we went ahead and just started using the embeddings function 
# MAGIC to create embeddings. 
# MAGIC  
# MAGIC But because we have a lot of data in this notebook, we're 
# MAGIC going to need to do a little bit of extra work. We're going 
# MAGIC to need some helper functions to batch the 
# MAGIC data and send it to the embedding API. So first we'll start off 
# MAGIC by defining some helper functions. And the first 
# MAGIC function we're going to use here is called generate 
# MAGIC batches. And this function basically takes in our data 
# MAGIC and creates batches of size five. And the reason we 
# MAGIC need to do that is according to the documentation and at this 
# MAGIC point, the API we're using can only handle up to five text 
# MAGIC instances per request. So, we'll need to take our data 
# MAGIC and split it into batches of five in order to 
# MAGIC get our text embeddings. So, we can try out this function 
# MAGIC and we'll take just the first 200 rows of our data frame, and 
# MAGIC we'll call generateBatches() on those 200 rows, and we can see 
# MAGIC what the result is of using this function. 
# MAGIC  
# MAGIC If we call generateBatches() on this subset of our data frame, 
# MAGIC you can see that it creates these nice batches that 
# MAGIC are of size five and that'll be very useful when we 
# MAGIC want to embed all of the data in our data frame. The 
# MAGIC next helper function we'll define is called encodeTextToEmbeddings, which 
# MAGIC is a wrapper around the 
# MAGIC function getEmbeddings, which you've used in the previous labs to 
# MAGIC get embeddings for our text input. So, let's try running 
# MAGIC this function on a batch of our data. We just created a 
# MAGIC batch of five sentences, so we can run 
# MAGIC this encode text to embeddings function, and then we can 
# MAGIC print out the length of the result. So here, if we 
# MAGIC run encode text to embeddings, we get back five 
# MAGIC embeddings, and that's because we passed in five text instances, and 
# MAGIC they're each of size 768. And that number 768 should look familiar by now 
# MAGIC because that's the number of dimensions that the text embedding gecko returns 
# MAGIC for any text input you provide. 
# MAGIC  
# MAGIC So, in addition to making sure you batch your 
# MAGIC instances into batches of size five, there's one other thing you need 
# MAGIC to be aware of, and that's just that most Google Cloud 
# MAGIC services do have rate limits on how many 
# MAGIC requests you can send per minute. So, we have 
# MAGIC written a helper function for you called encode text to embedding batched, 
# MAGIC and this function will manage both batching the data and manage 
# MAGIC the rate limits. So, if you want to use this for 
# MAGIC your own projects, this is the code that you 
# MAGIC would actually execute, but we do want to be mindful of rate limits 
# MAGIC in this online classroom. So, we're not going 
# MAGIC to actually generate embeddings for all 2,000 rows of our 
# MAGIC data right now. Instead, we're just going to load 
# MAGIC this data in. 
# MAGIC But again, for your own projects, you can use this helper function here, 
# MAGIC encode text to embeddings batched, and you pass in the data you want 
# MAGIC embedded, and it will handle both batching the data 
# MAGIC for you and also making sure that the 
# MAGIC rate limits are handled appropriately. Next, 
# MAGIC we are going to load in these embeddings that we've 
# MAGIC generated ahead of time. Just to make sure these embeddings map 
# MAGIC properly to the Stack Overflow questions, we'll also just reload in a 
# MAGIC new CSV file of Stack Overflow questions just to make sure that 
# MAGIC they match and aren't different from what we've loaded in from 
# MAGIC BigQuery. And then next, using pickle, we can load in 
# MAGIC this pickle file, which has all of our embeddings for us. So, let's go ahead 
# MAGIC and take a look at this embeddings vector. 
# MAGIC We'll print out the shape and also the array as well. So, we 
# MAGIC can see that our data is now of size 2,000 by 768. So that's 
# MAGIC our 2,000 stack overflow posts, and each one is 
# MAGIC represented by a 768-dimensional embeddings vector. Now, that we have this data 
# MAGIC and we have it embedded, we can get started with 
# MAGIC some different applications. The first application we'll try 
# MAGIC out is clustering our data, and we're going to use the 
# MAGIC K-Means algorithm to cluster these posts. So first, we'll import K-Means from Scikit-Learn, and 
# MAGIC then we'll also import PCA from Scikit-Learn, which we used 
# MAGIC in an earlier lesson, and this will help us 
# MAGIC to visualize our clusters. Now, just to make 
# MAGIC our visualization a little bit easier, we're actually just going 
# MAGIC to visualize the first 1,000 rows of our data 
# MAGIC set. So, we'll take our full data set of question 
# MAGIC embeddings and we'll only be looking at the first 1,000. 
# MAGIC  
# MAGIC  
# MAGIC These are the posts that were tagged as Python or HTML. 
# MAGIC For our clustering, we'll first need to define the 
# MAGIC number of clusters and we'll set that to two. And then, we will 
# MAGIC create and fit our k-means model on our clustering data set that 
# MAGIC we just created, which is the first 1,000 rows of our 
# MAGIC stack overflow data set. And once we've done that, we can also extract the 
# MAGIC labels and this will just tell us for 
# MAGIC each item in our data set, which of the two clusters does it belong to. 
# MAGIC Now, as before, we can't visualize all of these 768 
# MAGIC dimensions, so we'll use PCA just to represent our data for 2D visualization. So, 
# MAGIC this is code we used in a previous lesson, but we'll 
# MAGIC create our PCA model with the components number set to 2. We'll fit 
# MAGIC this model, and then we will transform the model on our clustering data set. 
# MAGIC  
# MAGIC So, once we've done this, we can now import matplotlib and 
# MAGIC we can plot this data. So, let's go ahead and plot and 
# MAGIC visualize what our clusters look like. So here, we have 
# MAGIC our data, and you can see that it forms 
# MAGIC two pretty distinct clusters. And these are questions that were 
# MAGIC tagged as HTML over here on the left 
# MAGIC with these red circles. And then on the right, we have 
# MAGIC questions that were tagged as Python, and it's pretty good 
# MAGIC at dividing these Stack Overflow posts into two distinct 
# MAGIC categories. And as a reminder, the clustering model 
# MAGIC didn't have these two labels. All it had were 
# MAGIC the embeddings of the Stack Overflow posts, but it was 
# MAGIC able to separate the data into two fairly distinct 
# MAGIC clusters. And we've just added these labels back in to make it 
# MAGIC easier for us to visualize. 
# MAGIC So far, we've talked a lot about how embeddings can help 
# MAGIC us find similar data points. but that also means that we can 
# MAGIC use embeddings to identify points that 
# MAGIC are different or outside of our data distribution. So, we'll now 
# MAGIC use embeddings to help us with anomaly detection 
# MAGIC or outlier detection. To do this, we're going to 
# MAGIC use this isolation forest class in scikit-learn. This will return 
# MAGIC an anomaly score for each sample in our data set 
# MAGIC using this isolation forest algorithm. And note that you 
# MAGIC don't need to know the details of how this algorithm 
# MAGIC works, just that it's an unsupervised learning algorithm that detects data anomalies. 
# MAGIC So, all of the questions in our 
# MAGIC data set are about programming, so we're going to add in 
# MAGIC a question about a very different topic, baking. 
# MAGIC  
# MAGIC Here's some input text. Let's say someone's asking, I'm making cookies, but 
# MAGIC I don't remember the correct ingredient proportions and I've been 
# MAGIC unable to find anything on the web. This 
# MAGIC is definitely pretty different from the questions we have 
# MAGIC in our data set. Once we've defined this input text, we can 
# MAGIC embed it and because it's just a single instance, we can call 
# MAGIC the getEmbeddings function on our embeddings model. Now, 
# MAGIC we'll take this embedding, and we'll append it to 
# MAGIC our array of other embeddings for our stack overflow data. And then, we'll also 
# MAGIC need to convert this data into an array, and 
# MAGIC this will just help us later for visualization and for 
# MAGIC running the isolation forest model. So, let's take 
# MAGIC a look at these new embeddings array. The shape is 
# MAGIC now 2001 by 768, and that's because we've added 
# MAGIC this additional question to our data set before we had 2,000. Stack 
# MAGIC Overflow questions, and now we have 2,001 because 
# MAGIC we've added this extra question about baking. 
# MAGIC  
# MAGIC  
# MAGIC Now, before we can fit our isolation forest model, 
# MAGIC we need to do one more thing, and this is just 
# MAGIC going to make things a little bit easier when we want to 
# MAGIC visualize the results. We will add in a 
# MAGIC new row to our data frame that contains this baking 
# MAGIC question. So, we'll add in the input text. There's no 
# MAGIC output text for this particular question because it wasn't a real 
# MAGIC Stack Overflow post. And we'll say that the 
# MAGIC label is baking. So, let's add this to our data frame. And again, we're just doing 
# MAGIC this because it will help us to visualize the results in 
# MAGIC just a little bit. So now, we're ready to create 
# MAGIC our isolation forest model right here using scikit-learn. And once we've 
# MAGIC created the model, we can fit and predict on our embeddings array. 
# MAGIC And this model will return negative 1 for outliers 
# MAGIC and one for inliers. 
# MAGIC  
# MAGIC So, once it's been fit, we can take our Stack 
# MAGIC Overflow data set and filter for all of the rows that have 
# MAGIC been predicted with negative 1. So, let's see 
# MAGIC what the results look like. The last question here is our 
# MAGIC baking question. And so this was, in fact, identified as 
# MAGIC an outlier. And that is because it's pretty different 
# MAGIC from all of the other examples in this 
# MAGIC data set. You might notice that there are also some programming 
# MAGIC questions about the programming language R 
# MAGIC that were identified as outliers. So if you're 
# MAGIC interested, maybe you can go in and check out the input 
# MAGIC text and see why these particular posts might have 
# MAGIC been identified as outliers. Maybe they were mislabeled, 
# MAGIC and they weren't actually about the programming language R, or 
# MAGIC maybe it was some other reason. And then finally, before we 
# MAGIC move on to the next application, we'll just drop this baking question 
# MAGIC from our dataset because we won't need it 
# MAGIC for the last application in this lesson. So that's 
# MAGIC what this cell does here. We'll just remove it. 
# MAGIC  
# MAGIC And now our stack overflow dataset has been returned to only being 
# MAGIC about programming questions, and it only has 2,000 rows. Now, for 
# MAGIC our final application in this lesson, we'll see how we can also use these 
# MAGIC embedding vectors as features for supervised 
# MAGIC learning. Embeddings take as input some text, and they 
# MAGIC produce some structured output that can be processed 
# MAGIC by a machine. So, this means that we can pass these vectors 
# MAGIC to any of our favorite supervised classification algorithms. 
# MAGIC  
# MAGIC In this lesson, we will be using random forest, 
# MAGIC but feel free to swap this out for another classifier in scikit-learn 
# MAGIC if there's another one you prefer. Now, there are 
# MAGIC many different ways that we could frame a 
# MAGIC classification problem around this data set. Maybe you 
# MAGIC could think about trying to predict if a 
# MAGIC post mentions pandas, or we could try and re-query the data 
# MAGIC and get the score for the different posts 
# MAGIC and try and predict how many upvotes each 
# MAGIC post had. But in this notebook, what we'll try out is 
# MAGIC just predicting the category of the post. So to do that, 
# MAGIC we'll need a couple of other things from scikit-learn. We will also import accuracy score 
# MAGIC and the train test split utility. And then, 
# MAGIC for this prediction task, we're going to define an array called 
# MAGIC X, which will be our embeddings. So, let's go ahead and do that right 
# MAGIC here. And this is just our embeddings data that 
# MAGIC we've created previously in this lesson. And for the 
# MAGIC labels, we will extract the categories for each of these 
# MAGIC posts. 
# MAGIC So, we're just pulling the category column from our stack 
# MAGIC overflow data frame. And now, we have our X and our Y, but 
# MAGIC we got to do one more thing. We need to shuffle the data and we need 
# MAGIC to split it into training and testing sets. So, we'll 
# MAGIC use the scikit-learn train test split utility, and we will 
# MAGIC set an 80-20 split, so that means that our test data will be 
# MAGIC 20% of our original data set. And now, once we've done that, 
# MAGIC we have an X train, an X test, a Y train, and a 
# MAGIC Y test data set, which means we're ready to fit our random forest classifier. So, we'll 
# MAGIC start by creating the classifier, and 
# MAGIC we'll set the number of estimators to 200, 
# MAGIC but you can feel free to change this if you like. And once 
# MAGIC we've created this classifier, we can fit the model. 
# MAGIC And we'll fit the model on our training embeddings and also 
# MAGIC on their corresponding categories. And once this model is 
# MAGIC finished, we'll be able to 
# MAGIC predict on some test data. 
# MAGIC  
# MAGIC So now, we'll call predict and we will pass 
# MAGIC in our X test data set. And again, this is just the embeddings 
# MAGIC in our test set. And we can finally compute the 
# MAGIC accuracy score to see how well this model did. So 0.70, not bad 
# MAGIC for a very minimal pre-processing. So now, you've seen a few 
# MAGIC different ways that we can apply embeddings, we can cluster them, 
# MAGIC we can use them for classification or to 
# MAGIC detect data points outside of our data distribution. 
# MAGIC And in the next tutorial, we're going to take a 
# MAGIC quick break from embeddings and talk a little bit about text generation. 
# MAGIC So, I will see you there. 
# MAGIC

# COMMAND ----------



# COMMAND ----------


