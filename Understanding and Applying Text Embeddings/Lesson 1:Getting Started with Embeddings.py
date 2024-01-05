# Databricks notebook source
# MAGIC %md
# MAGIC Getting Started With Text Embeddings
# MAGIC Project environment setup
# MAGIC Load credentials and relevant Python Libraries
# MAGIC If you were running this notebook locally, you would first install Vertex AI. In this classroom, this is already installed.
# MAGIC !pip install google-cloud-aiplatform

# COMMAND ----------

from utils import authenticate
credentials, PROJECT_ID = authenticate() # Get credentials and project ID

# COMMAND ----------

# MAGIC %md
# MAGIC Enter project details

# COMMAND ----------

print(PROJECT_ID)

# COMMAND ----------

REGION = 'us-central1'

# COMMAND ----------

# Import and initialize the Vertex AI Python SDK

import vertexai
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the embeddings model
# MAGIC Import and load the model.

# COMMAND ----------

from vertexai.language_models import TextEmbeddingModel

# COMMAND ----------

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Generate a word embedding

# COMMAND ----------

embedding = embedding_model.get_embeddings(
    ["life"])

# COMMAND ----------

# MAGIC %md
# MAGIC The returned object is a list with a single TextEmbedding object.
# MAGIC The TextEmbedding.values field stores the embeddings in a Python list.

# COMMAND ----------

vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])

# COMMAND ----------

Generate a sentence embedding.

# COMMAND ----------

embedding = embedding_model.get_embeddings(
    ["What is the meaning of life?"])

# COMMAND ----------

vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC Similarity
# MAGIC Calculate the similarity between two sentences as a number between 0 and 1.
# MAGIC Try out your own sentences and check if the similarity calculations match your intuition.

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

emb_1 = embedding_model.get_embeddings(
    ["What is the meaning of life?"]) # 42!

emb_2 = embedding_model.get_embeddings(
    ["How does one spend their time well on Earth?"])

emb_3 = embedding_model.get_embeddings(
    ["Would you like a salad?"])

vec_1 = [emb_1[0].values]
vec_2 = [emb_2[0].values]
vec_3 = [emb_3[0].values]

# COMMAND ----------

# MAGIC %md
# MAGIC Note: the reason we wrap the embeddings (a Python list) in another list is because the cosine_similarity function expects either a 2D numpy array or a list of lists.
# MAGIC
# MAGIC vec_1 = [emb_1[0].values]

# COMMAND ----------

print(cosine_similarity(vec_1,vec_2)) 
print(cosine_similarity(vec_2,vec_3))
print(cosine_similarity(vec_1,vec_3))

# COMMAND ----------

# MAGIC %md
# MAGIC From word to sentence embeddings
# MAGIC One possible way to calculate sentence embeddings from word embeddings is to take the average of the word embeddings.
# MAGIC This ignores word order and context, so two sentences with different meanings, but the same set of words will end up with the same sentence embedding.

# COMMAND ----------

in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."

# COMMAND ----------

# MAGIC %md
# MAGIC Remove stop words like ["the", "in", "for", "an", "is"] and punctuation.

# COMMAND ----------

in_pp_1 = ["kids", "play", "park"]
in_pp_2 = ["play", "kids", "park"]

# COMMAND ----------

# MAGIC %md
# MAGIC Generate one embedding for each word. So this is a list of three lists.

# COMMAND ----------

embeddings_1 = [emb.values for emb in embedding_model.get_embeddings(in_pp_1)]

# COMMAND ----------

# MAGIC %md
# MAGIC Use numpy to convert this list of lists into a 2D array of 3 rows and 768 columns.

# COMMAND ----------

import numpy as np
emb_array_1 = np.stack(embeddings_1)
print(emb_array_1.shape)

# COMMAND ----------

embeddings_2 = [emb.values for emb in embedding_model.get_embeddings(in_pp_2)]
emb_array_2 = np.stack(embeddings_2)
print(emb_array_2.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC - Take the average embedding across the 3 word embeddings 
# MAGIC - You'll get a single embedding of length 768.

# COMMAND ----------

emb_1_mean = emb_array_1.mean(axis = 0) 
print(emb_1_mean.shape)

# COMMAND ----------

emb_2_mean = emb_array_2.mean(axis = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC - Check to see that taking an average of word embeddings results in two sentence embeddings that are identical.

# COMMAND ----------

print(emb_1_mean[:4])
print(emb_2_mean[:4])

# COMMAND ----------

# MAGIC %md
# MAGIC Get sentence embeddings from the model.
# MAGIC These sentence embeddings account for word order and context.
# MAGIC Verify that the sentence embeddings are not the same.
# MAGIC

# COMMAND ----------

print(in_1)
print(in_2)

# COMMAND ----------

embedding_1 = embedding_model.get_embeddings([in_1])
embedding_2 = embedding_model.get_embeddings([in_2])

# COMMAND ----------

vector_1 = embedding_1[0].values
print(vector_1[:4])
vector_2 = embedding_2[0].values
print(vector_2[:4])

# COMMAND ----------

# MAGIC %md
# MAGIC I think, text embeddings are really fun to play with. Let's dive in, and 
# MAGIC take a look at some examples of text embeddings. So, 
# MAGIC here's my empty Jupyter notebook. If you're running this 
# MAGIC locally on your own computer, you need to 
# MAGIC have the Google Cloud, AI platform. installed. So, you run pip 
# MAGIC install Google Cloud AI platform. But I already 
# MAGIC have that on this computer, so I don't need to do 
# MAGIC that. Let me start by authenticating myself to the Google Cloud AI platform, 
# MAGIC I'm going to use this authenticate function, which is a helper 
# MAGIC function that I'm using just for this particular 
# MAGIC Jupyter Notebook environment to load my credentials and the 
# MAGIC project ID. If you want, you can print out these objects. 
# MAGIC  
# MAGIC So, project ID is just a string that specifies what project you're 
# MAGIC on. And then, I'm going to run my commands using a server in 
# MAGIC the region US Central, and then lastly, with this 
# MAGIC import Vertex AI, and I'm going to initialize this by specifying my 
# MAGIC project ID, the region where my API calls reserved as well 
# MAGIC as passing the credentials which contains the secret authentication 
# MAGIC key to authenticate me to the Vertex AI platform. 
# MAGIC If you set up your own accounts on 
# MAGIC Google Cloud, there are few steps needed to register 
# MAGIC an account, then set up and pull out the 
# MAGIC project ID, which becomes a string that you copy 
# MAGIC in here, and then you can select the region, US central one 
# MAGIC will work fine for many people, or you can 
# MAGIC choose service closer to wherever you are. And we also 
# MAGIC have an optional Jupyter notebook later that steps through 
# MAGIC in detail how to get your own credentials for the Google cloud 
# MAGIC platform. 
# MAGIC But for now, I would say, don't worry about this. This 
# MAGIC is all you need to get through this short course. And 
# MAGIC if you want to run this locally on your own machine. you 
# MAGIC can follow the instructions in that later optional Jupyter 
# MAGIC Notebook to figure out how to get your own credentials and project 
# MAGIC ID and so on. So, the main topic for this 
# MAGIC short course is to use text embedding models. So, I'm going 
# MAGIC to import the text embedding model like so. Next, I'm going 
# MAGIC to specify this particular text embedding Gecko-001 model, which 
# MAGIC is the model we'll use today. So, what this command does 
# MAGIC is it saves an embedding model here. 
# MAGIC And now, to actually compute an embedding, 
# MAGIC this is what you do. Set embedding equals, 
# MAGIC call the embedding model to get an embedding. Let's 
# MAGIC start off with this single word string life. Next, let's 
# MAGIC set vector equals embedding 0 dot values. This just 
# MAGIC extracts the values out of the embedding. And let's 
# MAGIC print the length of vector. And let's print 
# MAGIC the first 10 elements of the vector. So here, vector is a 768 
# MAGIC dimensional vector and these are the first 10 
# MAGIC elements of the embedding. Feel free to print 
# MAGIC out more elements of this if you want to take 
# MAGIC a look at all these numbers. 
# MAGIC So, what we just did was we took the single word life, 
# MAGIC really the text string with the word life 
# MAGIC and computed an embedding of it. Let's look at a different example. I 
# MAGIC can also pass in now a question. What is the meaning of life? and 
# MAGIC take this text string and compute that embedding. Once 
# MAGIC again, we end up with a 768 dimensional vector that computes 
# MAGIC 768 different sort of features for this sentence, and 
# MAGIC this is the first 10 elements. Because each of these embeddings is 
# MAGIC a lot of numbers, it's difficult to look at these 
# MAGIC numbers to understand what they mean. But it 
# MAGIC turns out one of the most useful applications of 
# MAGIC these embeddings is to try to decide how similar 
# MAGIC are two different sentences or two 
# MAGIC different phrases or two different paragraphs of text. 
# MAGIC So let's take a look at some more examples and 
# MAGIC how similar different embeddings are. For this, I'm going 
# MAGIC to use the scikit-learn packages, cosine similarity, measure similarity. What 
# MAGIC this does is basically take two 
# MAGIC vectors, and normalize them to have length one, and then compute 
# MAGIC their dot product. But this gives one way to measure 
# MAGIC how similar are two different 768-dimensional or really any 
# MAGIC other dimensional vectors. And I'm going to 
# MAGIC compute three embeddings. For the first sentence, I'm 
# MAGIC going to embed what is the meaning of life. Is 
# MAGIC it 42 or is it something else? And if you don't know the 
# MAGIC 42 reference, it's actually a reference to one of my 
# MAGIC favorite novels that you can search online for the number 42 if 
# MAGIC you're interested. 
# MAGIC  
# MAGIC But let's also embed how does one spend their time well on Earth, which, 
# MAGIC you know, seems a little bit like asking, what's 
# MAGIC the meaning of life? And then sentence three is, would you 
# MAGIC like a salad? I hope the meaning of my life is much more 
# MAGIC than eating salads. So hopefully, sentence three 
# MAGIC has maybe a little bit, but not too much to do 
# MAGIC with sentence one. And then, similar as above, let's just pull 
# MAGIC out the vectors of these embeddings. And now, let 
# MAGIC me compute and print out the similarity of, um, all three pairs 
# MAGIC of sentences. So, let me add this over here and rerun this. And now, 
# MAGIC we see that the similarity between Vec1 and Vec2, first two sentences 
# MAGIC is higher, 0.655. So, what is the meaning of life is judged by 
# MAGIC this embedding to be more similar, to how does one 
# MAGIC spend their time while on Earth. And the similarity between 
# MAGIC sentences 2 and 3 is 0.52, between 1 and 3 is 
# MAGIC 0.54. 
# MAGIC So, this accurately judges that the first two sentences 
# MAGIC are more similar in meaning than 1 and 3 or 2 and 3. 
# MAGIC And it accomplishes this even though there are 
# MAGIC no words in common between the first sentence 
# MAGIC and the second sentence. What I'd encourage you to do is pause this video 
# MAGIC and in the Jupyter Notebook on the left, go 
# MAGIC ahead and type in some other sentence. Maybe 
# MAGIC write some sentences about your favorite 
# MAGIC programming language or your favorite algorithm 
# MAGIC and maybe your favorite animals or 
# MAGIC your favorite weekend activities and plug in a few 
# MAGIC different sentences and see if it accurately judges whether 
# MAGIC different sentences are more or less similar to each 
# MAGIC other. I do want to point out one thing, which 
# MAGIC is that you might see that these numbers, they 
# MAGIC all look like they're in a pretty similar range. Cosine similarity 
# MAGIC in theory can go anywhere from 0 to 1. But it 
# MAGIC turns out that because these vectors are very 
# MAGIC high dimensional vectors, there are 768 dimensional vectors. It turns 
# MAGIC out that the cosine similarity values you get 
# MAGIC out will tend to fall within a relatively 
# MAGIC narrow range. You probably won't ever get 
# MAGIC 0 distance or 1.0 distance. But it turns out that even though these 
# MAGIC numbers feel like they may be in a relatively narrow range, the 
# MAGIC relative values between these distances are 
# MAGIC still very helpful. 
# MAGIC And again, if you plug in different sentences 
# MAGIC and play with this yourself, hopefully you get a better sense 
# MAGIC of what these similarity measures might be like. 
# MAGIC Let's take a deeper look at why sentence embeddings 
# MAGIC are more powerful, I think, than word embeddings. Let's look 
# MAGIC at another two different inputs. First input, the kids 
# MAGIC play in the park. You know, during recess, the kids 
# MAGIC play in the park. And in the second input is the play was for 
# MAGIC kids in the park. So, someone puts on a play that is 
# MAGIC a show for a bunch of kids to watch. If you 
# MAGIC were to remove what's called stop words, so stop words 
# MAGIC like the, in, for, and is, those are words that are often 
# MAGIC perceived to have less semantic meaning in English sometimes. 
# MAGIC But if you were to remove the stop words from both 
# MAGIC of these sentences, you really end up with an identical set 
# MAGIC of three words. Kids play park and play kids park. 
# MAGIC Now, let's compute the embedding of the words in the first inputs. I'm gonna 
# MAGIC do a little bit of data wrangling in a second. So, I'm gonna 
# MAGIC import the NumPy library. And then, let me use this code snippet to call 
# MAGIC the embedding model on the first input, kids play park. 
# MAGIC And then, the rest of this code here using an iterator and then 
# MAGIC NumPy stack, It's just a little bit of data wrangling 
# MAGIC to reformat the outputs of the embedding model 
# MAGIC into a 3 by 768 dimensional array. So, that just takes the 
# MAGIC embeddings and puts it in a, in an array like 
# MAGIC that. If you want, feel free to pause the video and print 
# MAGIC out the intermediate values to see what this 
# MAGIC is doing. But now, let me just do this as well for the 
# MAGIC second input. So, embedding array 2 is another 3 by 768 dimensional 
# MAGIC array. And there are three rows because there are three embeddings, 
# MAGIC one for each of these three words. 
# MAGIC So, one way that many people used to build 
# MAGIC sentence level embeddings is to, then take these 
# MAGIC three embeddings for the different words and to average 
# MAGIC them together. So, if I were to say the embedding for my first input, 
# MAGIC the kids play in the park after stop word removal. So, 
# MAGIC kids play park is, I'm going to take the embedding array one, and 
# MAGIC take the mean along x is zero. So that just averages it across 
# MAGIC the three words we have. And, you know, do 
# MAGIC the same for my second embedding. If I then print out 
# MAGIC the two embedding vectors, not surprisingly, you 
# MAGIC end up with the same value. So, because these two lists have 
# MAGIC exactly the same words, when you embed the words, and 
# MAGIC then average the embeddings of the individual words, you 
# MAGIC end up with the same values. Here I'm printing out just the first four 
# MAGIC elements of this array. 
# MAGIC You can feel free to check that, you know, all of these 768 elements 
# MAGIC of this array are identical. In contrast, if you were to 
# MAGIC call the embedding on the original input sentence 
# MAGIC like so then if you print out the values of the embeddings, 
# MAGIC you can see that they're then very different. And that's because 
# MAGIC the embedding model, um, in addition to not ignoring stop words, a common 
# MAGIC word like is, a, of, the, it also is much more sophisticated 
# MAGIC in understanding the word order so that it 
# MAGIC understands that the semantics or the meaning of 
# MAGIC the kids play in the park is very 
# MAGIC different than the play was for kids in the park. So, I 
# MAGIC do strongly encourage you to pause this video and go play 
# MAGIC with this yourself. Plug in different sentences, see what 
# MAGIC embeddings you get, look through the lines of code, make sure 
# MAGIC they make sense to you and play with 
# MAGIC these embeddings. 
# MAGIC So, before wrapping up this video, just to go over the key pieces of 
# MAGIC syntax we use in this course, you saw me use import Vertex AI. 
# MAGIC Vertex AI is the name of the Google Cloud Machine Learning platform, 
# MAGIC and then we use vertex in it, which require specifying 
# MAGIC the project ID, which references your 
# MAGIC Google Cloud project, the location where the service will 
# MAGIC run, so which data center will this run in, 
# MAGIC and then also the secret credentials for authentication. 
# MAGIC After setting up Vertex AI, this was the syntax 
# MAGIC to get an embedding. We will import the text embedding model, 
# MAGIC then specify the text embedding model, and load the model into embedding 
# MAGIC model, and then simply call get embeddings on a piece of text. 
# MAGIC With that, I hope you pause this video and go 
# MAGIC back to the Jupyter Notebook, and plug in other pieces of text. 
# MAGIC Write something fun or write something not fun if 
# MAGIC you insist, but plug in different pieces of text, 
# MAGIC and see what embeddings, and what results you get, 
# MAGIC and I hope you have fun with it. When you're done, let's go 
# MAGIC on to the next video where we'll dive into deeper conceptual understanding 
# MAGIC of what embeddings are and how they work. I 
# MAGIC look forward to seeing you in the next video. 
# MAGIC
