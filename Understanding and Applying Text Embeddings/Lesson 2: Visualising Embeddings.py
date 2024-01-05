# Databricks notebook source
# MAGIC %md
# MAGIC Project environment setup
# MAGIC Load credentials and relevant Python Libraries

# COMMAND ----------

from utils import authenticate
credentials, PROJECT_ID = authenticate() #Get credentials and project ID

# COMMAND ----------

REGION = 'us-central1'

# COMMAND ----------

# MAGIC %md
# MAGIC Enter project details

# COMMAND ----------

# Import and initialize the Vertex AI Python SDK

import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)

# COMMAND ----------

# MAGIC %md
# MAGIC Embeddings capture meaning
# MAGIC

# COMMAND ----------

in_1 = "Missing flamingo discovered at swimming pool"

in_2 = "Sea otter spotted on surfboard by beach"

in_3 = "Baby panda enjoys boat ride"


in_4 = "Breakfast themed food truck beloved by all!"

in_5 = "New curry restaurant aims to please!"


in_6 = "Python developers are wonderful people"

in_7 = "TypeScript, C++ or Java? All are great!" 


input_text_lst_news = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]

# COMMAND ----------

import numpy as np
from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")

# COMMAND ----------

- Get embeddings for all pieces of text.
- Store them in a 2D NumPy array (one row for each embedding).

# COMMAND ----------

embeddings = []
for input_text in input_text_lst_news:
    emb = embedding_model.get_embeddings(
        [input_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings) 

# COMMAND ----------

print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reduce embeddings from 768 to 2 dimensions for visualization
# MAGIC - We'll use principal component analysis (PCA).
# MAGIC - You can learn more about PCA in [this video](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/73zWO/reducing-the-number-of-features-optional) from the Machine Learning Specialization. 

# COMMAND ----------

from sklearn.decomposition import PCA

# Perform PCA for 2D visualization
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings_array)
new_values = PCA_model.transform(embeddings_array)

# COMMAND ----------

print("Shape: " + str(new_values.shape))
print(new_values)

# COMMAND ----------

# MAGIC %md
# MAGIC Embeddings and Similarity
# MAGIC Plot a heat map to compare the embeddings of sentences that are similar and sentences that are dissimilar.

# COMMAND ----------

in_1 = """He couldn’t desert 
          his post at the power plant."""

in_2 = """The power plant needed 
          him at the time."""

in_3 = """Cacti are able to 
          withstand dry environments.""" 

in_4 = """Desert plants can 
          survive droughts.""" 

input_text_lst_sim = [in_1, in_2, in_3, in_4]

# COMMAND ----------

embeddings = []
for input_text in input_text_lst_sim:
    emb = embedding_model.get_embeddings([input_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings) 

# COMMAND ----------

from utils import plot_heatmap

y_labels = input_text_lst_sim

# Plot the heatmap
plot_heatmap(embeddings_array, y_labels = y_labels, title = "Embeddings Heatmap")

# COMMAND ----------

# MAGIC %md
# MAGIC Note: the heat map won't show everything because there are 768 columns to show.  To adjust the heat map with your mouse:
# MAGIC - Hover your mouse over the heat map.  Buttons will appear on the left of the heatmap.  Click on the button that has a vertical and horizontal double arrow (they look like axes).
# MAGIC - Left click and drag to move the heat map left and right.
# MAGIC - Right click and drag up to zoom in.
# MAGIC - Right click and drag down to zoom out.
# MAGIC
# MAGIC #### Compute cosine similarity
# MAGIC - The `cosine_similarity` function expects a 2D array, which is why we'll wrap each embedding list inside another list.
# MAGIC - You can verify that sentence 1 and 2 have a higher similarity compared to sentence 1 and 4, even though sentence 1 and 4 both have the words "desert" and "plant".

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

def compare(embeddings,idx1,idx2):
    return cosine_similarity([embeddings[idx1]],[embeddings[idx2]])

# COMMAND ----------

print(in_1)
print(in_2)
print(compare(embeddings,0,1))

# COMMAND ----------

print(in_1)
print(in_4)
print(compare(embeddings,0,3))

# COMMAND ----------

# MAGIC %md
# MAGIC Type Markdown and LaTeX:  𝛼2

# COMMAND ----------

# MAGIC %md
# MAGIC In this video, we'll take a look at some visualizations of embeddings. 
# MAGIC When you're building a practical application, you often 
# MAGIC not be outputting the visualization as the final step, unless you're, 
# MAGIC say, taking a collection of documents and 
# MAGIC deliberately want to visualize what different 
# MAGIC texts in the documents are saying. So what documents 
# MAGIC are similar, what documents are dissimilar. But outside applications 
# MAGIC like that, I don't end up using 
# MAGIC visualizations that much, but in this video, we'll take a look at 
# MAGIC some of them to build intuition about what these embeddings are actually 
# MAGIC doing. I think we'll get some pretty parts out 
# MAGIC of this. So, let's take a look. 
# MAGIC Let's start off by authenticating myself to 
# MAGIC the Veritex AI platform, same as before. For 
# MAGIC this visualization, I want to use this 
# MAGIC collection of seven sentences. Mustang flamingo, discover 
# MAGIC that swimming pool, see all the spotters who 
# MAGIC have bought baby panda, boat ride, breakfast theme, food 
# MAGIC truck, new curry restaurants, and then two others. Python 
# MAGIC developers are wonderful people. I think that's a 
# MAGIC totally true statement. And then TypeScript, C++, and Java 
# MAGIC all are great. I have to admit, I have 
# MAGIC my own preferences, but I won't state them 
# MAGIC in this video. And then, similar to last time, let me import 
# MAGIC NumPy, and I'm going to set up my embedding model as 
# MAGIC follows, and then for my seven sentences in these 
# MAGIC input one through input seven, in one through in 
# MAGIC seven, let's run this snippet of code to loop over, you know, my seven inputs 
# MAGIC to compute embeddings for all of them. 
# MAGIC  
# MAGIC So, for input text in this list of sentences, 
# MAGIC set the embedding to, you know, call the embedding model on that, 
# MAGIC extract the values, and then stick it 
# MAGIC in this embeddings list, and. And then a little bit 
# MAGIC of data munging to turn it into a NumPy array. And so, let's 
# MAGIC run it. Returns really quickly. And so right now, this 
# MAGIC embeddings array, let me print out the shape 
# MAGIC of the embeddings array. So, the shape is 7 by 768. And 
# MAGIC there are seven rows because there were seven 
# MAGIC sentences that we embedded, and each row has 768 numbers. Now, what 
# MAGIC I'd like to do is to next visualize these seven embeddings. But 
# MAGIC I can't plot a 768-dimensional vector on this 2D computer monitor. 
# MAGIC So, I'm going to use a technique called PCA, or Principal Components Analysis. If 
# MAGIC you know what PCA is, great, if you don't, don't worry 
# MAGIC about it, is a technique for taking very high-dimensional data, say 768-dimensional data, and compressing that 
# MAGIC down to two dimensions. 
# MAGIC  
# MAGIC If you're interested to learn more about PCA, you 
# MAGIC can take an online machine learning class such as the Machine Learning 
# MAGIC Specialization. But all you really need to know for the 
# MAGIC purpose of this video is, this is a way to take this 768-dimensional data 
# MAGIC and squash it down to two dimensions. So, this code 
# MAGIC calls the Scikit-learn PCA library, It fits a 
# MAGIC PC model to compress it to two-dimensional data, and you 
# MAGIC get a set of new values. If I now print 
# MAGIC out the new values, it's now 7 by 2 
# MAGIC instead of 7 by 768. It's compressed. Each of these embedding vectors down 
# MAGIC to two dimensions. Thus, losing a lot of information 
# MAGIC along the way, but make it easier to plot 
# MAGIC on this computer monitor. 
# MAGIC Next, here's the code we can use to plot. I'm going 
# MAGIC to use matplotlib, a common plotting function, and plot 
# MAGIC a 2D plot of the first and second values of 
# MAGIC each of these now two-dimensional embeddings on the horizontal and 
# MAGIC vertical axes of a figure. Let me do that, and here's our plot. Let's 
# MAGIC see. Here is baby panda, mussel, flamingo, sea otter. So, our 
# MAGIC animal-related sentences are pretty close to 
# MAGIC each other. Here is my food truck and the curry restaurant, pretty 
# MAGIC close to each other. Python developers are wonderful people. I 
# MAGIC mean, that's just true, and you know, all great programming languages. 
# MAGIC So, the coding-related sentences are down here. So, 
# MAGIC this illustrates how the algorithm does embed sentences 
# MAGIC that are more similar together. 
# MAGIC  
# MAGIC Strongly encourage you to pause the video, 
# MAGIC go back and plug in your own sentences. Maybe write some 
# MAGIC fun sentences about your friends and send the resulting 
# MAGIC visualization to them. It seems like there's 
# MAGIC a lot you could do to play with this. Just to be clear, if you're writing 
# MAGIC an application, I would not use a two-dimensional embedding. I 
# MAGIC would always measure distances in that original 768 
# MAGIC dimensional, uh, feature embedding. And I'm reducing this to 
# MAGIC two dimensions just for visualization purposes. But when measuring similarity, 
# MAGIC I would not measure similarity of the 2D 
# MAGIC data. I would measure similarity in that original, much higher dimensional 
# MAGIC space, which actually gives more 
# MAGIC accurate distance metrics. Because PCA, principal components 
# MAGIC analysis, is throwing away quite 
# MAGIC a lot of information to generate the visualization. 
# MAGIC  
# MAGIC Now, let's look at another example. Input 1, N1, he couldn't desert 
# MAGIC his post at the power plant, the power plant needed him at the time. So, N1 
# MAGIC and N2 seem pretty similar, N plus 1 and N plus 2. Cacti are 
# MAGIC able to withstand dry environments and desert plants can 
# MAGIC survive droughts. So, N3 and N4 seem pretty similar, even though 
# MAGIC sentence one and sentence four, both of 
# MAGIC the word desert and plant versus plants. But here's 
# MAGIC my list of four sentences. Then, here's my same 
# MAGIC code snippet as before to generate the embeddings. 
# MAGIC  
# MAGIC Lastly, let's plot a heat map showing the values of the embedding. Here's 
# MAGIC some code to plot a heat map. And what 
# MAGIC this is showing is, is using these different colors from 
# MAGIC shades of blue through shades of red to 
# MAGIC show for each of the four embeddings, does the 
# MAGIC first component have a high value or low value? Does the 
# MAGIC second component have high value or low value? And so 
# MAGIC on through many elements of this vector. I don't normally use 
# MAGIC heatmaps to visualize embeddings. I'll say a little 
# MAGIC bit more about this in a second. But from this heatmap, 
# MAGIC you know, hopefully you can see that the desert plant embedding and 
# MAGIC the cacti embedding in this very high dimensional space, 
# MAGIC the first two patterns in the heat map 
# MAGIC look a bit more similar to each other and the patterns on 
# MAGIC the third and fourth heat maps, you know, kind of 
# MAGIC look a little bit more similar to each other. And so, this illustrates 
# MAGIC that the embedding also maps the first two sentences close to 
# MAGIC each other than sentences three and four. And just as one 
# MAGIC fun exercise, I encourage you to pause this video and if you 
# MAGIC want to try, get the code from the earlier notebook to actually compute 
# MAGIC the similarity between these embeddings to 
# MAGIC see if the embeddings between the first two inputs 
# MAGIC are really more similar to each other than say the first and 
# MAGIC third or the first and the fourth. 
# MAGIC If you want to do that, just as a reminder, the function we 
# MAGIC used to compute similarities back in 
# MAGIC the first notebook was the cosine similarity rarity function. So, 
# MAGIC feel free to do that. If you want to see if you can 
# MAGIC compute these pairwise similarities. Before I wrap up this video, 
# MAGIC just one caveat, which is that even though I showed this visualization, 
# MAGIC this particular visualization isn't a completely mathematically legitimate 
# MAGIC thing to do. And to get a little bit technical 
# MAGIC for the next 30 seconds, so if you don't follow what I say for the 
# MAGIC next 30 seconds, don't worry about it. But it turns out that the axes 
# MAGIC used to define an embedding is relatively arbitrary 
# MAGIC and is subject to random rotation. And that's why, 
# MAGIC whereas finding pairwise similarity between embeddings is 
# MAGIC a pretty robust and sound operation, if you were 
# MAGIC to look at, say, the first component of an embedding and visualize 
# MAGIC that by looking at the number of heatmap, 
# MAGIC that first number is very difficult to ascribe a specific 
# MAGIC meaning to it which is why we can look at this as an informal visualization 
# MAGIC to build intuition about embeddings, but 
# MAGIC for practical things, I find it less 
# MAGIC useful to look at the outputs of the embeddings a single component 
# MAGIC at a time. If you don't understand what I just 
# MAGIC said, don't worry about it. The key takeaway is that this 
# MAGIC is just something of an informal visualization, and 
# MAGIC these particular visualizations are not 
# MAGIC very robust, and they could easily change if 
# MAGIC you were to use a different embedding model, 
# MAGIC say, but hopefully the intuition that certain sentences are more 
# MAGIC similar to each other in the embedding space still 
# MAGIC comes through. So that's it. 
# MAGIC Once again, please pause the video, write in your own sentences, 
# MAGIC maybe play the cosine similarity, use all this to 
# MAGIC build intuition about what the embeddings are doing. 
# MAGIC In the next video, Nikita will start to 
# MAGIC dive into how you can use these embeddings 
# MAGIC to build a variety of applications. When you're ready, please 
# MAGIC go on to the next video. 
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


