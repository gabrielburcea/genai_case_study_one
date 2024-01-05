# Databricks notebook source
# MAGIC %md
# MAGIC Hi and welcome to this short course, Understanding and Applying Text 
# MAGIC Embeddings with Vertex AI, built in partnership with Google Cloud. 
# MAGIC In this course, you'll learn about different properties and applications 
# MAGIC of text embeddings. We'll dive together into how to 
# MAGIC compute embeddings, that is feature vector representations 
# MAGIC of text sequences of arbitrary length, and we'll 
# MAGIC see how these sentence embeddings are a powerful 
# MAGIC tool for many applications like classification, outlier detection, and 
# MAGIC text clustering. If you've heard of word 
# MAGIC embedding algorithms like Word2Vec or GloVe, 
# MAGIC that just examine a single word at a 
# MAGIC time, this is a bit like that, but much more powerful, 
# MAGIC and much more general because it operates at 
# MAGIC the level of the meaning of a sentence 
# MAGIC or even a paragraph of text, and also works for 
# MAGIC sentences that contain words not seen in the 
# MAGIC training set. 
# MAGIC In this course, you'll also learn how to combine text 
# MAGIC generation capabilities of large language models with these sentence 
# MAGIC level embeddings and build a small-scale question 
# MAGIC answering system that answers questions about 
# MAGIC Python based on the database of Stack Overflow posts. I'd like to 
# MAGIC introduce the other instructor for this course, Nikita Namjushi. 
# MAGIC  
# MAGIC Thanks, Andrew. I'm so excited to be teaching 
# MAGIC this course with you. As part of my job at Google Cloud AI, I 
# MAGIC help developers build with large language models and I'm really 
# MAGIC looking forward to sharing practical tips that I've learned from 
# MAGIC working with many cloud customers and many many LLM applications. 
# MAGIC  
# MAGIC This course will consist of the following topics. 
# MAGIC In the first half, which I'll present, we'll first use an embeddings model 
# MAGIC to create and explore some text embeddings. Then we'll 
# MAGIC look together to go through a conceptual understanding 
# MAGIC of how these embeddings work and how embeddings for 
# MAGIC text sequences of arbitrary length are 
# MAGIC created and also use code to visualize different properties 
# MAGIC of embeddings. The second half 
# MAGIC is taught by Nikita. 
# MAGIC Well, after you've had a chance to explore some 
# MAGIC different properties of embeddings, you'll then see how to use 
# MAGIC them for classification, clustering, and outlier detection. Because sentence 
# MAGIC level embeddings start to get at the meaning of 
# MAGIC an entire sentence, this really helps an algorithm to reason more 
# MAGIC deeply and make better decisions about text. So, after this, 
# MAGIC we'll see how to use a text generation model and some 
# MAGIC of the different parameters you can adjust. And 
# MAGIC finally, we'll put everything you've learned about embeddings, semantic 
# MAGIC similarity, and text generation together to 
# MAGIC build a small-scale question-answering system. 
# MAGIC  
# MAGIC Many people have contributed to this course. We're 
# MAGIC grateful for Eva Liu and Carl Tanner from 
# MAGIC the Google Cloud team, and also on the DeepLearning.ai side, 
# MAGIC Daniel Vigilagra and Eddie Hsu. 
# MAGIC The first lesson will be about how to 
# MAGIC get started with embedding text. 
# MAGIC That sounds great. Let's get started. 
