# Databricks notebook source
"""Welcome to Fine-Tuning Large Language Models, taught 
by Sharon Zhou. 
Really glad to be here. 
When I visit with different groups, I often hear people ask, 
how can I use these large language models on my 
own data or on my own task? 
Whereas you might already know about how to 
prompt a large language model, this course goes 
over another important tool, fine-tuning them. 
Specifically, how to take, say, an open-source 
LLM and further train it on your own data. 
While writing a prompt can be pretty good 
at getting an LLM to follow directions to 
carry out the task, like extracting keywords 
or classifying text as positive or negative sentiment. 
If you fine tune, you can then get the LLM to even 
more consistently do what you want. 
And I found that prompting an LLM to 
speak in a certain style, like being more helpful or more polite, 
or to be succinct versus verbose to a specific certain extent, 
that can also be challenging. 
Fine-tuning turns out to also be a good way to adjust an LLM's tone. 
People are now aware of the amazing capabilities of 
ChatGPT and other popular LLMs to answer questions about a huge range 
of topics. 
But individuals and companies would like to have that 
same interface to their own private and proprietary data. 
One of the ways to do this is to train 
an LLM with your data. 
Of course, training a foundation LLM takes 
a massive amount of data, maybe hundreds of billions 
or even more than a trillion words of data, 
and massive GPU compute resources. 
But with fine-tuning, you can take an existing 
LLM and train it further on your own data. 
So, in this course, you'll learn what fine-tuning is, when it 
might be helpful for your applications, how fine-tuning 
fits into training, how it differs from prompt engineering 
or retrieval augmented generation alone, and 
how these techniques can be used 
alongside fine-tuning. 
You'll dive into a specific variant of fine-tuning that's 
made GPT-3 into chat GPT called instruction fine-tuning, which teaches an LLM 
to follow instructions. 
Finally, you'll go through the steps of fine-tuning your 
own LLM, preparing the data, training the 
model, and evaluating it, all in code. 
This course is designed to be accessible to 
someone familiar with Python. 
But to understand all the code, it will help to further have 
basic knowledge of deep learning, such as what the 
process of training a neural network is like, and what is, 
say, a trained test split. 
A lot of hard work has gone into this course. 
We'd like to acknowledge the whole Lam and I team, and 
Nina Wei in particular on design, as well as on the DEEPLEARNING.AI side, 
Tommy Nelson and Geoff Ludwig. 
In about an hour or so through this short course, 
you gain a deeper understanding of how you can build 
your own LLM through fine-tuning an existing LLM on your own data. 
 
Let's get started. """
