# Databricks notebook source
# MAGIC %md
# MAGIC All right, you made it to our last lesson and 
# MAGIC these will be some considerations you should take 
# MAGIC on getting started now, some practical tips, and also a bit 
# MAGIC of a sneak preview on more advanced training methods. 
# MAGIC So first, some practical steps to fine-tuning. 
# MAGIC Just to summarize, first you want to figure out your task, 
# MAGIC you want to collect data that's related to your tasks 
# MAGIC inputs and outputs and structure it as such. 
# MAGIC If you don't have enough data, no problem, just generate some or use 
# MAGIC a prompt template to create some more. 
# MAGIC And first, you want to fine tune a small model. 
# MAGIC I recommend a 400 million to a billion 
# MAGIC parameter model just to get a sense of 
# MAGIC where the performance is at with this model. 
# MAGIC And you should vary the amount of data you actually give to 
# MAGIC the model to understand how much data actually influences 
# MAGIC where the model is going. 
# MAGIC And then you can evaluate your model to see what's 
# MAGIC going well or not. 
# MAGIC And finally, you want to collect more data to improve 
# MAGIC the model through your evaluation. 
# MAGIC Now, from there, you can now increase your task complexity, 
# MAGIC so you can make it much harder now. 
# MAGIC And then you can also increase the model 
# MAGIC size for performance on that more complex task. 
# MAGIC So for task-defined tune, you learned about, you know, reading 
# MAGIC tasks and writing tasks. 
# MAGIC Writing tasks are a lot harder. 
# MAGIC These are the more expansive tasks like chatting, 
# MAGIC writing emails, writing code, and that's because there are more 
# MAGIC tokens that are produced by the model. 
# MAGIC So this is a harder task in general for the model. 
# MAGIC And harder tasks tend to result in needing 
# MAGIC larger models to be able to handle them. 
# MAGIC Another way of having a harder task is 
# MAGIC just having a combination of tasks, asking the model to 
# MAGIC do a combination of things instead of just one task. 
# MAGIC And that could mean having an agent be 
# MAGIC flexible and do several things at once or in just in one 
# MAGIC step as opposed to multiple steps. 
# MAGIC So now that you have a sense of model sizes that you 
# MAGIC need for your task complexity, there's also a compute requirement 
# MAGIC basically around hardware of what you need to run your models. 
# MAGIC  
# MAGIC For the labs that you ran, you saw those 70 million 
# MAGIC parameter models that ran on CPU. 
# MAGIC They weren't the best models out there. 
# MAGIC And I recommend starting with something a little 
# MAGIC bit more performant in general. 
# MAGIC So if you see here in this table, the first row, 
# MAGIC I want to call out of a "1 V100" GPU that's available, for example, on 
# MAGIC AWS, but also any other cloud platform and you see 
# MAGIC that it has 16 gigabytes of memory and 
# MAGIC that means it can run a 7 billion parameter model for inference 
# MAGIC but for training, training needs far more memory for to 
# MAGIC store the gradients and the optimizers so it only can actually fit 
# MAGIC a 1 billion parameter model and if you want to fit a 
# MAGIC larger model you can see some of the other options available here 
# MAGIC great so maybe you thought that that was not enough for you 
# MAGIC you want to work with much larger models? 
# MAGIC Well, there's something called PEFT or parameter efficient fine tuning, which 
# MAGIC is a set of different methods that help 
# MAGIC you do just that, be much more efficient in how you're using 
# MAGIC your parameters and training your models. 
# MAGIC And one that I really like is LoRa, which stands for low rank adaptation. 
# MAGIC  
# MAGIC And what LoRa does is that it reduces 
# MAGIC the number of parameters you have to train 
# MAGIC weights that you have to train by a huge amount. 
# MAGIC For GPT-3, for example, they found that they could 
# MAGIC reduce it by 10,000x, which resulted in 3x less memory needed 
# MAGIC from the GPU. 
# MAGIC And while you do get slightly below accuracy to fine 
# MAGIC tuning, this is still a far more efficient way 
# MAGIC of getting there and you get the same 
# MAGIC inference latency at the end. 
# MAGIC So what is exactly happening with LoRa? 
# MAGIC Well, you're actually training new weights in 
# MAGIC some of the layers of the model and you're freezing 
# MAGIC the main pre-trained weights, which you see 
# MAGIC here in blue. 
# MAGIC So that's all frozen and you have these new orange weights. 
# MAGIC Those are the LoRa weights. 
# MAGIC And the new weights, and this gets a little bit mathy, 
# MAGIC are the rank decomposition matrices of 
# MAGIC the original weights change. 
# MAGIC But what's important is less so, you know, the math behind that here, it's 
# MAGIC that you can train these separately, alternatively to 
# MAGIC the pre-trained weights, but then at 
# MAGIC inference time be able to merge them back into the 
# MAGIC main pre-trained weights and get that fine-tuned model more efficiently. 
# MAGIC  
# MAGIC What I'm really excited about to use LoRa for is adapting it to new 
# MAGIC tasks and so that means you could train 
# MAGIC a model with LoRa on one customer's data 
# MAGIC and then train another one on another customer's data and 
# MAGIC then be able to merge them each in at inference time when 
# MAGIC you need them. 
