# Databricks notebook source
"""
Compare finetuned vs. non-finetuned models
"""

# COMMAND ----------

import os
import lamini

lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")

# COMMAND ----------

from llama import BasicModelRunner

# COMMAND ----------

"""Try Non-Finetuned models"""

# COMMAND ----------

non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-hf")

# COMMAND ----------

non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")

# COMMAND ----------

print(non_finetuned_output)

# COMMAND ----------

print(non_finetuned("What do you think of Mars?"))

# COMMAND ----------

print(non_finetuned("taylor swift's best friend"))

# COMMAND ----------

print(non_finetuned("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))

# COMMAND ----------

"""Compare to finetuned models"""


# COMMAND ----------

finetuned_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")

# COMMAND ----------

finetuned_output = finetuned_model("Tell me how to train my dog to sit")

# COMMAND ----------

print(finetuned_output)

# COMMAND ----------

print(finetuned_model("[INST]Tell me how to train my dog to sit[/INST]"))

# COMMAND ----------

print(non_finetuned("[INST]Tell me how to train my dog to sit[/INST]"))

# COMMAND ----------

print(finetuned_model("What do you think of Mars?"))

# COMMAND ----------

print(finetuned_model("taylor swift's best friend"))

# COMMAND ----------

print(finetuned_model("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))

# COMMAND ----------

"""
Compare to ChatGPT
"""

# COMMAND ----------

chatgpt = BasicModelRunner("chat-gpt")

# COMMAND ----------

print(chatgpt("Tell me how to train my dog to sit"))

# COMMAND ----------

"""
In this lesson, you'll get to learn why you should fine-tune, what 
fine-tuning really even is, compare it to 
prompt engineering, and go through a lab where you get to 
compare a fine-tuned model to a non-fine-tuned model. 
 
Cool, let's get started! 
Alright, so why should you fine-tune LLMs? 
Well before we jump into why, let's talk about what fine-tuning really 
is. 
So what fine-tuning is, is taking these general purpose 
models like GPT-3 and specializing them into something 
like ChatGPT, 
the specific chat use case to make it chat well, or using GPT-4 
and turning that into a specialized GitHub co-pilot use 
case to auto-complete code. 
An analogy I like to make is a PCP, a primary care physician, 
is like your general purpose model. 
You go to your PCP every year for a general checkup, 
but a fine-tune or specialized model is like a cardiologist or dermatologist, 
a doctor that has a specific specialty and can actually 
take care of your heart problems or skin problems in much more 
depth. 
So what fine tuning actually does for your 
model is that it makes it possible for 
you to give it a lot more data than what fits into 
the prompt so that your model can learn 
from that data rather than just get access to it, 
from that learning process is able to upgrade itself from that PCP 
into something more specialized like a dermatologist. 
So you can see in this figure you might have some symptoms 
that you input into the model like skin irritation, 
redness, itching, and the base model 
which is the general purpose model might just 
say this is probably acne. 
A model that is fine-tuned on dermatology data however 
might take in the same symptoms and be 
able to give you a much clearer, more specific diagnosis. 
In addition to learning new information, fine-tuning can also help 
steer the model to more consistent outputs or more 
consistent behavior. 
For example, you can see the base model here. 
When you ask it, what's your first name? 
It might respond with, what's your last name? 
Because it's seen so much survey data out there of different questions. 
 
So it doesn't even know that it's supposed to answer that question. 
But a fine-tuned model by contrast, when you ask it, what's your 
first name? 
would be able to respond clearly. 
My first name is Sharon. 
This bot was probably trained on me. 
In addition to steering the model to more 
consistent outputs or behavior, fine tuning can help 
the model reduce hallucinations, which is a common problem 
where the model makes stuff up. 
Maybe it will say my first name is Bob when this was 
trained on my data and my name is definitely not Bob. 
Overall, fine tuning enables you to customize the model 
to a specific use case. 
In the fine-tuning process, which we'll go 
into far more detail later, it's actually very 
similar to the model's earlier training recipe. 
So now to compare it with something that you're 
probably a little bit more familiar with, which 
is prompt engineering. 
This is something that you've already been doing for a while 
with large language models, but maybe even for over the 
past decade with Google, which is just putting a query in, editing 
the query to change the results that you see. 
So there are a lot of pros to prompting. 
One is that you really don't need any data to get started. 
You can just start chatting with the model. 
There's a smaller upfront cost, so you don't really 
need to think about cost, since every single time you ping 
the model, it's not that expensive. 
And you don't really need technical knowledge to get started. 
You just need to know how to send a text message. 
What's cool is that there are now methods you can use, such 
as retrieval augmented generation, or RAG, to 
connect more of your data to it, to selectively choose what kind of data 
goes into the prompt. 
Now of course, if you have more than a little bit of data, 
then it might not fit into the prompt. 
So you can't use that much data. 
Oftentimes when you do try to fit in a ton of data, 
unfortunately it will forget a lot of that data. 
There are issues with hallucination, which is when the model 
does make stuff up and it's hard to correct that 
incorrect information that it's already learned. So while using retrieval augmented 
generation can be great to connect your data, it will 
also often miss the right data,get the incorrect data and cause the 
model, to output the wrong thing. 
Fine tuning is kind of the opposite of prompting. 
So you can actually fit in almost an 
unlimited amount of data, which is nice because 
the model gets to learn new information on that data. 
As a result, you can correct that incorrect information that it 
may have learned before, or even put in 
recent information that it hadn't learned about previously. 
There's less cost afterwards if you do fine-tune a 
smaller model and this is particularly relevant if 
you expect to hit the model a lot of times. So have 
a lot of either throughput or you expect 
it to just handle a larger load. 
And also retrieval augmented generation can 
be used here too. I think sometimes people think it's 
a separate thing but actually you can use 
it for both cases. 
So you can actually connect it with far more data 
as well even after it's learned all this information. 
There are cons, however. 
You need more data, and that data has to 
be higher quality to get started. 
There is an upfront compute cost as well, so it's 
not free necessarily. 
It's not just a couple dollars just to get started. 
Of course, there are now free tools out there to get started, 
but there is compute involved in making this happen, 
far more than just prompting. 
And oftentimes you need some technical 
knowledge to get the data in the right place, and that's 
especially, you know, surrounding this data piece. 
And, you know, there are more and more tools now that's 
making this far easier, but you still need some 
understanding of that data. 
And you don't have to be just anyone who can send 
a text message necessarily. 
So finally, what that means is for prompting, 
you know, that's great for generic use cases. 
It's great for different side projects and prototypes. 
It's great to just get started really, really 
fast. 
Meanwhile, fine tuning is great for more enterprise or domain-specific 
use cases, and for production usage. 
And we'll also talk about how it's useful for privacy in this 
next section, which is the benefits of fine-tuning your own 
LLM. So if you have your own LLM that 
you fine-tuned, one benefit you get is around performance. 
 
So this can stop the LLM from making stuff up, 
especially around your domain. 
It can have far more expertise in that domain. 
It can be far more consistent. 
So sometimes these models will just produce, you know, 
something really great today, but then tomorrow you hit it and it 
isn't consistent anymore. 
It's not giving you that great output anymore. 
And so this is one way to actually make it 
far more consistent and reliable. 
And you can also have it be better at moderating. If you've 
played a lot with ChatGPT, you might have seen ChatGPT say, I'm sorry, 
I can't respond to that. 
And you can actually get it to say the same 
thing or something different that's related to your company or 
use case to help the person chatting with it, stay 
on track. 
And again, so now I want to touch on privacy. 
When you fine tune your own LLM, this can happen in your VPC or on 
premise. 
This prevents data leakage and data breaches that 
might happen on off the shelf, third party solutions. 
And so this is one way to keep that data safe that you've 
been collecting for a while that might be the last few days, 
it might be the last couple decades as well. 
Another reason you might want to fine tune your own LLM is 
around cost, so one is just cost transparency. 
You maybe you have a lot of people using your model and 
you actually want to lower the cost per request. 
Then fine tuning a smaller LLM can actually 
help you do that. 
And overall, you have greater control 
over costs and a couple other factors as well. 
That includes uptime and also latency. 
You can greatly reduce the latency for certain applications 
like autocomplete. 
You might need latency that is sub 200 
milliseconds so that it is not perceivable by 
the person doing autocomplete. 
You probably don't want autocomplete to happen 
across 30 seconds, which is currently the case with 
running GPD 4 sometimes. 
And finally, in moderation, we talked about that 
a little bit here already. 
But basically, if you want the model to say, I'm 
sorry to certain things, or to say, I don't know 
to certain things, or even to have a custom response, This is 
one way to actually provide those guardrails to 
the model. 
And what's really cool is you're actually get to see an example of 
that in the notebooks. 
All right. 
So across all of these different labs, you'll be using a lot of 
different technologies to fine tune. 
So there are three Python libraries. 
One is PyTorch developed by Meta. 
This is the lowest level interface that you'll see. 
And then there's a great library by HuggingFace on top of PyTorch and 
a of the great work that's been done and it's much higher 
level. 
You can import datasets and train models very easily. 
And then finally, you'll see the Llamanai library, which 
I've been developing with my team. 
And we call it the llama library for 
all the great llamas out there. 
And this is an even higher level interface 
where you can train models with just three 
lines of code. 
All right. 
So let's hop over to the notebooks and see some fine-tuned models 
in action. 
Okay, so we're going to compare a fine-tuned model 
with a non-fine-tuned model. 
So first we're importing from the LLAMA library, again 
this is from LAMANI, the basic model runner. 
And all this class does is it helps us run open-source 
models. 
So these are hosted open-source models on GPUs to 
run them really efficiently. 
And the first model you can run here is the LLAMA2 model, 
which is very popular right now. 
And this one is not fine-tuned. 
So we're gonna just instantiate it based on this is its hugging 
face name and we're gonna say. 
Tell me how to train my dog to sit. 
So it's just you know, really really simple here into 
the non fine-tuned model we're gonna get the output out and. Let's print non 
tuned. 
Output and see oof. 
Okay. 
So we asked it. 
Tell me how to train my dog to sit. 
It said period, and then tell me how to train my dog to say, 
tell me how to teach my dog to come, tell me how to get my dog to heel. 
So clearly this is very similar to the what's 
your first name, what's your last name answer. 
This model has not been told or trained 
to actually respond to that command. 
So maybe a bit of a disaster, but let's keep looking. 
So maybe we can ask it, what do you think of Mars? 
So now, you know, at least it's responding to the question, but 
it's not great responses. 
I think it's a great planet. 
I think it's a good planet. 
I think it'll be a great planet. 
So it keeps going. 
Very philosophical, potentially even existential, 
if you keep reading. 
All right. 
What about something like a Google search query, 
like Taylor Swift's best friend? 
Let's see what that actually says. 
All right. 
Well, it doesn't quite get Taylor Swift's best 
friend, but it did say that it's a huge 
Taylor Swift fan. 
All right, let's keep exploring maybe something that's a 
conversation to see if it can do turns in a 
conversation like chat GPT. 
So this is an agent for an Amazon delivery order. 
Okay, so at least it's doing the different customer 
agent turns here, but it isn't quite 
getting anything out of it. This is not something usable for 
any kind of like fake turns or help 
with making an auto agent. 
All right, so you've seen enough of that. 
Let's actually compare this to Llama 2 that has been fine-tuned to 
actually chat. 
So I'm gonna instantiate the fine-tune model. 
Notice that this name, all that's different is this chat here. 
And then I'm gonna let this fine-tune model do the same thing. 
So tell me how to train my dog to sit. 
I'm gonna print that. 
Okay, very interesting. 
So you can immediately tell a difference. 
So tell me how to train my dog to sit. It's still trying to auto-complete that. 
 
So tell me how to train my dog to sit on command. 
But then it actually goes through almost a step-by-step guide 
of what to do to train my dog to sit. 
Cool, so that's much, much better. 
And the way to actually, quote unquote, 
get rid of this extra auto-complete thing is actually to 
inform the model that you want instructions. 
So I'm actually putting these instruction tags here. 
This was used for LLAMA2. 
You can use something different when you fine-tune 
your own model, but this helps with telling the model, hey, 
these are my instructions and these are the boundaries. 
I'm done with giving this instruction. 
Stop continuing to give me an instruction. 
So here you can see that it doesn't auto-complete that on-command thing. 
 
And just to compare, just to be fair, 
we can see what the non-fine-tuned model actually says. 
Great, it just repeats the same thing or something very similar. 
Not quite right. 
Cool, let's keep going down. So what do you think of Mars, this 
model? 
Oh, it's a fascinating planet. 
It's captured the imagination of humans for centuries. 
Okay, cool. 
So something that's much better output here. 
What about Taylor Swift's best friend? 
Let's see how this does. 
Okay, this one's pretty cute. 
It has a few candidates for who Taylor Swift's 
best friend actually is. 
Let's take a look at these turns from the Amazon delivery agent. 
Okay. 
It says, I see. Can you provide me with your order number? 
This is much, much better. 
It's interesting because down here, it also summarizes what's going on, 
which may or may not be something that you would want, 
and that would be something you can fine tune away. 
And now I'm curious what chat GPT would say for, tell 
me how to train my dog to sit. 
Okay. 
So it gives different steps as well. 
Great. 
Alright, feel free to use ChatGPT or any other 
model to see what else they can each 
do and compare the results. But it's pretty clear, I think, that 
the ones that have been fine-tuned, including ChatGPT and this Lama2Chat 
LLM, they're clearly better than the one that was 
not fine-tuned. 
Now in the next lesson, we're going to see where fine-tuning fits 
in in the whole training process. 
So you'll get to see the first step and how to even 
get here with this fine-tuned model. 

"""
