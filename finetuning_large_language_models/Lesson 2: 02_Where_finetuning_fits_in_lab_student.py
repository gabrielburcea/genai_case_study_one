# Databricks notebook source
"""
Finetuning data: compare to pretraining and basic preparation
"""

# COMMAND ----------

import jsonlines
import itertools
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset

# COMMAND ----------

"""
Look at pretraining data set
Sorry, "The Pile" dataset is currently relocating to a new home and so we can't show you the same example that is in the video. Here is another dataset, the "Common Crawl" dataset.



"""

# COMMAND ----------

#pretrained_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)

pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)


# COMMAND ----------

n = 5
print("Pretrained dataset:")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)

# COMMAND ----------

"""Contrast with company finetuning dataset you will be using"""

filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df

# COMMAND ----------

"""Various ways of formatting your data"""
examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
text

# COMMAND ----------

if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

# COMMAND ----------

prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

# COMMAND ----------

question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
text_with_prompt_template

# COMMAND ----------

prompt_template_q = """### Question:
{question}

### Answer:"""

# COMMAND ----------

num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

# COMMAND ----------

pprint(finetuning_dataset_text_only[0])

# COMMAND ----------

pprint(finetuning_dataset_question_answer[0])

# COMMAND ----------

with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

# COMMAND ----------

finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)

# COMMAND ----------

"""
This might mean correcting old incorrect 
information so maybe there's you know more updated 
recent information that you want the model to 
actually be infused with. 
And of course more commonly you're doing both with these models, so 
oftentimes you're changing the behavior and you 
want it to gain new knowledge. 
So taking it a notch down, so tasks for fine-tuning, it's really 
just text in, text out for LLMs. And I 
like to think about it in two different categories, so 
you can think about it one as extracting text, so you 
put text in and you get less text out. So a 
lot of the work is in reading, and this could be extracting keywords, topics, it 
might be routing, based on all the data that you 
see coming in. You route the chat, for example, to some 
API or otherwise. 
Different agents are here, like different agent capabilities. 
And then that's in contrast to expansion. 
So that's where you put text in, and you get more text out. 
So I like to think of that as writing. 
And so that could be chatting, writing emails, writing code, 
and really understanding your task exactly, 
the difference between these two different tasks, 
or maybe you have multiple tasks that you want to fine-tune 
on is what I've found to be the clearest indicator of success. 
So if you want to succeed at fine-tuning the model, it's getting 
clearer on what task you want to do. 
And clarity really means knowing what 
good output looks like, what bad output looks like, 
but also what better output looks like. 
So when you know that something is doing 
better at writing code or doing better at routing a task, 
that actually does help you actually fine-tune 
this model to do really well. 
Alright, so if this is your first time fine-tuning, 
I recommend a few different steps. 
So first, identify a task by just prompt engineering a 
large LLM and that could be chat GPT, for example, 
and so you're just playing with chat GPT 
like you normally do. 
And you find some, you know, tasks that it's doing okay at, so 
not not great, but like not horrible either, so 
you know that it's possible within the realm of possibility, but 
it's not it's not the best and you want it to much better 
for your task. 
So pick that one task and just pick one. 
And then number four, get some inputs and 
outputs for that task. So you put in some text 
and you got some text out, get inputs where you 
put in text and get text out and outputs, 
pairs of those for this task. 
And one of the golden numbers I like to use 
is 1000 because I found that that is a good 
starting point for the amount of data that you need. 
And make sure that these inputs and outputs 
are better than the okay result from that LLM before. 
You can't just generate these outputs necessarily all the time. 
And so make sure you have those pairs of data and you'll 
explore this in the lab too, this whole pipeline here. 
Start out with that and then what you do is 
you can then fine tune a small LLM on this 
data just to get a sense of that performance bump. 
And then so this is only if you're a first time, this 
is what I recommend. 
So now let's jump into the lab where you 
get to explore the data set that was used for pre-training versus 
for fine-tuning, so you understand exactly what these 
input- output pairs look like. 
Okay, so we're going to get started by importing a few different 
libraries, so we're just going to run that. 
And the first library that we're going to use is 
the datasets library from. HuggingFace, and they have this great 
function called loadDataset where you can just pull 
a dataset from on their hub and be able to run it. 
So here I'm going to pull the pre-training dataset called the pile that 
you just saw a little bit more about and here I'm just grabbing 
the split which is train versus test and 
very specifically I'm actually grabbing streaming equals true because 
this data set is massive we can't download it 
without breaking this new book so I'm 
actually going to stream it in one at a 
time so that we can explore the different pieces of data in 
there. 
So just loading that up and now I'm going to just look at 
the first five so this. 
It's just using iter tools. 
Great. 
Ok, so you can see here, in the pre-trained data set, there's a 
lot of data here that looks kind of scraped. 
So this text says, it is done and submitted. 
You can play Survival of the Tastiest on Android. 
And so that's one example. 
And let's see if we can find another one here. 
Here is another one. 
So this is just code that was scraped, XML code that was scraped. So that's 
another data point. 
You'll see article content, you'll see this topic about Amazon 
announcing a new service on AWS, and then here's about Grand 
Slam Fishing Charter, which is a family business. 
So this is just a hodgepodge of different data sets scraped from 
essentially the internet. 
And I kind of want to contrast that with fine-tuning 
data set that you'll be using across the different labs. 
We're grabbing a company data set of question-answer pairs, you know, 
scraped from an FAQ and also put together about internal engineering 
documentation. 
And it's called Lamini Docs, it's about the company Lamini. 
And so we're just going to read that JSON file and take a look at 
what's in there. 
Okay, so this is much more structured data, 
right? So there are question-answer pairs here, and it's very 
specific about this company. 
So the simplest way to use this data 
set is to concatenate actually these questions and 
answers together and serve that up into the model. 
So that's what I'm going to do here. I'm going to turn that into a dict 
and then I'm going to see what actually concatenating one 
of these looks like. 
So, you know, just concatenating the question and directly 
just giving the answer after it right here. 
And of course you can prepare your data in any way possible. 
I just want to call out a few different common ways of 
formatting your data and structuring it. 
So question answer pairs, but then also 
instruction and response pairs, input output pairs, 
just being very generic here. 
And also, you can actually just have it, 
since we're concatenating it anyways, it's just 
text that you saw above with the pile. 
All right, so concatenating it, that's very simple, but 
sometimes that is enough to see results, sometimes 
it isn't. 
So you'll still find that the model might need just more structure 
to help with it, and this is very similar 
to prompt engineering, actually. 
So taking things a bit further, you can also process your data 
with an instruction following, in this case, question-answering 
prompt template. 
And here's a common template. 
Note that there's a pound-pound-pound before the question type of marker 
so that that can be easily used as structure to tell the 
model to expect what's next. 
It expects a question after it sees that for the question. 
And it also can help you post-process the model's outputs 
even after it's been fine-tuned. 
So we have that there. 
So let's take a look at this prompt template in 
action and see how that differs from the 
concatenated question and answer. 
So here you can see how that's how the prompt template 
is with the question and answer neatly done 
there. 
And often it helps to keep the input 
and output separate so I'm actually going to take 
out that answer here and keep them separated 
out because this helps us just using the 
data set easily for evaluation and for you 
know when you split the data set into 
train and test. 
So now what I'm gonna do is put all of this, apply 
all of this template to the entire data set. 
So just running a for loop over it and just hydrating the prompt. 
So that is just adding that question and 
answer into this with F string or dot 
format stuff here with Python. 
All right, so let's take a look at the difference between that text-only thing 
and the question-answer format. 
Cool. 
So it's just text-only, it's all concatenated here that you're putting in, and 
here is just question-answer, much more structured. 
 
And you can use either one, but of course I 
do recommend structuring it to help with evaluation. 
That is basically it. 
The most common way of storing this data 
is usually in JSON lines files, so "jsonl files.jsonl." 
It's basically just, you know, each line is a JSON object and that's it, 
and so just writing that to file there. 
You can also upload this data set onto HuggingFace, 
shown here, because you'll get to use this later 
as well and you'll get to pull it from the 
cloud like that. 
Next, you'll dive into a specific variant of fine-tuning called 
instruction fine-tuning. 

"""
