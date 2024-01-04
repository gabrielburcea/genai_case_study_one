# Databricks notebook source
"""
Training

"""

# COMMAND ----------

"""Technically, it's only a few lines of code to run on GPUs (elsewhere, ie. on Lamini).
from llama import BasicModelRunner

model = BasicModelRunner("EleutherAI/pythia-410m") 
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True) 

Choose base model.
Load data.
Train it. Returns a model ID, dashboard, and playground interface.
Let's look under the hood at the core code running this! This is the open core of Lamini's llama library :)"""


# COMMAND ----------

import os
import lamini

lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML

# COMMAND ----------

import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines

from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner



logger = logging.getLogger(__name__)
global_config = None

# COMMAND ----------

"""
Load the Lamini docs dataset
"""

# COMMAND ----------

dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = False

# COMMAND ----------

dataset_path = "lamini/lamini_docs"
use_hf = True

# COMMAND ----------

"""
Set up the model, training config, and tokenizer
"""

# COMMAND ----------

model_name = "EleutherAI/pythia-70m"

# COMMAND ----------

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

print(train_dataset)
print(test_dataset)

# COMMAND ----------

"""Load the base model"""

# COMMAND ----------

base_model = AutoModelForCausalLM.from_pretrained(model_name)

# COMMAND ----------

device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

# COMMAND ----------

base_model.to(device)

# COMMAND ----------

"""Define function to carry out inference"""

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

# COMMAND ----------

"""
Try the base model
"""


test_text = test_dataset[0]['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

# COMMAND ----------

"""
Setup training

"""

# COMMAND ----------

max_steps = 3

# COMMAND ----------

trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name

# COMMAND ----------

training_args = TrainingArguments(

  # Learning rate
  learning_rate=1.0e-5,

  # Number of training epochs
  num_train_epochs=1,

  # Max steps to train for (each step is a batch of data)
  # Overrides num_train_epochs, if not -1
  max_steps=max_steps,

  # Batch size for training
  per_device_train_batch_size=1,

  # Directory to save model checkpoints
  output_dir=output_dir,

  # Other arguments
  overwrite_output_dir=False, # Overwrite the content of the output directory
  disable_tqdm=False, # Disable progress bars
  eval_steps=120, # Number of update steps between two evaluations
  save_steps=120, # After # steps model is saved
  warmup_steps=1, # Number of warmup steps for learning rate scheduler
  per_device_eval_batch_size=1, # Batch size for evaluation
  evaluation_strategy="steps",
  logging_strategy="steps",
  logging_steps=1,
  optim="adafactor",
  gradient_accumulation_steps = 4,
  gradient_checkpointing=False,

  # Parameters for early stopping
  load_best_model_at_end=True,
  save_total_limit=1,
  metric_for_best_model="eval_loss",
  greater_is_better=False
)

# COMMAND ----------

model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

# COMMAND ----------

trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# COMMAND ----------

### Train a few steps

training_output = trainer.train()

# COMMAND ----------

"""Save model locally"""

save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)

# COMMAND ----------

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)

# COMMAND ----------

finetuned_slightly_model.to(device) 

# COMMAND ----------

"""Run slightly trained model"""

test_question = test_dataset[0]['question']
print("Question input (test):", test_question)

print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))


# COMMAND ----------

test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer)

# COMMAND ----------

"""
Run same model trained for two epochs
"""

finetuned_longer_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")

finetuned_longer_model.to(device)
print("Finetuned longer model's answer: ")
print(inference(test_question, finetuned_longer_model, tokenizer))

# COMMAND ----------

"""
Run much larger trained model and explore moderation
"""

bigger_finetuned_model = BasicModelRunner(model_name_to_id["bigger_model_name"])
bigger_finetuned_output = bigger_finetuned_model(test_question)
print("Bigger (2.8B) finetuned model (test): ", bigger_finetuned_output)

# COMMAND ----------

count = 0
for i in range(len(train_dataset)):
 if "keep the discussion relevant to Lamini" in train_dataset[i]["answer"]:
  print(i, train_dataset[i]["question"], train_dataset[i]["answer"])
  count += 1
print(count)

# COMMAND ----------

"""Explore moderation using small model
First, try the non-finetuned base model:"""

base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
print(inference("What do you think of Mars?", base_model, base_tokenizer))

# COMMAND ----------

"""
Now try moderation with finetuned small model
"""

print(inference("What do you think of Mars?", finetuned_longer_model, tokenizer))

# COMMAND ----------

"""
Finetune a model in 3 lines of code using Lamini
"""

model = BasicModelRunner("EleutherAI/pythia-410m") 
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True) 

# COMMAND ----------

out = model.evaluate()

# COMMAND ----------

lofd = []
for e in out['eval_results']:
    q  = f"{e['input']}"
    at = f"{e['outputs'][0]['output']}"
    ab = f"{e['outputs'][1]['output']}"
    di = {'question': q, 'trained model': at, 'Base Model' : ab}
    lofd.append(di)
df = pd.DataFrame.from_dict(lofd)
style_df = df.style.set_properties(**{'text-align': 'left'})
style_df = style_df.set_properties(**{"vertical-align": "text-top"})
style_df

# COMMAND ----------

"""

In this lesson, you'll step through the entire training process, and 
at the end see the model improve on your task, specifically 
for you to be able to chat with it. 
Alright, let's jump into it. 
Alright, training in LLM, what does this look like? 
So, the training process is actually quite similar to other 
neural networks. 
So, as you can see here, you know, the same setup that we 
had seen the LLM predict "sd!!@". 
What's going on? 
Well, first you add that training data up at the top. 
Then you calculate the loss, so it predicts something 
totally off in the beginning, predict the loss compared 
to the actual response it was supposed to give, that's 
a pawn. 
And then you update the weights, you back prop through 
the model to update the model to improve it, 
such that in the end it does learn to then 
output something like a pawn. 
There are a lot of different hyperparameters that 
go into training LLMs. 
We won't go through them very specifically, but 
across a few that you might want to play with is learning 
rate, learning scheduler, and various optimizer hyperparameters 
as well. 
All right, so now diving a level deeper into the code. 
So these are just general chunks of training 
process code in PyTorch. 
So first you want to go over the number of epochs, 
an epoch is a pass over your entire data set. 
So you might go over your entire data set multiple times. 
And then you want to load it up in batches. 
So that is those different batches that you saw when you're 
tokenizing data. 
So that's sets of data together. 
And then you put the batch through your 
model to get outputs. 
You compute the loss from your model and 
you take a backwards step and you update your optimizer. 
Okay. So now that you've gone through every step 
of this low level code in PyTorch, we're actually 
going to go one level higher into HuggingFace and 
also another level higher into the Llama library by 
Llama and I, just to see how the training 
process works in practice in the lab. 
So let's take a look at that. 
Okay. 
So first up is seeing how the training 
process has been simplified over time, quite a bit with 
higher and higher level interfaces, that PyTorch code you saw. 
Man, I remember running that during my PhD. 
Now there are so many great libraries out 
there to make this very easy. 
One of them is the Lamini Llama library, and it's just training your model in 
three lines of code that's hosted on an external GPU, and it 
can run any open source model, and you can get the model back. 
And as you can see here, it's just requesting that 410 
million parameter model. 
You can load that data from that same JSON lines file, 
and then you just hit "model.train". 
And that returns a dashboard, a playground interface, 
and a model ID that you can then call and continue training 
or run with for inference. 
All right, so for the rest of this lab, we're 
actually going to focus on using the Pythia 
70 million model. 
You might be wondering why we've been playing with that really small, tiny 
model, and the reason is that it can run on CPU nicely 
here for this lab, so that you can actually see 
the whole training process go. 
But realistically, for your actual use cases, 
I recommend starting with something a bit larger, 
maybe something around a billion parameters, 
or maybe even this 400 million one if your task 
is on the easier side. 
Cool. 
So first up, I'm going to load up all of these libraries, And 
one of them is a utilities file with a bunch of different 
functions in there. 
Some of them that we've already written together on the tokenizer, 
and others you should take a look at for just logging and showing 
outputs. 
So first let's start with the different configuration parameters 
for training. 
So there are two ways to actually, you know, import data. You've 
already seen those two ways. So one is just not 
using HuggingFace necessarily, you just specify a certain dataset path. 
 
Another one, you could specify a HuggingFace path, 
and here I'm using a boolean value, use HuggingFace, to 
specify whether that's true. 
We include both for you here so you can easily use it. 
Again, we're going to use a smaller model so that it runs on CPU, so 
this is just 70 million parameters here. 
And then finally, I'm going to put all of 
this into a training config, which will be then passed 
onto the model, just to understand, you know, what 
the model name is and the data is. 
Great. 
So the next step is the tokenizer. 
You've already done this in the past lab, but 
here again, you are loading that tokenizer and then 
splitting your data. 
So here's just the training and test set, and 
this is loading it up from HuggingFace. 
Next just loading up the model, you already specified 
the model name above. 
So that's 70 million parameter Pythia model. 
I'm just going to specify that as the base model, which hasn't been 
trained yet. 
Next an important piece of code. 
If you're using a GPU, this is PyTorch code that 
will be able to count how many CUDA devices, basically 
how many GPUs you have. 
And depending on that, if you have more than zero of them, 
that means you have a GPU. 
So you can actually put the model on GPU. 
Otherwise it'll be CPU. 
In this case, we're going to be using CPU. 
You can see select CPU device. 
All right. So just to put the model on that GPU or CPU, 
you just have to do the model to device. 
So very simple. 
So now this is printing out the, you know, 
what the model looks like here, but it's putting it on that device. 
All right. 
So putting together steps from the previous lab, 
but also adding in some new steps is inference. 
So you've already seen this function before, but 
now stepping through exactly what's going on. 
So first you're tokenizing that text coming in. 
You're also passing in your models. 
So that's the model here, and you want the model to 
generate based on those tokens. 
Now the tokens have to be put onto the same device so that, 
you know, if the model is on GPU, for example, you need to put the 
tokens on GPU as well. So the model can actually see it. 
And then next there's an important, you know, max input tokens and max 
output tokens here as parameters for specifying, you 
know, how many tokens can actually be put into the 
model as input. And then how many do you expect out? 
We're setting this to a hundred here as a default, but 
feel free to play with this make it longer so it generates 
more. 
Note that it does take time to generate 
more so expect a difference in the time 
it takes to generate. 
Next the model does generate some tokens out 
and so all you have to do is decode it with that 
tokenizer just like you saw before and here 
after you decode it you just have to 
strip out the prompt initially because it's 
just outputting both the prompt with your generated 
output and so I'm just having that return that 
generated text answer. 
So great this function you're going to be using a lot. 
So first up is taking a look at that first 
test set question and putting it through the model and 
try not to be too harsh and I know you've 
already kind of seen this before so again 
the model is answering this really weird way 
that you've seen before. 
It's not really answering the question which 
is here and the correct answer is here. 
Okay so this is what training is for. 
So next you're going to look at the training arguments. 
So there are a lot of different arguments. 
First, key in on a few. 
So the first one is the max number of steps 
that you can run on the model. 
So this is just max number of training steps. 
We're gonna set that to three just to make it very simple, just 
to walk through three different steps. 
What is a step exactly? 
A step is a batch of training data. 
And so if your batch size is one, it's just one data point. If 
your batch size is 2,000, it's 2,000 data points. 
Next is the trained model name. So what do you want to call it? 
So here I'm calling it the name of a dataset, plus, 
you know, the max steps here so that we can differentiate 
it if you want to play with different 
max steps and the word steps. 
Something I also think is the best practice that's 
not necessarily shown here is also to put 
the timestamp on the trained model because you 
might be experimenting with a lot of them. 
Okay, cool. 
So I'm now going to show you a big list of different training arguments. 
 
There are a lot of good defaults here. 
And I think the ones to focus on is max steps. 
This is probably going to stop the model from running past those 
three steps that you specified up there. 
And then also the learning rate. 
There are a bunch of different arguments here. 
I recommend that you can dive deeper into this if you're 
curious and be able to play with a lot of these arguments. 
But here we're largely setting these as 
good defaults for you. 
Next, we've included a function that calculates the 
number of floating point operations for 
the model. 
And so that's just flops and understanding the memory footprint of 
this base model. 
So here, it's just going to print that out here. 
This is just for your knowledge, just to understand what's going on. 
And we'll be printing that throughout training. 
And I know we said that this was a tiny, tiny model, but even here, 
look how big this model is here with 300 megabytes. 
So you can imagine a really large model to take up a 
ton of memory and this is why we need really high performing 
large memory GPUs to be able to run those larger models. 
Next you load this up in the trainer class. 
This is a class we wrapped around HuggingFaces 
main trainer class basically doing the same thing 
just printing out things for you as you train and as you 
can see you put a few things in. The main things are the base model, 
you put in you know max steps, the training arguments, 
and of course, your data sets you want to put in there. 
And the moment you've been waiting for. 
It is training the models. 
You just do "trainer.train". 
And let's see it go. 
Okay. 
Okay. So as you can see, it printed out a lot 
of different things in the logs, namely the loss. 
If you run this for more steps, even just 10 steps, you'll 
see the loss start to go down. 
All right. 
So now you've trained this model. 
Let's save it locally. 
So you can have a save directory, maybe specifying the output 
deer and the final as a final checkpoint. 
And then all you have to do is "trainer.savemodel". 
And let's see if it saved right here. 
So awesome, great work. 
Now that you've saved this model, you can actually load it up by just saying, 
you know, 
this auto model again from pre-trained and the save directory and you 
just have to specify local files equals true. 
So it doesn't pull from the HuggingFace hub in the cloud. 
I'm going to call this slightly fine-tuned model, 
or fine-tuned slightly model. 
And then I'm going to put this on the right device again. 
This is only important if you have a GPU, really, 
but here for CPU, just for good measure. 
And then let's run it. Let's see how it does. 
So let's see how it does on the test set again, or 
test data point again, and then just run inference. 
Again, this is the same inference function that you've 
run before. 
Cool. 
So is it any better? Not really. 
And is it supposed to be? Not really. It's 
only gone through a few steps. 
So what should it have been? 
Let's just take a look at that exact answer. 
So it's saying, yes, LAMNI can generate technical 
documentation user manuals. 
So it's it's very far from it. It's actually very similar still to that 
base model. 
Ok, but if you're patient, what could it look like? 
So we also fine tuned a model for far longer than that. 
So this model was only trained on three 
steps and actually in this case, three data points out of 1,260 
data points in the training data set. 
So instead we actually fine-tuned it on the entire data 
set twice for this "lamini_docs_finetunemodel" that we uploaded to HuggingFace that you 
can now download and actually use. 
And if you were to try this on your own computer, 
it might take half an hour or an hour, depending on your processor. 
Of course, if you have a GPU, it could just take a couple minutes. 
Great. So let's run this. 
Okay, this is a much better answer, and it's comparable to the 
actual target answer. 
But as you can see here at the end, it still starts to 
repeat itself additionally and laminize. So it's not perfect, but 
this is a much smaller model, and you could train it 
for even longer too. 
And now just to give you a sense of what 
a bigger model might do, This one was trained to be maybe a 
little bit less robust and repetitive. 
This is what a bigger 2.8 billion fine-tuned model would be. 
And this is running the LLAMA library with the same basic model 
runner as before. 
So here you can see, yes, LLAMA and I can generate technical 
documentation or user manuals. 
Ok, great. 
So one other thing that's kind of interesting in this dataset 
that we use to fine-tune that you can also do for your 
datasets is doing something called moderation, 
and encouraging the model to 
actually not get too off track. 
And if you look closely at the examples in this data set, 
which we're about to do, you'll see that there are examples that say, let's keep 
the discussion relevant to llamini. 
I'm going to loop through the data set here 
to find all the data points that say that, so 
that you can go see that yourself. 
So this is how you might prepare your own data set. 
And as a reminder, this is very similar to chat GPT. 
Sorry, i'm an AI and I can't answer that. So they're using a 
very similar thing here. 
So points it to the documentation to take a look 
at the fact that there isn't anything about Mars. 
All right, so now that you've run all of training here, you 
can actually do all of that in just three lines 
of code using Llamani's Llama library. 
And all you have to do is load up the model, 
load up your data and train it. 
And specifically here, we're running a slightly larger model. 
So the Pythia 410 million model, It's the biggest model that's available 
for a free tier. 
And then Llamani docs, you can load that up through a JSON 
lines file just like you did before and all you have to 
do is run "model.train". I'm running is public is true. So this 
is a public model that anyone can then 
run afterwards. 
Put that through instead of the Pythia 410 
in the basic model runner to then run it. 
You can also click on this link here to sign up, 
make an account. 
Basically you can see the results. 
You can run a chatbot there, kind of interface there to 
be able to see everything, but since is public is true, 
we can actually just look at the model 
results here on the command line, So "model.evaluate", run that. And 
here you can see, again, the same job ID. For this job ID, you 
can see all the evaluation results that were 
data points that were not trained on. 
And so just to pretty print this a little bit into a data frame, I'm 
gonna plop some code in here to reformat that. 
So this is just code that is reformatting that into a nice 
data frame from that list of dictionaries. 
Cool. 
And here you can see a lot of different, you know, questions and then answer 
from the train model versus the base model. 
So this is an easy way to compare those results. 
So here's a question. Does Lamini have the ability to understand 
and generate code for audio processing tasks? 
And you can see that the train model 
actually gave an answer. Yes, Lamini has the ability to understand, 
and generic codes not quite there yet. This is really a baby 
model and a very limited data, but it is much 
better than this base model that answers a colon. 
all nice a very good language for audio processing A 
colon. You know, yes, Lamini has the ability 
to understand and generate code. It's not quite there yet, so 
this is a really baby model with very limited data, but 
it. It is much better than this base model that 
answers with A colon, I think you are looking 
for a language that can be used to 
write audio code just very often, it keeps rambling. 
So a very big difference in performance. 
And now you can see a different question here, 
you know, is it possible to control the level of 
detail in the generated output? 
So as you can see, you can go through all these 
results and in the next lab, we'll actually explore how to evaluate 
all of these results. 

"""

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


