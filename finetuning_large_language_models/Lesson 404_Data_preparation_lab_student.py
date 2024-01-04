# Databricks notebook source
"""
Data preparation
"""

# COMMAND ----------

import pandas as pd
import datasets

from pprint import pprint
from transformers import AutoTokenizer

# COMMAND ----------

"""
Tokenizing text
"""

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# COMMAND ----------

text = "Hi, how are you?"

# COMMAND ----------

encoded_text = tokenizer(text)["input_ids"]

# COMMAND ----------

encoded_text

# COMMAND ----------

decoded_text = tokenizer.decode(encoded_text)
print("Decoded tokens back into text: ", decoded_text)

# COMMAND ----------

"""Tokenize multiple texts at once"""


# COMMAND ----------

list_texts = ["Hi, how are you?", "I'm good", "Yes"]
encoded_texts = tokenizer(list_texts)
print("Encoded several texts: ", encoded_texts["input_ids"])

# COMMAND ----------

"""
Padding and truncation
"""

# COMMAND ----------

tokenizer.pad_token = tokenizer.eos_token 
encoded_texts_longest = tokenizer(list_texts, padding=True)
print("Using padding: ", encoded_texts_longest["input_ids"])

# COMMAND ----------

encoded_texts_truncation = tokenizer(list_texts, max_length=3, truncation=True)
print("Using truncation: ", encoded_texts_truncation["input_ids"])

# COMMAND ----------

tokenizer.truncation_side = "left"
encoded_texts_truncation_left = tokenizer(list_texts, max_length=3, truncation=True)
print("Using left-side truncation: ", encoded_texts_truncation_left["input_ids"])

# COMMAND ----------

encoded_texts_both = tokenizer(list_texts, max_length=3, truncation=True, padding=True)
print("Using both padding and truncation: ", encoded_texts_both["input_ids"])

# COMMAND ----------

"""
Prepare instruction dataset
"""

# COMMAND ----------

import pandas as pd

filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
examples = instruction_dataset_df.to_dict()

if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

prompt_template = """### Question:
{question}



### Answer:"""

num_examples = len(examples["question"])
finetuning_dataset = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
  text_with_prompt_template = prompt_template.format(question=question)
  finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})

from pprint import pprint
print("One datapoint in the finetuning dataset:")
pprint(finetuning_dataset[0])

# COMMAND ----------

"""Tokenize a single example"""

# COMMAND ----------

text = finetuning_dataset[0]["question"] + finetuning_dataset[0]["answer"]
tokenized_inputs = tokenizer(
    text,
    return_tensors="np",
    padding=True
)
print(tokenized_inputs["input_ids"])

# COMMAND ----------

max_length = 2048
max_length = min(
    tokenized_inputs["input_ids"].shape[1],
    max_length,
)

# COMMAND ----------

tokenized_inputs = tokenizer(
    text,
    return_tensors="np",
    truncation=True,
    max_length=max_length
)

# COMMAND ----------

tokenized_inputs["input_ids"]

# COMMAND ----------

"""
Tokenize the instruction dataset
"""

# COMMAND ----------

def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    else:
      text = examples["text"][0]

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs

# COMMAND ----------

finetuning_dataset_loaded = datasets.load_dataset("json", data_files=filename, split="train")

tokenized_dataset = finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)

print(tokenized_dataset)

# COMMAND ----------

tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

# COMMAND ----------

"""
Prepare test/train splits
"""

# COMMAND ----------

split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
print(split_dataset)

# COMMAND ----------

"""Some datasets for you to try"""
finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = datasets.load_dataset(finetuning_dataset_path)
print(finetuning_dataset)

# COMMAND ----------

taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "lamini/open_llms"

# COMMAND ----------

dataset_swiftie = datasets.load_dataset(taylor_swift_dataset)
print(dataset_swiftie["train"][1])

# COMMAND ----------

# This is how to push your own dataset to your Huggingface hub
# !pip install huggingface_hub
# !huggingface-cli login
# split_dataset.push_to_hub(dataset_path_hf)

# COMMAND ----------

"""
Now after exploring the data that you'll be using, in this lesson you'll learn 
about how to prepare that data for training. 
All right, let's jump into it. 
So next on what kind of data you need to prep, 
well there are a few good best practices. 
So one is you want higher quality data and actually 
that is the number one thing you need for fine-tuning 
rather than lower quality data. 
What I mean by that is if you give it garbage inputs, it'll 
try to parrot them and give you garbage outputs. 
So giving really high quality data is important. 
Next is diversity. 
So having diverse data that covers a lot of aspects 
of your use case is helpful. 
If all your inputs and outputs are the same, 
then the model can start to memorize them and if that's 
not exactly what you want, then the model will start 
to just only spout the same thing over 
and over again. And so having diversity in 
your data is, is really important. 
Next is real or generated. 
I know there are a lot of ways to create generated data, 
and you've already seen one way of doing that using an LLM, but 
actually having real data is very, very effective and helpful most 
of the time, especially for those writing tasks. 
 
And that's because generated data already has 
certain patterns to it. You might've heard of some services that 
are trying to detect whether something is generated 
or not. And that's actually because there are patterns 
in generated data that they're trying to detect. 
 
And as a result, if you train on more of the same patterns, it's 
not going to learn necessarily new patterns or 
new ways of framing things. 
And finally, I put this last because actually 
in most machine learning applications, 
having way more data is important than less data. 
But as you actually just seen before, pre-training 
handles a lot of this problem. 
Pre-training has learned from a lot of data, all 
from the internet. 
And so it already has a good base understanding. It's 
not starting from zero. 
And so having more data is helpful for the model, but not as 
important as the top three and definitely not as 
important as quality. 
So first, let's go through some of the steps of collecting your data. 
So you've already seen some of those instruction response pairs. 
So the first step is collect them. 
The next one is concatenate those pairs or 
add a prompt template. You've already seen that as well. 
The next step is tokenizing the data, um, 
adding padding or truncating the data. 
So it's the right size going into the model and you'll see 
how to tokenize that in the lab. 
So the steps to prepping your data is one 
collecting those instruction response pairs. 
Maybe that's question answer pairs, and then it's concatenating 
those pairs together, adding some prompt template, like you 
did before. 
The third step is tokenizing that data. 
And the last step is splitting that data 
into training and testing. 
Now in tokenizing, what, what does that really mean? 
Well, tokenizing your data is taking your text data and 
actually turning that into numbers that represent each of 
those pieces of text. 
It's not actually necessarily by word. 
It's based on the frequency of, you know, common 
character occurrences. 
And so in this case, one of my favorites is the ING token, 
which is very common in tokenizers. 
And that's because that happens in every single gerund. 
So in here, you can see finetuning, ING. 
So every single, you know, verb in the gerund, you know, fine-tuning 
or tokenizing all has ING and that maps 
onto the token 278 here. 
And when you decode it with the same tokenizer, 
it turns back into the same text. 
Now there are a lot of different tokenizers and 
a tokenizer is really associated with 
a specific model for each model as it was trained on it. 
And if you give the wrong tokenizer to your model, it'll 
be very confused because it will expect different numbers 
to represent different sets of letters 
and different words. 
So make sure you use the right tokenizer and you'll 
see how to do that easily in the lab. 
Cool, so let's head over to the notebook. 
Okay, so first we'll import a few different libraries and 
actually the most important one to see here is the AutoTokenizer class 
from the Transformers library by HuggingFace. 
And what it does is amazing. 
It automatically finds the right tokenizer or for your 
model when you just specify what the model is. So all you have to do 
is put the model and name in, and this is the same 
model name that you saw before, which is a 70 
million Pythium base model. 
Okay, so maybe you have some text that says, 
you know, hi, how are you? 
So now let's tokenize that text. 
So put that in, boom. 
So let's see what encoded text is. 
All right, so that's different numbers representing 
text here. 
Tokenizer outputs a dictionary with input 
IDs that represent the token, so I'm just printing that here. 
And then let's actually decode that back into the text and see if it actually 
turns back into hi, how are you? 
Cool, awesome, it turns back into hi, how are you, so that's great. 
All right, so when tokenizing, you probably are putting 
in batches of inputs, so let's just take a look at 
a few different inputs together, so there's hi, how are you, I'm good, and 
yes. 
So putting that list of text through, you can just put it 
in a batch like that. 
Into the tokenizer, you get a few different things here. 
So here's hi, how are you again. 
I'm good, it's smaller. 
And yes, it's just one token. 
So as you can see, these are varying in length. 
Actually, something that's really important for models is 
that everything in a batch is the same length, because 
you're operating with fixed size tensors. 
And so the text needs to be the same. 
So one thing that we do do is something called padding. 
Padding is a strategy to handle these variable length encoded texts. 
Um, and for our padding token, you have to specify, you know, 
what you want to, what number you want to represent for, 
for padding. And specifically we're using a zero, which 
is actually the end of sentence token as well. 
So when we run, padding equals true through the tokenizer, 
you can see the yes string has a lot of 
zeros padded there on the right, just to match the length of this hi, 
how are you string. 
Your model will also have a max length that it can handle 
and take in so it can't just fit everything in and you've played 
with prompts before and you've noticed probably that there is a 
limit to the prompt length and so this is the same thing 
and truncation is a strategy to handle making 
those encoded text much shorter and that fit 
actually into the model so this is one way to make it 
shorter so as you can see here I'm just artificially changing 
the max length to three, setting truncation to true, 
and then seeing how it's,much shorter now, for hi, how 
are you? 
It's truncating from the right, so it's just getting rid of everything here 
on the right. 
Now, realistically, actually one thing that's 
very common is, you know, you're writing a prompt, maybe you have your instruction 
somewhere,and you have a lot of the important things maybe on 
the other side,on the right and that's getting truncated out. 
 
So, you know, specifying truncation side to 
the left actually can truncate it the other way. 
So this really depends on your task. 
And realistically for padding and truncation, 
you want to use both. So let's just actually set both in there. So 
truncation's true and padding's true here. 
I'm just printing that out so you can see the zeros here, but 
also getting truncated down to three. 
Great, so that was really a toy example. 
I'm going to now paste some code that you did in 
the previous lab on prompts. 
So here it's loading up the data set file with the 
questions and answers, putting it into the prompt, hydrating 
those prompts all in one go. 
So now you can see one data point here of question and answer. 
So now you can run this tokenizer on 
just one of those data points. 
So first concatenating that question with that answer and 
then running it through the tokenizer. I'm 
just returning the tensors as a NumPy array 
here just to be simple and running it 
with just padding and that's because I don't know how long these tokens actually 
will be, and so what's important is that 
I then figure out, you know, the minimum between the max length and 
the tokenized inputs. 
Of course, you can always just pad to the longest. 
You can always pad to the max length and so that's 
what that is here. 
And then I'm tokenizing again with truncation up 
to that max length. 
So let me just print that out. 
And just specify that in the dictionary, and cool. 
So that's what the tokens look like. 
All right, so let's actually wrap this into a full-fledged function 
so you can run it through your entire 
data set. 
So this is, again, the same things happening here 
that you already looked at, grabbing the max length, 
setting the truncation side. 
So that's a function for tokenizing your data set. 
And now what you can do is you can load up that dataset. 
There's a great map function here. So you can map the tokenize 
function onto that dataset. 
And you'll see here I'm doing something really simple. 
So I'm setting batch size to one, it's very simple. 
It is gonna be batched and dropping last batch true. That's 
often what we do to help with mixed size inputs. 
And so the last batch might be a different size. 
Cool. 
Great, and then so the next step is to split the data set. 
So first I have to add in this labels columns 
as for hugging face to handle it, and then I'm going to run this 
train test split function, and I'm going 
to specify the test size as 10% of the data. 
So of course you can change this depending on how 
big your data set is. 
Shuffle's true, so I'm randomizing the order of this 
data set. 
I'm just going to print that out here. 
So now you can see that the data set has been split 
across training and test set, 140 for a test set there. 
And of course this is already loaded up 
in Hugging Face like you had seen before, 
so you can go there and download it and see 
that it is the same. 
So while that's a professional data set, it's about 
a company, maybe this is related to your 
company for example, you could adapt it to your company. 
We thought that might be a bit boring, it doesn't have to be, so 
we included a few more interesting datasets that you 
can also work with and feel free to customize and train your 
models for these instead. 
One is for Taylor Swift, one's for the popular band BTS, and 
one is on actually open source large language models 
that you can play with. 
And just looking at, you know, one data point from the TayTay dataset, 
let's take a look. 
All right, what's the most popular. 
Taylor Swift song among millennials? 
How does this song relate to the millennial generation? 
Okay, okay. 
So, you can take a look at this yourself and yeah, 
these data sets are available via HuggingFace. 
And now in the next lab, now that you've 
prepped all this data, tokenized it, you're ready 
to train the model. 

"""
