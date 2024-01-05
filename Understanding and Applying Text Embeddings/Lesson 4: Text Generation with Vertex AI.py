# Databricks notebook source
# MAGIC %md
# MAGIC ## Lesson 5: Text Generation with Vertex AI

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

# MAGIC %md
# MAGIC ### Prompt the model
# MAGIC - We'll import a language model that has been trained to handle a variety of natural language tasks, `text-bison@001`.
# MAGIC - For multi-turn dialogue with a language model, you can use, `chat-bison@001`.

# COMMAND ----------

import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)

# COMMAND ----------

from vertexai.language_models import TextGenerationModel

# COMMAND ----------

generation_model = TextGenerationModel.from_pretrained(
    "text-bison@001")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Question Answering
# MAGIC - You can ask an open-ended question to the language model.

# COMMAND ----------

prompt = "I'm a high school student. \
Recommend me a programming activity to improve my skills."

# COMMAND ----------

print(generation_model.predict(prompt=prompt).text)

# COMMAND ----------

#### Classify and elaborate
- For more predictability of the language model's response, you can also ask the language model to choose among a list of answers and then elaborate on its answer.

# COMMAND ----------

prompt = """I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""

# COMMAND ----------

print(generation_model.predict(prompt=prompt).text)

# COMMAND ----------

#### Extract information and format it as a table

# COMMAND ----------

prompt = """ A bright and promising wildlife biologist \
named Jesse Plank (Amara Patel) is determined to make her \
mark on the world. 
Jesse moves to Texas for what she believes is her dream job, 
only to discover a dark secret that will make \
her question everything. 
In the new lab she quickly befriends the outgoing \
lab tech named Maya Jones (Chloe Nguyen), 
and the lab director Sam Porter (Fredrik Johansson). 
Together the trio work long hours on their research \
in a hope to change the world for good. 
Along the way they meet the comical \
Brenna Ode (Eleanor Garcia) who is a marketing lead \
at the research institute, 
and marine biologist Siri Teller (Freya Johansson).

Extract the characters, their jobs \
and the actors who played them from the above message as a table
"""

# COMMAND ----------

response = generation_model.predict(prompt=prompt)

print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC - You can copy-paste the text into a markdown cell to see if it displays a table.
# MAGIC
# MAGIC
# MAGIC | Character | Job | Actor |
# MAGIC |---|---|---|
# MAGIC | Jesse Plank | Wildlife Biologist | Amara Patel |
# MAGIC | Maya Jones | Lab Tech | Chloe Nguyen |
# MAGIC | Sam Porter | Lab Director | Fredrik Johansson |
# MAGIC | Brenna Ode | Marketing Lead | Eleanor Garcia |
# MAGIC | Siri Teller | Marine Biologist | Freya Johansson |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adjusting Creativity/Randomness
# MAGIC - You can control the behavior of the language model's decoding strategy by adjusting the temperature, top-k, and top-n parameters.
# MAGIC - For tasks for which you want the model to consistently output the same result for the same input, (such as classification or information extraction), set temperature to zero.
# MAGIC - For tasks where you desire more creativity, such as brainstorming, summarization, choose a higher temperature (up to 1).

# COMMAND ----------

temperature = 0.0

# COMMAND ----------

temperature = 0.0

# COMMAND ----------

response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)

# COMMAND ----------

print(f"[temperature = {temperature}]")
print(response.text)

# COMMAND ----------

temperature = 1.0

# COMMAND ----------

response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)

# COMMAND ----------

print(f"[temperature = {temperature}]")
print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Top P
# MAGIC - Top p: sample the minimum set of tokens whose probabilities add up to probability `p` or greater.
# MAGIC - The default value for `top_p` is `0.95`.
# MAGIC - If you want to adjust `top_p` and `top_k` and see different results, remember to set `temperature` to be greater than zero, otherwise the model will always choose the token with the highest probability.

# COMMAND ----------

top_p = 0.2

# COMMAND ----------

prompt = "Write an advertisement for jackets \
that involves blue elephants and avocados."

# COMMAND ----------

response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_p=top_p,
)

# COMMAND ----------

print(f"[top_p = {top_p}]")
print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Top k
# MAGIC - The default value for `top_k` is `40`.
# MAGIC - You can set `top_k` to values between `1` and `40`.
# MAGIC - The decoding strategy applies `top_k`, then `top_p`, then `temperature` (in that order).

# COMMAND ----------

top_k = 20
top_p = 0.7

# COMMAND ----------

response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_k=top_k,
    top_p=top_p,
)

# COMMAND ----------

print(f"[top_p = {top_p}]")
print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC Our goal in this course is to build a question answering system. 
# MAGIC And we can actually do that already with just embeddings, 
# MAGIC but it turns out we can build an even better question answering 
# MAGIC system if we use the text generation capabilities 
# MAGIC of large language models. So, let's see how that works. We'll start 
# MAGIC off by importing our credentials and authenticating so that 
# MAGIC we can use the Vertex AI service. And we'll need to 
# MAGIC set our region, and then import Vertex AI and initialize 
# MAGIC the SDK. And once we've done the required 
# MAGIC setup, we can get started. So in this example, 
# MAGIC we'll start off by loading in the text generation model from Vertex. And 
# MAGIC before we were loading an embeddings model, this 
# MAGIC time we're loading this text generation model. 
# MAGIC When we load this model, it will be a different name. 
# MAGIC  
# MAGIC This model is called text bison instead of 
# MAGIC the embeddings gecko model we were looking at earlier. 
# MAGIC And the TextBison model is fine-tuned for a variety of natural language 
# MAGIC tasks like sentiment analysis, classification, summarization, 
# MAGIC and extraction. But note that this model 
# MAGIC is ideal for tasks that are completed with 
# MAGIC just a single API response, so not for continuous conversations. If 
# MAGIC you do have a use case that requires back-and-forth interactions, there's 
# MAGIC a separate model for that called ChatBison 
# MAGIC that you can check out instead. Now, when we talk about 
# MAGIC text generation in the context of large language 
# MAGIC models, these models will take as input some text 
# MAGIC and produce some output text that's likely to follow. And 
# MAGIC this input text we provide is called a prompt. 
# MAGIC So, let's start off with an open-ended kind of generative brainstorming task. 
# MAGIC  
# MAGIC We'll define our prompt as, I'm a high school student. Recommend me 
# MAGIC a programming activity to improve my skills. This, again, is 
# MAGIC the input that we will pass to our model. So, once we've defined our 
# MAGIC prompt, we can print out the response from 
# MAGIC our model. So, we will call predict and we will 
# MAGIC pass in the prompt and then we can print out the results. So, let's see 
# MAGIC what the model produces. The model suggesting that 
# MAGIC we write a program to solve a problem 
# MAGIC you're interested in. Definitely good advice. Also taking 
# MAGIC a programming course and a couple of 
# MAGIC other ideas here. 
# MAGIC Now, this is a pretty open-ended response, and it might be 
# MAGIC useful for brainstorming, but it's pretty variable. And with 
# MAGIC large language models, we can kind of get them to take on different 
# MAGIC behaviors by writing strategic input text. So, if we 
# MAGIC wanted to have maybe a more restrictive answer, we could 
# MAGIC take this open-ended generative task and turn it into a classification 
# MAGIC task to basically reduce the output variability. So, let's 
# MAGIC see what that might look like. Here is a rephrasing 
# MAGIC of that same prompt. It's just a little different. 
# MAGIC  
# MAGIC Now we're saying, I'm a high school student, which of 
# MAGIC these activities do you suggest and why? So, instead of 
# MAGIC just keeping it completely open-ended, we provided a 
# MAGIC few different options A, learn Python, B, learn JavaScript, or C, learn 
# MAGIC Fortran. So, let's see what happens when we pass this prompt to 
# MAGIC the model. Again, we'll be printing out the results of calling the 
# MAGIC predict function on our text generation model, and 
# MAGIC we're passing in this prompt text. So, this time the model is suggesting 
# MAGIC that we should learn a Python, And again, 
# MAGIC it's a pretty reasonable answer to our question, but it's just 
# MAGIC a little bit more restrictive based on our particular prompt. 
# MAGIC And this sort of art and science of 
# MAGIC figuring out what the best prompt is for your use case is 
# MAGIC called prompt engineering. And there are a lot of different tips 
# MAGIC and best practices. So if you're curious, you could definitely 
# MAGIC check out some of the other deeplearning.ai courses on prompt engineering. But 
# MAGIC for now, we're just going to try out one 
# MAGIC other task with this large language model. So, something I think is 
# MAGIC really interesting about large language models is that we can 
# MAGIC use them to extract information. 
# MAGIC  
# MAGIC In other words, take data that's in one format and reformat 
# MAGIC it into another format. So here, we've got 
# MAGIC this long chunk of text, but it's a synopsis for an imaginary 
# MAGIC movie about a wildlife biologist. So, in this imaginary 
# MAGIC movie synopsis, we've got the names of the characters and 
# MAGIC their different jobs and also the actors who 
# MAGIC played them. So, what we're going to do is instruct this model to extract 
# MAGIC all three of those fields. We'll instruct it to extract the 
# MAGIC characters, their jobs, and the actors who played them. 
# MAGIC So, this long piece of text is the prompt that we'll pass 
# MAGIC to the model. So, let's try and see what the response is. 
# MAGIC  
# MAGIC Here you can see that the model did in fact extract all 
# MAGIC of the characters in the synopsis, as well as their jobs, 
# MAGIC and then the actors who played them. And if we wanted to get 
# MAGIC this in maybe a different format, we could even say something like, 
# MAGIC extract this information from the above message as a table. 
# MAGIC And we can see what happens here. We actually get some markdowns. So, 
# MAGIC we can test this markdown and see if it is 
# MAGIC actually in fact valid markdown. Let's 
# MAGIC turn this into a markdown cell. And there we go. We 
# MAGIC got this nice table. 
# MAGIC So, we can actually use large language 
# MAGIC models to extract information and convert 
# MAGIC data from one format to another. So, let's review the key syntax that 
# MAGIC we just ran through. We first imported the 
# MAGIC text generation model, and then we selected the specific 
# MAGIC model we wanted to use, which in this case was TextBison, and 
# MAGIC this was the 001 version. And then, we defined a prompt, which 
# MAGIC is the input text, and we called predict, and 
# MAGIC we passed in this prompt. Now, in addition to adjusting the 
# MAGIC words and the word order of our prompt, there are some additional hyperparameters 
# MAGIC that we can set in order to get the model to produce 
# MAGIC some different results. So earlier, I said that these models 
# MAGIC take as input some text and they produce 
# MAGIC some output text that's likely to follow, but we 
# MAGIC can actually be a little bit more precise in this definition. 
# MAGIC Really these models take as input some text, 
# MAGIC maybe the garden was full of beautiful, and they produce as output an 
# MAGIC array of probabilities over tokens that could come next. 
# MAGIC And Andrew talked a little bit about tokens 
# MAGIC in a previous lesson, but tokens are essentially 
# MAGIC the basic unit of text that is processed 
# MAGIC by a large language model. So, depending on the 
# MAGIC different tokenization method, this might be words or sub words or 
# MAGIC other fragments of text. Now, I'm saying tokens, but 
# MAGIC in the slides here, I'm actually just going to show individual 
# MAGIC words and that's just to make it a little bit easier to understand 
# MAGIC the concepts. But again, these models are returning an array 
# MAGIC of probabilities over tokens that could 
# MAGIC come next. And from this array, we need to 
# MAGIC decide which one we should choose. This is known 
# MAGIC as a decoding strategy. 
# MAGIC So, a simple strategy might be to select the 
# MAGIC token with the highest probability at each time step. 
# MAGIC And this is known as greedy decoding. But it can result 
# MAGIC in some uninteresting and sometimes even repetitive answers. 
# MAGIC Now, on the flip side, if we were to just 
# MAGIC randomly sample over the distribution, we might end up 
# MAGIC with some unusual tokens or some unusual responses. 
# MAGIC And by controlling this degree of randomness, we can control how 
# MAGIC unusual or how rare the words are that 
# MAGIC get put together in our response. One of the parameters we can set in 
# MAGIC order to control this randomness is called temperature. 
# MAGIC Lower temperature values are better for use cases 
# MAGIC that require a more deterministic or less open-ended 
# MAGIC responses. 
# MAGIC So, maybe if you're doing something like classification or an 
# MAGIC extraction task like the synopsis for or an imaginary 
# MAGIC movie task we just looked at, you might wanna start with 
# MAGIC a lower temperature value. On the other hand, higher 
# MAGIC temperature values are better for more open-ended use 
# MAGIC cases. So, maybe something like brainstorming or 
# MAGIC even summarization where you might want 
# MAGIC more unusual responses or unusual words. Typically, with 
# MAGIC neural networks, we have some raw output called logits. And we 
# MAGIC pass these logits to the softmax function in order to get a 
# MAGIC probability distribution over classes. You can 
# MAGIC think of the different classes in this case as just being the 
# MAGIC different tokens that we might return to the user. 
# MAGIC So, when we apply temperature, what we're doing is we're taking 
# MAGIC our softmax function and we're dividing each of the logits values by our 
# MAGIC temperature value. So, on the slide here, you can first 
# MAGIC see our softmax function. And then below that, you can see 
# MAGIC the softmax function with temperature applied where each 
# MAGIC of our logits values z are divided by 
# MAGIC theta. And that's how we actually apply temperature. Now, if this didn't 
# MAGIC make a whole lot of sense to you, don't worry about it. The 
# MAGIC actual mechanics here aren't that important. What's really more important 
# MAGIC is that you get an intuitive understanding of how temperature 
# MAGIC works. So, one way to think 
# MAGIC about temperature is that as you decrease the temperature value, 
# MAGIC you're increasing the likelihood of selecting the most probable 
# MAGIC token. And if we take that 
# MAGIC to the extreme and we make a temperature value 
# MAGIC of zero, it will be deterministic. That means the most probable 
# MAGIC token will always be selected. 
# MAGIC  
# MAGIC On the other hand, you can think of increasing the temperature as 
# MAGIC basically flattening the probability distribution and 
# MAGIC increasing the likelihood of selecting less probable tokens. 
# MAGIC With the Vertex AI model we're looking at, you 
# MAGIC can set a temperature value between zero and one, and 
# MAGIC for most use cases, starting with a number like 0.02 can 
# MAGIC be a good starting place, and you can adjust it from there. So, let's 
# MAGIC see how we actually set this value in the notebook. Let's start off with 
# MAGIC the temperature value of zero. Again, this is deterministic, 
# MAGIC and it's going to select the most likely token at each time 
# MAGIC step. So here's a prompt. Let's say, complete the sentence. As I prepared the 
# MAGIC picture frame, I reached into my toolkit to fetch 
# MAGIC my, and we'll see what the model responds with. 
# MAGIC  
# MAGIC So, we will call the predict function as we've 
# MAGIC done before on our generation model. And we'll pass in the prompt, but this 
# MAGIC time, we're also going to pass in the temperature value. And then we 
# MAGIC can print out this response. So, the model says, 
# MAGIC as I prepared the picture frame, I reached into 
# MAGIC my toolkit to fetch my hammer. And that seems 
# MAGIC like a pretty reasonable response, probably the most 
# MAGIC likely thing someone would fetch from 
# MAGIC their toolkit for this particular example. And remember, temperature 
# MAGIC of 0 is deterministic. So, even if 
# MAGIC we run this again, we will get the exact same answer. So, let's try 
# MAGIC this time setting the temperature to 1. And again, we can 
# MAGIC call the predict function on our model. And we will print 
# MAGIC out the result with this different temperature value. 
# MAGIC And this time, we reached into the toolkit to fetch my 
# MAGIC saw. I ran this earlier. I saw sandpaper, which 
# MAGIC I thought was a pretty interesting response. 
# MAGIC  
# MAGIC The model also actually produced some 
# MAGIC additional information here as well. So, you can try this out, and you'll 
# MAGIC get a different response if you run this again. So, I 
# MAGIC encourage you to try out some different temperature 
# MAGIC values and see how that changes the responses 
# MAGIC from the model. Now, in addition to temperature, there 
# MAGIC are two other hyperparameters that you 
# MAGIC can set to impact the randomness and the output of the model. 
# MAGIC So, let's return to our example from earlier where we had 
# MAGIC an input sentence, the garden was full of beautiful, and this 
# MAGIC probability array over tokens. One strategy for selecting the next token 
# MAGIC is called TopK, where you sample from a shortlist of 
# MAGIC the TopK tokens. So, in this case, if we set K to two, that's the 
# MAGIC two most probable tokens, flowers and trees. 
# MAGIC  
# MAGIC Now, TopK can work fairly well for examples where 
# MAGIC you have several words that are all fairly likely, 
# MAGIC but it can produce some interesting or sometimes not 
# MAGIC particularly great results when you have 
# MAGIC a probability distribution that's very skewed. So, in other words, you 
# MAGIC have a one word that's very likely and a bunch of other words that 
# MAGIC are not very likely. And that's because the top K value is 
# MAGIC hard coded for a number of tokens. So, it's not dynamically adapting 
# MAGIC to the number of tokens. So, to address this limitation, 
# MAGIC another strategy is top P, where we can dynamically 
# MAGIC set the number of tokens to sample from. 
# MAGIC And in this case, we would sample from 
# MAGIC the minimum set of tokens whose cumulative of 
# MAGIC probability is greater than or equal to P. 
# MAGIC So, in this case, if we set P to be 0.75, we just add the probabilities starting from the 
# MAGIC most probable token. So, that's flowers at 0.5, and 
# MAGIC then we add 0.23, 0.05, and now we've hit the threshold of 0.75. So, 
# MAGIC we would sample from these three tokens alone. So, you don't 
# MAGIC need to set all of these different values, but 
# MAGIC if you were to set all of them, this is how 
# MAGIC they all work together. 
# MAGIC  
# MAGIC First, the tokens are filtered by top K, 
# MAGIC and from those top K, they're further filtered by top P. And 
# MAGIC then finally, the output token is selected 
# MAGIC using temperature sampling. And that's how we arrive at the 
# MAGIC final output token. So, let's jump into the notebook and try 
# MAGIC and set some of these values. So, first we'll start 
# MAGIC off by setting a top P value of 0.2. And note that by default, 
# MAGIC the top P value is going to be set at 
# MAGIC 0.95. And this parameter can take values between 0 and 1. So, 
# MAGIC here is a fun prompt. Let's ask for an advertisement about jackets that involves blue 
# MAGIC elephants and avocados, two of my 
# MAGIC favorite things. So, we can call the generation model predict 
# MAGIC function again. And this time, we'll pass in 
# MAGIC the prompt. We'll also pass in a temperature value, let's try 
# MAGIC something like 0.9, and then we'll also pass in top P. And note that 
# MAGIC temperature by default at zero does result in a deterministic response. 
# MAGIC  
# MAGIC  
# MAGIC It's greedy decoding, so the most likely token will be selected at each 
# MAGIC timestamp. So, if you want to play around with 
# MAGIC top P and top K, just set the temperature value to something 
# MAGIC other than zero. So, we can print out the response here 
# MAGIC and see what we get. And here, is an advertisement introducing 
# MAGIC this new blue elephant avocado jacket. So 
# MAGIC lastly, let's just see what it looks like to set 
# MAGIC top P and top K. So let's set a top K to 20. And by default, top K is 
# MAGIC going to be set to 40. And this parameter takes values between one 
# MAGIC and 40. So, we'll half that default value. And then, we'll also 
# MAGIC set top P so we can set all three of these parameters we just learned 
# MAGIC about. And we'll use the exact same prompt 
# MAGIC as before, we'll just keep it as write an advertisement for jackets that 
# MAGIC involves blue elephants and avocados. 
# MAGIC  
# MAGIC And this time, when we call the predict 
# MAGIC function on our generation model, we'll pass in the prompt, the 
# MAGIC temperature value, the top k value, and 
# MAGIC the top p value. And just as a reminder, this 
# MAGIC means that the output tokens will be first filtered by 
# MAGIC the top k tokens, then further filtered by top p, and 
# MAGIC lastly, the response tokens will be selected with 
# MAGIC temperature sampling. So here, we've got a response here, and we 
# MAGIC can see that it is a little different from the 
# MAGIC one we saw earlier. So, I encourage you to try out some different 
# MAGIC values for top p, top k, and temperature, and also 
# MAGIC try out some different prompts and see what kinds of interesting responses 
# MAGIC or use cases or behaviors you can get these 
# MAGIC large language models to take on. So, just as a quick recap of the 
# MAGIC syntax we just learned, again, we've been importing this text generation 
# MAGIC model, and then we loaded this text bison 
# MAGIC model, and we define a prompt, which is the 
# MAGIC input text to our model. And when we call predict, we can, in addition to 
# MAGIC passing in a prompt, also pass in a value for temperature, top K and 
# MAGIC top P. So now that you know a little bit about how to use these models for 
# MAGIC text generation, I encourage you to jump into the 
# MAGIC notebook, try out some different temperature, top P and 
# MAGIC top K values, and also experiment with some different 
# MAGIC prompts. And when you're ready, we'll take what you've learned 
# MAGIC about text generation and combine it with embeddings 
# MAGIC in the next lesson to build a question 
# MAGIC answering system. 
# MAGIC
