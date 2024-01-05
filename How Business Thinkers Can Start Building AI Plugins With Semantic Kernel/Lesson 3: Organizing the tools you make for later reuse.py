# Databricks notebook source
"""
Grow the existing business
Save money and time
Add completely new business
Prepare for the unknown
"""

# COMMAND ----------

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

print("A kernel is now ready.")    

# COMMAND ----------

strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily at some of the best pizzerias","Strong local reputation","Prime location on university campus" ]
weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]

pluginsDirectory = "./plugins-sk"

pluginBT = kernel.import_semantic_skill_from_directory(pluginsDirectory, "BusinessThinking");

my_context = kernel.create_new_context()
my_context['input'] = 'makes pizzas'
my_context['strengths'] = ", ".join(strengths)
my_context['weaknesses'] = ", ".join(weaknesses)

costefficiency_result = await kernel.run_async(pluginBT["SeekCostEfficiency"], input_context=my_context)
costefficiency_str = str("### ✨ Suggestions for how to gain cost efficiencies\n" + str(costefficiency_result))
display(Markdown(costefficiency_str))

# COMMAND ----------

opportunities = [ "Untapped catering potential","Growing local tech startup community","Unexplored online presence and order capabilities","Upcoming annual food fair" ]
threats = [ "Competition from cheaper pizza businesses nearby","There's nearby street construction that will impact foot traffic","Rising cost of cheese will increase the cost of pizzas","No immediate local regulatory changes but it's election season" ]

pluginBT = kernel.import_semantic_skill_from_directory(pluginsDirectory, "BusinessThinking");

my_context = kernel.create_new_context()
my_context['input'] = 'makes pizzas'
my_context['strengths'] = ", ".join(strengths)
my_context['weaknesses'] = ", ".join(weaknesses)
my_context['opportunities'] = ", ".join(opportunities)
my_context['threats'] = ", ".join(threats)

bizstrat_result = await kernel.run_async(pluginBT["BasicStrategies"],input_context=my_context)
bizstrat_str = "## ✨ Business strategy thinking based on SWOT analysis\n"+str(bizstrat_result)
display(Markdown(bizstrat_str))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

"""
So, you're cooking now with AI, and what you discover? It 
gets messy. It's like, whoa, all these functions here and there, and this 
variable here, like, how do I organize this? You need 
plastic containers. You need ways to organize this AI 
tool stuff that you're building. Plugins, functions, native things, semantic 
things. How do I organize them? No problem. We have an 
organization system for your kitchen to be able to manage your tools. 
Not only that, improve them, refine them like 
a knife sharpener. So, let's jump in, get organized so we 
can cook even better AI. 
So, you built a lot of inline functions in your code, 
and you discovered the beauty of having native functions 
which can maybe make a pig Latin sort of a thing happen 
which isn't as cool as converting the entire SWAT into 
a different domain, you get both native functions, AI 
prompts, and it turns out that, you know, large 
language models are bad at math, generally speaking, so 
you want native functions to do math. But 
if you want to do semantic stuff, where do you go? AI prompts. And 
you can enter things in pure code to 
build a semantic function, or we have an option to package 
your templated prompt and your configuration 
information. 
And if you see here, this input is here. 
This input is here. This configuration file is a JSON file. 
And this prompt file is a regular text file. And don't 
forget, we're living in the completion world. You can 
complete this prompt with the semantic completion model. Just a 
reminder, because we want to win the conceptual battle. Semantic completion 
is what we're doing. Every kind of 
prompt template, complete the blank. That's what we're doing a lot of. 
We haven't run into semantic similarity yet. We're going to go there eventually, 
but let's pay attention. Two kinds of ways to use this kind 
of AI. Now, back to business, what are we doing? We're 
trying to address the challenges of a business, 
a small business or a big business. And with 
a business, you have pretty simple rules of how to win. 
You want to grow your business. You want to 
save money and time, and sometimes you want to add a completely new 
business line. you know, brand new revenue, yum, or you wanna 
actually prepare for the unknown, like some kind of oops 
can occur. 
Remember the pandemic, that was pretty bad. So, 
with those ideas in mind, let's go and build a kernel. If 
you remember, everything's the same here, it looks 
familiar to you. I'm gonna run this and your kernel is ready. And we're 
gonna do differently this time is we're not gonna type a 
lot of stuff in line to create a function. We're going to just leverage a 
pre-made business thinking plugin in that if you want to know 
where it is, look in your directory, it looks like this. There's a plugins 
SK folder. Inside it is a business thinking folder. There are basic strategies, 
seek cost efficiency, seek time efficiency. There's two 
files each. One is the config.json. The 
other is the text file that includes a template prompt. And once you stick 
these in folders, you don't have 
to put them in the code. 
 
You can just refer to them, which sounds pretty awesome, doesn't 
it? Okay, so we're going to do a couple of things. We're going 
to grab a connection to this plugin. Notice this is 
in plugins SK, directory plugins SK, and I am 
gonna go to the sub directory, business thinking, business thinking. I'm 
going to grab a couple of my SWOT 
items, strengths. Remember the unique garlic pizza? You gotta love it. 
I'm going to also make a new context where I 
ticked. We're going to make three variables. One is the input to the 
plugin, Mixpizza. I'm going to take those strengths and weaknesses. See that all 
these, this is a list here. I'm going 
to join the list together. So, it's like one long string. Okay, so 
we have three choices. So, we have basic strategies, cost efficiency, time efficiency. 
 
Let's use the cost efficiency plugin, which 
looks like this. If you recall, it's a 
templated prompt. It takes a business with strengths 
and weaknesses, and it formats it into a 
beautiful table. Let's go do this. Let's see joint weaknesses. 
Let's, okay, I'm gonna go cost efficiency. How are you, how are you 
feeling out there? Is this kind of something that 
you want to do with your time? Certainly, 
you do, because you're here., And I am with you here. I 
am doing the work of typing with you, and you can play me at two 
X, no embarrassment. Input context equals my context. Okay, looks 
good. This is gonna run this seek cost efficiency, "skprompt.txt" with the 
configuration, attach them to the model parameters. It's gonna 
take this context. And if I am right, we are going to 
print it out. 
Let's get the pleading ready for you. So, 
you'd have to listen to me type. That's gonna print 
out the suggestions. It's gonna print it out, display it, pretty print 
it. And let's see what happens. Okay, well, isn't that beautiful? So, 
what happened is it took the strengths and weaknesses, and 
the cost efficiency plugin is laying out, hey, cross-training 
staff would be a good idea. Many optimizations could help you. Maybe you 
should repair stuff inside the pizza shop because 
you had that flood damage. Maybe you should figure out to make 
calzones or not. So basically, what happened is it took 
all that, laid it out in another level of 
analysis for cost efficiency. I'm sure you're wondering like how did it 
do that? Isn't it amazing? Think of how much time you 
would have taken to actually sort this information in your own head. 
it just did it for you. That's cost efficiency. You 
do the same thing for time efficiency by just changing this plugin here and 
you can keep the variables like that. Don't 
worry, no one's gonna judge you. And go ahead and 
look at the result on your own. 
 
And also, go ahead and change these strengths if you want to. You'll 
just see how this is not canned, it's actually generated magic, right? Okay, 
so what did we do here? We ran the business thinking plugins, capability to 
gain cost efficiencies. You've tried out 
the time efficiency. Hopefully it's a delicious meal. And now, we're going 
to actually look at another kind of business lens using strategy, 
because that's how we roll here. So, let's do that. So, we're 
going to do is we're going to again, use the plugin and the 
plugin is imported from the directory. We're gonna use the business thinking plugin. 
And remember that if you save or change your "skprompt.txt" 
or "config.json", make sure you reload the plugin. Otherwise, it'll 
use the old one. Makes sense, right? You have to 
kind of reset. It's like pulling the plug out of 
your wall or turning your computer on and off. And I'm going 
to source into the context strengths and weaknesses. 
 
 
I forgot that in the previous cell, I need to include 
the opportunities and threats. Let's put them here. I 
have opportunities and threats. I have pulled in the 
business thinking plugin. I have set up a 
pretty decent context that uses the entire SWOT. You 
might feel a bit excited that something magical 
is gonna happen because this stuff is like, it's just 
so strange. I mean, yeah, I want it to figure out an 
entire business strategy. I'm gonna use the kernel to run the plugin. Basic strategy. 
You have to remember that this could, this would usually 
cost you money, But here, we are taking solid business thinking and 
placing it inside the plugin and let me 
give you the plated form. I'm going to give you the plated form of 
the output. The basic strategies take in all this input, 
and it will now, let's cook this up. And again, I like to use this 
display because it makes these beautiful tables. Okay. 
So, what is it telling us? It's telling us a lot. 
 
This is how you build on your strengths. This is how you 
can grow your revenue by taking advantage of 
opportunities. This is how to be resilient. 
If you notice, we had that challenge 
about the cheese and the rising cost of cheese. So, 
suggesting that you experiment with alternative cheese options, 
isn't that smart? Or you're going to have street construction 
impacting foot traffic. Well, you should 
increase your marketing. So, this analysis 
here is extraordinary, it's of extremely high quality and you 
just have to change the SWOT to adapt it to a different 
business. If you can go back and change 
interview questions, you can change the actual SWOT results. It's very fluid, 
this kind of AI. And if it gets confusing, don't forget, let's make it 
simpler for you. 
There are really only two problems that a 
business is dealing with, there are two buckets, 
a bucket of time and a bucket of money. They're losing money, they're losing 
time. It's why cost efficiency, time efficiency is 
so important. You remember those plugins you saw above. And also, 
being able to see the forest from the trees is what business 
strategies is about. And now you can see that 
you could apply this method to different interviews and 
generate incredible advice. That is the 
power of this new kind of AI. And if you 
think about it, there's an easier way to do all this if 
you don't want to live in just Jupyter Notebooks. There 
is a Visual Studio Code extension for Semantic Kernel that 
lets you do all of this prompt tuning from 
your own IDE. Okay, so we did a lot 
of code stuff. I showed you how to do it in different ways. 
We like to show you different ways because it 
gives you a choice, but it's also how you achieve scale. 
You achieve scale through knowing different ways, easy, hard, 
more powerful, less powerful. And now, let's move 
into design thinking. 


"""

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


