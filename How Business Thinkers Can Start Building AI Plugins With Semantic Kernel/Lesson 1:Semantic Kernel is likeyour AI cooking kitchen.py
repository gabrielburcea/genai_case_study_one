# Databricks notebook source
"""
Semantic Kernel is likeyour AI cooking kitchen
"""

# COMMAND ----------

"""
Get a kernel ready
"""

# COMMAND ----------

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

print("You made a kernel!")

# COMMAND ----------

import semantic_kernel as sk
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion

kernel = sk.Kernel()

kernel.add_text_completion_service("huggingface", HuggingFaceTextCompletion("gpt2", task="text-generation"))

print("You made an open source kernel using an open source AI model!")

# COMMAND ----------

"""
Reminder: A kernel is like the stove in a kitchen. It doesn't do anything unless you start to cook with it. Let's proceed to get it a few functions to 🔥 cook up for us.
"""

# COMMAND ----------

"""
So there's a ton of AI content out there. 
You're hoping this is the one that's gonna help you figure this stuff out. 
You know it has to be hands-on, otherwise it's just blah blah blah. 
That's why we're here. 
Symantec Kernel has been made for large-scale enterprises, but it's also 
available to you. 
Someone who's trying to figure things out. 
and we're going to jump into it in the spirit of cooking 
because as you know you can't cook if you don't have a kitchen. 
Spider-Kernel is a kitchen for your AI recipes so you might be able 
to make meals for yourself, 
for your friends, your family, for customers that will 
you know make your life a little bit better. 
So let's jump in, get started, come along. 
So welcome to your kitchen. 
Well we're not going to get too deep in the kitchen yet, so 
sit tight. 
We're going to do a quick overview, kind of like when you buy a big machine, 
you get it with a manual. 
As a manual, we want to help you understand what Symantec Kernel means. 
Number one, it's a kernel. 
What is a kernel? 
A kernel is something that is at the 
center of a computational system that is really important. 
So the kernel is the essential glue, the orchestrator of different tasks. 
So super important. 
Secondly, it's semantic. 
Why is it semantic? 
Why did we use that word? 
It's because Sam Scalace, the person who got this all started, 
said this kind of computation that uses AI models and memory models 
that use language is inherently semantic, 
not syntactic, not brittle, more flexible. 
So semantic was a word we used for semantic kernel. 
And it's got a job. 
The job is to be your amazing kitchen 
to take any kind of sort of test kitchen recipes and bring 
it all the way to production. 
Not just production to serve like five people at your house, 
but to serve five million people all over the world. 
So let's jump into Notebook. 
So every Notebook is a bit daunting but as you know, 
you want to bring in things into the 
world as import and you give it a short name. 
I'm going to import semantic kernel. 
I'm going to make a kernel and then I'm going 
to want to connect the kernel with some model. 
I'm use this syntax because I'm calling the OpenAI settings from 
dotenv and I'm going to add 
to the kernel a text completion service that we're 
going to give it a label OpenAI. 
We're going to make sure we clarify that it is a. OpenAI 
chat completion. 
We're gonna choose a model. 
I am in love with GPT 3.5 Turbo 0301. 
Why that one? 
Because it's like a good mushroom that you find that works really 
well for your cooking and you stay with it. 
So 0301, tastes great. 
And again, when you're using this in whatever year 
you see this course, probably change the name, change, but 
in essence, this is what you're doing. 
you're going to grab Symantec kernel bindings, make 
a kernel, grab an API key from the.env file 
and add a completion service. 
Let us plate that, let's pull out a plate here 
and fully finished plate here. 
What's different is you might be using Azure OpenAI, so 
there's a flag here. 
Everything we do when we make a kernel is gonna look like 
this. 
So let's say, let's verify, and we made a kernel. 
Let's run this. 
Awesome, you have a kernel. 
How's that feel? 
Okay, that was easy, right? 
Now you might be thinking, wait, Symantec Kernel only uses 
OpenAI or Azure OpenAI. 
I can't use this. 
I love open source. 
Well, Symantec Kernel is open source. 
So let's think about that, right? 
Because it's open source, you might imagine it has connectors to. 
Hugging Face, everyone's favorite TV show for AI models. 
And of course, what I can do is I can make a kernel and I can, 
wait for it, add a text completion service as well. 
I'm gonna add a text completion service. 
It's a HuggingFace one. This is a label, mind you. 
And I'm gonna add a HuggingFace text completion. 
Oh, sorry, GPT-2, right? 
Don't forget, open source lags, the closed model world, so. 
But isn't it wonderful to use a free model? 
Okay, got my parens. 
and you made an open-source kernel using an open-source AI 
model, whoa. 
And let's see here, this has to have a few libraries loaded, of course, but 
I think we have them pre-loaded, 
and ta-da, so there you have a kernel that uses OpenAI, or 
Azure OpenAI, or a Hugging Face completion model. 
Okay, that is fairly pain-free, hopefully, for you. 
One reminder that this is all open source, so if there's 
any code you're using that you want to learn 
more about it, just go to the Semantic Kernel repo. 
It's on GitHub, where all this kind of cool AI 
code seems to hang out. 
Let's move on to the next section, because you made it through 
the first section. 
Congratulations! 
Let's go! 


"""


