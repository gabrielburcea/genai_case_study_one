# Databricks notebook source
"""
Cooking up flavorful SWOTs with the kernel
"""

# COMMAND ----------

"""
Get a kernel ready
"""

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

"""
Let's make a semantic function
"""

sk_prompt = """
{{$input}}

Summarize the content above in less than 140 characters.
"""
summary_function = kernel.create_semantic_function(prompt_template = sk_prompt,
                                                    description="Summarizes the input to length of an old tweet.",
                                                    max_tokens=200,
                                                    temperature=0.1,
                                                    top_p=0.5)       
print("A semantic function for summarization has been registered.");

# COMMAND ----------

sk_input = """
Let me illustrate an example. Many weekends, I drive a few minutes from my house to a local pizza store to buy 
a slice of Hawaiian pizza from the gentleman that owns this pizza store. And his pizza is great, but he always 
has a lot of cold pizzas sitting around, and every weekend some different flavor of pizza is out of stock. 
But when I watch him operate his store, I get excited, because by selling pizza, he is generating data. 
And this is data that he can take advantage of if he had access to AI.

AI systems are good at spotting patterns when given access to the right data, and perhaps an AI system could spot 
if Mediterranean pizzas sell really well on a Friday night, maybe it could suggest to him to make more of it on a 
Friday afternoon. Now you might say to me, "Hey, Andrew, this is a small pizza store. What's the big deal?" And I 
say, to the gentleman that owns this pizza store, something that could help him improve his revenues by a few 
thousand dollars a year, that will be a huge deal to him.
""";
# Text source: https://www.ted.com/talks/andrew_ng_how_ai_could_empower_any_business/transcript

summary_result = await kernel.run_async(summary_function, input_str=sk_input)

display(Markdown("### ✨ " + str(summary_result)))

# COMMAND ----------

summary_result = summary_function(sk_input)

display(Markdown("### ✨ " + str(summary_result)))

# COMMAND ----------

"""
Native functions
"""

from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)

class ExoticLanguagePlugin:
    def word_to_pig_latin(self, word):
        vowels = "AEIOUaeiou"
        if word[0] in vowels:
            return word + "way"
        for idx, letter in enumerate(word):
            if letter in vowels:
                break
        else:
            return word + "ay"
        return word[idx:] + word[:idx] + "ay"
    @sk_function(
        description="Takes text and converts it to pig latin",
        name="pig_latin",
        input_description="The text to convert to pig latin",
    )
    def pig_latin(self, sentence:str) -> str:
        words = sentence.split()
        pig_latin_words = []
        for word in words:
            pig_latin_words.append(self.word_to_pig_latin(word))
        return ' '.join(pig_latin_words)

exotic_language_plugin = kernel.import_skill(ExoticLanguagePlugin(), skill_name="exotic_language_plugin")
pig_latin_function = exotic_language_plugin["pig_latin"]

print("this is kind of not going to feel awesome but know this is a big deal")


# COMMAND ----------

final_result = await kernel.run_async(summary_function, pig_latin_function, input_str=sk_input) 

display(Markdown("### ✨ " + str(final_result)))

# COMMAND ----------

"""Strengths
What unique recipes or ingredients does the pizza shop use?
What are the skills and experience of the staff?
Does the pizza shop have a strong reputation in the local area?
Are there any unique features of the shop or its location that attract customers?
Weaknesses
What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)
Are there financial constraints that limit growth or improvements?
Are there any gaps in the product offering?
Are there customer complaints or negative reviews that need to be addressed?
Opportunities
Is there potential for new products or services (e.g., catering, delivery)?
Are there under-served customer segments or market areas?
Can new technologies or systems enhance the business operations?
Are there partnerships or local events that can be leveraged for marketing?
Threats
Who are the major competitors and what are they offering?
Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?
Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?
Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?
Strengths	Weaknesses
Unique garlic pizza recipe that wins top awards	High staff turnover
Owner trained in Sicily at some of the best pizzerias	Floods in the area damaged the seating areas that are in need of repair
Strong local reputation	Absence of popular calzones from menu
Prime location on university campus	Negative reviews from younger demographic for lack of hip ingredients
Meanwhile there's money being left on the table (their opportunities) with calamities (their threats) waiting in the wings to possibly happen and may knock their pizza shop out cold.

🔖 Opportunities and Threats
Opportunities	Threats
Untapped catering potential	Rising competition from cheaper pizza businesses nearby
Growing local tech startup community	There's nearby street construction that will impact foot traffic
Unexplored online presence and order capabilities	Rising cost of cheese
Upcoming annual food fair	No immediate local regulatory changes but it's election season

"""

# COMMAND ----------

"""🤷 What does this have to do with LLMs and Semantic Kernel?"""

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
print("Made a kernel!")

# COMMAND ----------

swot_interview= """
1. **Strengths**
    - What unique recipes or ingredients does the pizza shop use?
    - What are the skills and experience of the staff?
    - Does the pizza shop have a strong reputation in the local area?
    - Are there any unique features of the shop or its location that attract customers?
2. **Weaknesses**
    - What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)
    - Are there financial constraints that limit growth or improvements?
    - Are there any gaps in the product offering?
    - Are there customer complaints or negative reviews that need to be addressed?
3. **Opportunities**
    - Is there potential for new products or services (e.g., catering, delivery)?
    - Are there under-served customer segments or market areas?
    - Can new technologies or systems enhance the business operations?
    - Are there partnerships or local events that can be leveraged for marketing?
4. **Threats**
    - Who are the major competitors and what are they offering?
    - Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?
    - Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?
    - Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"""


sk_prompt = """
{{$input}}

Convert the analysis provided above to the business domain of {{$domain}}.
"""
shift_domain_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                    description="Translate an idea to another domain.",
                                                    max_tokens=1000,
                                                    temperature=0.1,
                                                    top_p=0.5)
my_context = kernel.create_new_context()

my_context['input'] = swot_interview
my_context['domain'] = "construction management"

result = await kernel.run_async(shift_domain_function, input_context=my_context)

display(Markdown(f"### ✨ Shift the SWOT interview questions to the world of {my_context['domain']}\n"+ str(result)))

# COMMAND ----------

sk_prompt = """
{{$input}}

Rewrite the text above to be understood by a {{$level}}.
"""
shift_reading_level_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                    description="Change the reading level of a given text.",
                                                    max_tokens=1000,
                                                    temperature=0.1,
                                                    top_p=0.5)

my_context['input'] = swot_interview
my_context['domain'] = "construction management"
my_context["level"] = "child"

result = await kernel.run_async(shift_domain_function, shift_reading_level_function, input_context=my_context)

display(Markdown(f"### ✨ Shift the SWOT interview questions to the world of {my_context['domain']} at the level of {my_context['level']}\n"+ str(result)))

# COMMAND ----------

"""
So, the stove's in your kitchen. You can finally start cooking. Last 
section felt a little bit boring, a little too much blah blah blah, but 
you have the oven, the stove, the big object ready to cook your AI. What 
are you going to cook on this stove? You're going 
to cook functions, semantic functions. What kind 
of cooking are you going to do? Business style 
stuff. We're going to cook up. SWOTs, strength, weakness, opportunity, 
threats, boring business stuff, and make it 
sing on your stovetop with AI. Let's see it happen. 
 
Semantic functions, functions made out of words, so-called 
prompts, and basically do things that you might not expect are 
possible, which is scale the kind of AI you're trying 
to build. Not just scale the number of 
people that can use it but scale the kind of system you 
can build with it. There are two kinds of functions in the 
semantic kernel world. One is a native function, classic computer 
code, you type it in, and it runs, or semantic 
functions composed of text that is 
parameterized. there's some kind of configuration attached to 
it, and it produces output as well as a 
native function, but you'll find that these semantic functions are usually 
powerful. Reminder what's a SWOT. A SWOT is strength, weaknesses, 
opportunities and threats. It's a good matrix to use 
when helping a business of any scale understand what it should be 
worried about. 
Okay we're going to get a kernel ready, and 
I'm going to grab from the semantic kernel as SK. You'll see me 
type this a lot, and you'll get bored very quickly. So, let me just copy 
and paste a bunch of stuff, and we're going to run that. 
The kernel is now ready. Next up we're going to do is we're 
going to make our first semantic function. Okay, let's do 
this. So, what we're going to do is we're going to make 
a prompt, a templated prompt. Let's read this for a second. It's 
a Python string so it's a multi-line string. It has a 
certain input, and the prompt is to summarize whatever is above in less than 140 characters. 
 
 
We're then going to take a detour to make 
a semantic function that takes this prompt template. There 
are some other parameters. That's boilerplate. Let me 
just add that for a second. There we go. I'm adding a 
description. I am telling how many tokens to use, setting the temperature, 
and giving the top p to describe the 
range of words to be used in the response, the completion response. 
And just a reminder, because this isn't 
going to run, and you won't see anything happen. Let me just remind myself 
that what it's doing is it's making a semantic function for summarization 
and now it's registered. 
 
So next up, if you notice this value input, 
I want to be able to attach some value to that. And I'm 
going to take text from a TED talk from Andrew Ng about 
a pizza shop owner who could use some 
help from AI, he thinks. And I'm going to then compute the following. I'm gonna 
generate a result by having the kernel do a 
run async. This is now adding the summary function into 
the pipeline. And I'm gonna give it at the head of the 
pipeline, SK input. And let's print this out in 
a pretty way. I'm gonna use my display markdown thing. And let's see how it does. 
 
Let's see how it's doing. It's running, great. It took this 
information, and it summarized it. What did it 
do? It basically took this summary function, it stuffed this 
input into the summary function, and then it 
generated, Sparkle, this short summary of that long text. You know, long 
text, nothing's wrong with it, long form is good. Okay, how are you 
feeling so far? Kind of cool, right? Let me just tell 
you there's another shorthand way of calling these different functions without using 
the kernel. In this case, I'm just 
going to call the function directly, just kind of like calling a 
procedure. You'll see this used once in a while by people. They want 
to show that you don't have to use it with an explicit 
kernel. And there it is. It did it the same way, but this 
is much more compact. 
It doesn't using this await kernel runs async, but 
does the same thing. Same result. See that? Okay. High 
feeling. Okay, we're going to go a little harder now in the 
world of native functions. Native functions. Native functions 
are not semantic functions, and they 
require a lot of so-called syntactic sugar meaning they they're 
wrapped in a bunch of things that I could type but 
pardon me for typing fast meaning cut and 
paste let's pull these things in we're going to make an exotic 
language plugin it's going to convert anything into. 
 
Pig Latin as you can see what it's doing is I define 
a semantic kernel function I give it a 
description I give it a name. I tell it what is this input variable, you 
know this input like what do I do with the 
input what is it. What do I do? Do this. What is it? It's this 
kind of text. It takes in Pig Latin, sentence comes 
in, it walks over the words, and of 
course it processes it. So here I'm going to 
say exotic language plugin, I'm going to register 
it in the kernel. I'm going to register this plugin. See that? A lot of 
syntax, right? But wait for it. This is the exciting part. 
I'm going to define the function. This is kind of, what's it 
called again? This is not going to feel awesome but know this is 
a big deal. 
You're taking a native function, and you're wrapping it in a format that 
Semantic Kernel can use. In the same way that Semantic Kernel can 
process semantic functions, it can 
process native functions. And next up, what we want to do is 
we want to run a pipeline of functions, The summary function and the 
Pig Latin function with the input. So, the Andrew 
Ing long form will go into summary function. It'll be fed 
as an output into the Pig Latin function, and guess what's 
going to happen? I'll assume you'll guess what's going to happen, but yeah. It's 
basically that sentence before, AI can blah, blah, blah, turned into Pig 
Latin, okay. 
Clap, clap, clap, applause, AI, but now you can see that it's 
using a native function, and a semantic function so 
you can interchange them pretty cool and you know, and 
all this like techno stuff you can get lost, I 
get lost pretty quickly myself, because it's so powerful. But end 
of the day, all you're doing is writing code to solve a problem, so the 
TED talk that Andrew Ing gave about how to 
help small businesses in particular a pizza shop 
owner struck a chord with me, and for that 
reason I have built this entire lesson for 
you to not just write code, chord with me, 
and for that reason I have built this entire lesson for 
you to not just write code, but think in 
business terms. 
So, let's sort of take off by, first of all, thinking 
of the SWOT instrument. To create a SWOT, you 
have to create a list of questions. And this is a fairly good 
list of questions that are composed to figure out, like 
a small business owner, like what are their strengths, weaks, 
opportunities, and threats? And if you organize these in a SWOT chart, 
the way they look is this. Let me show you this. I love a two-by-two, don't 
you? So, these are the strengths. Look at them quickly. 
Unique garlic pizza recipe, owner trained in Sicily, 
weaknesses, don't have calzones, had a flood in the 
area, the seating areas are in disrepair, opportunities, they don't do catering 
or cater to the local tech startup community. 
There's also an annual food fair. This is a way they could actually 
make new revenue. 
And there's threats. There are cheaper pizza shops, there's 
gonna be street construction, the cost 
of cheese is going up. And so, this is a snapshot, a swap. So, what 
does this have to do with large language models 
and semantic kernel? Good question. Well, first off, remember, it 
is a way to solve AI problems, semantic kernel, 
but it doesn't matter if you don't have a good 
problem. Now, let's make a kernel again. I'm gonna make a kernel. This 
is very familiar to you. And again, I just put print 
statements in to just remind myself. Made 
a kernel. I made a kernel. Well, and now what I want 
to do is I want to take these SWOT responses, analyses, 
and. I want to convert them to a different domain. Why 
would I want to do that? Well, because as they say, the mountain 
says you can, and AI gives you the ability 
to climb mountains so quickly. It's unbelievable. 
 
So, let me show how that works. So, what I've done here is I've taken 
the SWAT interview and I'm making a semantic function that converts the analysis 
into a different business domain. Kind of weird, 
huh? And what I'm going to do is I'm going to take this 
interview text, I'm going to ask it to apply it to the construction management domain. And 
I'm going to run it. And what happens is it takes questions 
that were geared for a pizza shop, and 
now I could give them to someone in construction management. Which sounds 
kind of science fiction-y, but 
here, it's changed all the pizza-esque things into construction methods, 
materials, budget overruns, do you see the zoning regulations? And 
so, it basically like shifted the context of the 
text. And that is something you cannot write in a native function, but in 
a semantic function, it's unusually easy to do. 
 
Let's throw in another, cause we like to chain things together. Let 
me show you another type of semantic function. This 
is a rewrite. Rewrite the text above to be 
understand by a blank level. So, what we're doing is we're making a 
new function to be able to change the reading level of whatever 
comes into it. And what I'm gonna do is I'm gonna redo the shift 
domain function, shift the questions into another domain, construction management. 
I'm going to flow that in 
the pipeline to changing the reading level of 
whatever comes into it to the level of 
a child. It's a lot easier to read the good things, the bad things, the good chances, the 
bad chances and there you have it. And I'd like you to change any 
of these parameters and see what happens. And I think 
you'll get pretty excited because I don't think you've 
ever written programs like this. 
 
If you're an LLM person or ML person in general, this 
is no surprise to you. But for someone working 
primarily in the domain of problems, it's extraordinary. And you 
want to remember that we are using this, the right 
hand, I talked about the semantic completion ability, fill in 
the blanks. This is the magic that we're experiencing right 
now. But there's this other type of AI capability that 
we're using out there called semantic similarity. I haven't touched upon 
that yet, so just be ready for when 
you notice that you can use both hands 
in the equation. 

"""
