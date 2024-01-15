# Databricks notebook source
'''Welcome to Reinforcement Learning from Human Feedback, 
or RLHF, built in partnership with Google Cloud. 
An LLM trained from public internet data would 
mirror the tone of the internet, so it can generate 
information that is harmful, false, or unhelpful. 
RLHF is an important tuning technique that has 
been critical to align an LLM's output with human preferences and values. 
This algorithm is, I think, a big deal and has been a 
central part to the rise of LLMs. And it turns out that 
ROHF can be useful to you, even if you're 
not training an LLM from scratch, but instead building an 
application whose values you want to set. 
While fine-tuning could be one way to do this, as 
you learn in this course, for many cases, RLHF 
can be more efficient. 
For example, there are many valid ways 
in which an LLM can respond to a prompt such as, 
what is the capital of France? 
It could reply with, Paris is the capital of France, 
or it could even more simply reply, Paris. 
Some of these responses were few more natural than others. 
And so, RROHF is a method for gathering 
human feedback on which responses they 
prefer in order to train the model to generate 
more responses that humans prefer. 
In this process, you start off with an LLM that's 
already been trained with instruction tuning, so 
it's already learned to follow instructions. 
You then gather a dataset that indicates a human label's 
preferences between multiple completions of the 
same prompt, and use this dataset as 
a reward signal, or to create a reward signal, to 
fine-tune an instruction an instruction tuned LLM. 
The result is a tuned large language model that 
generates completions or outputs that better aligns with the 
preferences of the human labelers. 
I am delighted to introduce the instructor, Nikita Namjishi, 
who is developer advocate for Gent of AI on Google Cloud. 
She is a regular speaker at Gen2AI developer events 
and has helped many people build Gen2AI applications. 
I look forward to her sharing her deep experience, 
her deep practical experience with Gen2AI and with 
ROHF with us here. 
Thank you, Andrew. 
I'm really excited to work with you and your team on this. 
In this course, you learn about the RLHF 
process and also gain hands-on practice exploring sample data 
sets for RLHF, tuning the LLAMA2 model using RLHF, and 
then also evaluating the newly tuned model. 
Nikita will go through these concepts using Google Cloud's 
Machine Learning Platform, Vertex AI. 
What really excites me about RLHF is that it helps 
us to improve an LLM's ability to solve tasks where the 
desired output is difficult to explain or describe. 
In other words, problems where there's no single correct answer. 
And in a lot of problems we naturally want to use LLMs for, 
there really is no one correct answer. 
It's such an interesting way of thinking about training 
machine learning models, and it's different 
from supervised fine-tuning, which you may 
already be familiar with. 
RLHF doesn't solve all of the problems of truthfulness and toxicity 
in large language models, but it's really been a key part 
of improving the quality of these models. And I 
think we're going to continue to see more 
techniques like this in the future as the 
field evolves. 
So, I'm really, really excited to share with you 
just how it works. And I'm happy to say you don't need to 
know any reinforcement learning to get started. 
Many people have worked to create this course. 
I'd like to thank, on the Google Cloud side, Bethany 
Wang, Mei Hu, and Jarek Kazmierczak. From 
DeepLighting.ai, Eddie Xu and Leslie Zerma had also contributed 
to this course. 
So with that, let's go on to the next video where Nikita 
will present an overview of RHF so you can see all the 
pieces of how it works and how they fit together. Let's go 
on to the next video. '''
