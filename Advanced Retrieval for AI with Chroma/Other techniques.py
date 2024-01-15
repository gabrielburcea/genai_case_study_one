# Databricks notebook source
'''Embeddings-based retrieval is still a very 
active area of research, and there's a 
lot of other techniques that you should be aware of. 
For example, you can fine-tune the embedding model 
directly using the same type of data as we used in the 
embeddings adapters lab. 
Additionally, recently there's been some really good 
results published in fine-tuning the LLM itself to expect 
retrieved results and reason about them. 
You can see some of the papers highlighted here. 
Additionally, you could experiment with a more 
complicated embedding adapter model using a full-blown 
neural network or even a transformer layer. 
Similarly, you can use a more complex 
relevance modeling model rather than just using the cross-encoder 
as we described in the lab. And finally, an often overlooked piece 
is that the quality of retrieved results often depends 
on the way that your data is chunked before it's stored 
in the retrieval system itself. 
There's a lot of experimentation going on 
right now about using deep models including transformers for 
optimal and intelligent chunking. And 
that wraps up the course. 
In this course, we covered the basics of retrieval 
augmented generation using embeddings-based retrieval. 
We looked at how we can use LLMs to augment and enhance our queries to 
produce better retrieval results. 
We looked at how we can use a cross-encoder model for re-ranking to 
score the retrieved results for relevancy. 
And we looked at how we can train an 
embedding adapter using data from human 
feedback about relevancy to improve our query results. 
Finally, we covered some of the most exciting work that's 
ongoing right now in the research literature around improving 
retrieval for AI applications. 
Thanks for joining the course and we're really looking forward to 
seeing what you'll build. '''
