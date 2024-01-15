# Databricks notebook source
'''RAG, or Retrieval Augmented Generation, 
retrieves relevant documents to give context to an LLM, 
and this makes it much better at answering 
queries and performing tasks. 
Many teams are using simple retrieval 
techniques based on semantic similarity or embeddings, 
but you learn more sophisticated techniques in this 
course, which will let you do much better than that. 
A common workflow in RAG is to take your query and embed that, 
then find the most similar documents, meaning ones with similar embeddings, 
and that's the context. But the problem with that 
is that it can tend to find documents that talk about similar 
topics as a query, but not actually contain the answer. But 
you can take the initial user query and rewrite. 
This is called query expansion. 
Rewrite it to pull in more directly related documents. 
Two key related techniques. One, to expand the optional 
query into multiple queries by rewording or rewriting 
it in different ways. And second, to even guess or hypothesize what 
the answer might look like to see if we can find anything 
in our document collection that looks more like an 
answer rather than only generally talking 
about the topics of the query. 
I'm delighted that your instructor for this course is Anton 
Troynikov. 
Anton has been one of the innovators driving forward 
the state-of-the-art and retrieval for AI applications. 
He is co-founder of Chroma, which provides one of the 
most popular open-source vector databases. 
If you've taken one of our LangChain short courses taught 
by Harrison Chase, you have very likely used Chroma. 
Thank you, Andrew. 
I'm really excited to be working with you on this course 
and share what I'm seeing out in the field in terms 
of what does and doesn't work in RAG deployments. 
We'll start off the course by doing a quick 
review of RAG applications. 
You will then learn about some of the 
pitfalls of retrieval where simple vector search doesn't 
do well. 
Then you'll learn several methods to improve the results. 
As Andrew mentioned, the first methods use an 
LLM to improve the query itself. 
Another method re-ranks query results with help from something 
called a cross encoder, which takes in a pair 
of sentences and produces a relevancy score. 
You'll also learn how to adapt the query embeddings based on user feedback 
to produce more relevant results. 
There's a lot of innovation going on in RAG right now. So 
in the final lesson, we'll also go over some of the cutting edge techniques 
that aren't mainstream yet and are only just now appearing 
in research. 
And I think they'll become much more mainstream soon. 
We'd like to acknowledge some of the folks that 
have worked on this course. 
From the Chroma team, we'd like to thank Jeff Huber, Hammad 
Bashir, Liquan Pei, and Ben Eggers, as well as Chroma's open-source 
developer community. 
From the Deep Learning team, we have Geoff Ladwig and Esmael 
Gargari. 
The first lesson starts with an overview of Rack. 
I hope you go on to watch that right after this. 
And with these techniques, it turns out it's possible for smaller 
teams than ever to build effective systems. 
So after this course, you might be able 
to build something really cool with an approach 
that previously would have been considered RAG tag. '''
