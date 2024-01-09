# Databricks notebook source
# MAGIC %md
# MAGIC Retrieval Augmented Generation, or RAG, has become a key 
# MAGIC method for getting LLMs to answer questions over a user's 
# MAGIC own data. But to actually build and productionize a high-quality RAG 
# MAGIC system, it costs a lot to have effective retrieval techniques, to 
# MAGIC give the LLM highly relevant context to 
# MAGIC generate his answer, and also to have an effective evaluation framework 
# MAGIC to help you efficiently iterate and 
# MAGIC improve your RAG system, both during initial development and 
# MAGIC during post-deployment maintenance. This course covers 
# MAGIC two advanced retrieval methods, sentence window retrieval 
# MAGIC and auto-merging retrieval, that deliver a significantly 
# MAGIC better context of the LLM than simpler methods. 
# MAGIC It also covers how to evaluate your LLM 
# MAGIC question-answering system with three evaluation metrics, context relevance, 
# MAGIC groundedness, and answer relevance. I'm excited 
# MAGIC to introduce Jerry Liu, co-founder and CEO of LlamaIndex, and 
# MAGIC Anupam Datta, co-founder and Chief Scientist of TruEra. For a 
# MAGIC long time, I've enjoyed following Jerry and LLamaIndex on social media 
# MAGIC and getting tips on evolving RAG 
# MAGIC practices. So I'm looking forward to him teaching this body of knowledge 
# MAGIC more systematically here. And Anupam has been a professor 
# MAGIC at CMU and has done research for over 
# MAGIC a decade on trustworthy AI and how to 
# MAGIC monitor, evaluate, and optimize AI app effectiveness. 
# MAGIC Thanks, Andrew. It's great to be here. 
# MAGIC Great to be with you, Andrew. 
# MAGIC Sentence window retrieval gives an LLM better context by 
# MAGIC retrieving not just the most relevant sentence, 
# MAGIC but the window of sentences that occur before 
# MAGIC and after it in the document. Auto-merging retrieval organizes the document 
# MAGIC into a tree-like structure where each parent node's text is 
# MAGIC divided among its child nodes. When meta child nodes are identified as relevant 
# MAGIC to a user's question, then the entire text of the parent node is 
# MAGIC provided as context for the LLM. I know this sounds 
# MAGIC like a lot of steps, but don't worry, we'll go over it in detail 
# MAGIC on code later. But the main takeaway is that this provides a way to dynamically 
# MAGIC retrieve more coherent chunks of text than simpler methods. 
# MAGIC To evaluate RAG-based LLM apps, the RAG triad, a 
# MAGIC triad of metrics for the three main steps of a RAG's execution, 
# MAGIC is quite effective. For example, we'll cover in detail how to compute 
# MAGIC context relevance, which measures how relevant the retrieved chunks 
# MAGIC of text are to the user's question. This helps you identify 
# MAGIC and debug possible issues with how your system is 
# MAGIC retrieving context for the LLM in the QA system. 
# MAGIC But that's only part of the overall QA system. We'll also cover additional evaluation 
# MAGIC metrics such as groundedness and answer relevance 
# MAGIC that let you systematically analyze what parts of your 
# MAGIC system are or are not yet working well so 
# MAGIC that you can go in in a targeted way to improve whatever 
# MAGIC part needs the most work. If you're familiar with the concept of error 
# MAGIC analysis and machine learning, this has similarities. And I've found that 
# MAGIC taking this sort of systematic approach helps you be 
# MAGIC much more efficient in building a reliable QA 
# MAGIC system. 
# MAGIC The goal of this course is to help you build production-ready, 
# MAGIC write-based LLM apps. And important parts of getting production 
# MAGIC ready is to iterate in a systematic way 
# MAGIC on the system. In the later half of this course, you gain 
# MAGIC hands-on practice iterating using these retrieval 
# MAGIC methods and evaluation methods. And you also 
# MAGIC see how to use systematic experiment tracking to 
# MAGIC establish a baseline and then quickly improve on 
# MAGIC that. 
# MAGIC We'll also share some suggestions for tuning 
# MAGIC these two retrieval methods based on 
# MAGIC our experience assisting partners who are 
# MAGIC building RAG apps. 
# MAGIC Many people have worked to create this course. I'd 
# MAGIC like to thank, on the LlamaIndex side, Logan 
# MAGIC Markehwich, and on the TruEra side, Shayak Sen, Joshua 
# MAGIC Reini, and Barbara Lewis. From DeepLearning.ai, Eddie Shyu 
# MAGIC and Dialla Ezzeddine also contributed to 
# MAGIC this course. 
# MAGIC The next lesson will give you an overview of what you'll 
# MAGIC see in the rest of the course. You'll try out question-answering systems that 
# MAGIC use sentence window retrieval or auto-merging 
# MAGIC retrieval and compare their performance on 
# MAGIC the RAG triad, context relevance, groundedness, and answer relevance. 
# MAGIC Sounds great. Let's get started. And I think you people 
# MAGIC are really clean up with this RAG stuff. 
