# Databricks notebook source
"""Lesson 2: RAG Triad of metrics"""

import warnings
warnings.filterwarnings('ignore')

import utils
​
import os
import openai
openai.api_key = utils.get_openai_api_key()

from trulens_eval import Tru
​
tru = Tru()
tru.reset_database()

from llama_index import SimpleDirectoryReader
​
documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

from llama_index import Document
​
document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))

from utils import build_sentence_window_index
​
from llama_index.llms import OpenAI
​
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
​
sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

from utils import get_sentence_window_query_engine
​
sentence_window_engine = \
get_sentence_window_query_engine(sentence_index)

output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response
Feedback functions

import nest_asyncio
​
nest_asyncio.apply()

from trulens_eval import OpenAI as fOpenAI
​
provider = fOpenAI()
1. Answer Relevance

from trulens_eval import Feedback
​
f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()
2. Context Relevance

from trulens_eval import TruLlama
​
context_selection = TruLlama.select_source_nodes().node.text

import numpy as np
​
f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

import numpy as np
​
f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)
3. Groundedness

from trulens_eval.feedback import Groundedness
​
grounded = Groundedness(groundedness_provider=provider)

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
Evaluation of the RAG application

from trulens_eval import TruLlama
from trulens_eval import FeedbackMode
​
tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)

eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

eval_questions

eval_questions.append("How can I be successful in AI?")

eval_questions

for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()

import pandas as pd
​
pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]

tru.get_leaderboard(app_ids=[])

tru.run_dashboard()


# COMMAND ----------

# MAGIC %md
# MAGIC In this lesson, we do a deep dive into evaluation. 
# MAGIC We'll walk you through some core concepts on how 
# MAGIC to evaluate RAG systems. Specifically, we 
# MAGIC will introduce the RAG triad, a triad of metrics for the three 
# MAGIC main steps of a RAG's execution. Context relevance, groundedness, and 
# MAGIC answer relevance. These are examples of an extensible framework of 
# MAGIC feedback functions. Programmatic evaluations of LLM apps. We then 
# MAGIC show you how to synthetically generate an evaluation 
# MAGIC data set, given any unstructured corpus. Let's get started. 
# MAGIC Now I'll use a notebook to walk you through the 
# MAGIC RAG triad, answer relevance, context relevance, and groundedness 
# MAGIC to understand how each can 
# MAGIC be used with truelens to detect hallucinations. At this 
# MAGIC point, you have already PIP 
# MAGIC installed TruLens, eval and Llama Index. So I'll not show you that 
# MAGIC step. The first step for you will be to set up an OpenAI 
# MAGIC API key. The OpenAI key is used for the completion step of the 
# MAGIC RAG and to implement the evaluations with TruLens. So 
# MAGIC here's a code snippet that does 
# MAGIC exactly that. And you're now all set up with the 
# MAGIC OpenAI key. The next section, I will quickly recap the query engine construction with Llama 
# MAGIC Index. Jerry has already walked you 
# MAGIC through that in lesson one in 
# MAGIC some detail. We will largely build on that lesson. 
# MAGIC The first step now is to set up a true object. From 
# MAGIC TruLens eval, we are going to import the true 
# MAGIC class. Then we'll set up a true object and instance of this class. And then 
# MAGIC this object will be used to reset the database. This database 
# MAGIC will be used later on to record the prompts, responses, intermediate 
# MAGIC results of the Llama Index app, as well as the 
# MAGIC results of the various evaluations we will be setting up with TrueLens. 
# MAGIC Now let's set up the Llama Index reader. So this 
# MAGIC snippet of code reads this PDF document from a directory on 
# MAGIC how to build a career in AI written 
# MAGIC by Andrew Ang and then loads this data into this document object. 
# MAGIC The next step is to merge all of this 
# MAGIC content into a single large document rather than having one document per 
# MAGIC page, which is the default setup. Next, we set up 
# MAGIC the sentence index, leveraging some of the Llama Index utilities. So you 
# MAGIC can see here that we are using OpenAI GPT 3.5 Turbo set at 
# MAGIC a temperature of 0.1. As the LLM that will be used for completion of the 
# MAGIC RAG. The embedding model is set to bge small, and version 1.5. 
# MAGIC And all of this content is being indexed with the sentence index 
# MAGIC object. Next, we set up the sentence window 
# MAGIC engine. And this is the query 
# MAGIC engine that will be used later on to do retrieval effectively 
# MAGIC from this advanced RAG application. Now that we 
# MAGIC have set up the query engine for sentence-window-based RAG, 
# MAGIC let's see it in action by actually asking a specific question. 
# MAGIC How do you create your AI portfolio? This will 
# MAGIC return a full object with the 
# MAGIC final response from the LLM, the intermediate pieces of retrieved context, 
# MAGIC as well as some additional metadata. Let's take a 
# MAGIC look at what the final response 
# MAGIC looks like. So here you can see the final response that came 
# MAGIC out of this sentence window-based rag. It provides a pretty good 
# MAGIC answer on the surface to this question of how do 
# MAGIC you create your AI portfolio. Later on, we will see 
# MAGIC how to evaluate answers of this form against 
# MAGIC the RAG triad to build confidence and identify failure modes for RAGs 
# MAGIC of this form. for rags of this form. Now that we have 
# MAGIC an example of a response to this question that looks quite good on 
# MAGIC the surface, we will see how to make use of feedback functions, 
# MAGIC such as the rag triad, to evaluate this kind of response more 
# MAGIC deeply, identify failure modes, as well as build confidence or 
# MAGIC iterate to improve the LLM application. Now that 
# MAGIC we have set up the sentence-window-based RAG 
# MAGIC application, let's see how we can evaluate it 
# MAGIC with the RAG triad. We'll do a little bit of housekeeping in the 
# MAGIC beginning. First step is this, is of course snippet that lets us 
# MAGIC launch a StreamLit dashboard from inside the notebook. You'll see 
# MAGIC later that we'll make use of that dashboard to see the results of 
# MAGIC the evaluation and to run experiments, to 
# MAGIC look at different choices of apps, and to see which to look at 
# MAGIC different choices of apps and to see which one 
# MAGIC is doing better. Next up, we initialize OpenAI GPT 3.5 Turbo 
# MAGIC as the default provider for our evaluations. And 
# MAGIC this provider will be used to implement the different feedback functions or evaluations, such 
# MAGIC as context relevance, answer relevance, and groundedness. Now, 
# MAGIC let's go deeper into each of the evaluations of the 
# MAGIC RAG triad, and we'll go 
# MAGIC back and forth a bit between slides and 
# MAGIC the notebook to give you the full context. First up, we'll discuss answer relevance. 
# MAGIC Recall that answer relevance is checking whether the final response 
# MAGIC is relevant to the query that was asked by the 
# MAGIC user. To give you a concrete example 
# MAGIC of what the output of answer 
# MAGIC relevance might look like, here's an example. The user 
# MAGIC asked the question, how can altruism be beneficial in building 
# MAGIC a career? This was the response that came 
# MAGIC out of the RAG application. And the answer 
# MAGIC relevance evaluation produces two pieces of output. One is 
# MAGIC a score. On a scale of 0 to 1, the 
# MAGIC answer was assessed to be highly relevant, so it 
# MAGIC got a score of 0.9. The second 
# MAGIC piece is the supporting evidence or the rationale or the chain 
# MAGIC of thought reasoning behind why the evaluation produced 
# MAGIC this score. So here you can see that supporting evidence found in the answer 
# MAGIC itself, which indicates to the LLM evaluation that it 
# MAGIC is a meaningful and relevant answer. I also 
# MAGIC want to use this opportunity to introduce the 
# MAGIC abstraction of a feedback function. Answer relevance is 
# MAGIC a concrete example of a feedback function. More generally, a feedback 
# MAGIC function provides a score on a scale of 0 
# MAGIC to 1 after reviewing an LLM 
# MAGIC app's inputs, outputs, and intermediate results. Let's now look 
# MAGIC at the structure of feedback functions using 
# MAGIC the answer relevance feedback function as a concrete example. The first component is a 
# MAGIC provider. And in this case, we can 
# MAGIC see that we are using an LLM from OpenAI 
# MAGIC to implement these feedback functions. Note 
# MAGIC that feedback functions don't have to be implemented necessarily using 
# MAGIC LLMs. We can also use BERT models and other kinds of mechanisms to implement feedback functions 
# MAGIC that I'll talk about in some more detail later 
# MAGIC in the lesson. The second component is that 
# MAGIC leveraging that provider, we will implement a feedback 
# MAGIC function. In this case, that's the relevance feedback function. 
# MAGIC We give it a name, a human-readable name that'll be shown 
# MAGIC later in our evaluation dashboard. And for this 
# MAGIC particular feedback function, we run it on the user input, 
# MAGIC the user query, and it also takes as input the final output 
# MAGIC or response from the app. So given 
# MAGIC the user question and the final answer 
# MAGIC from the RAG, this be by function will make use 
# MAGIC of a LLM provider such as OpenAI GPT 3.5 to come 
# MAGIC up with a score for how relevant the responses to 
# MAGIC the question that was asked. And in addition, 
# MAGIC it'll also provide supporting evidence or chain of thought reasoning 
# MAGIC for the justification of that score. Let's now switch back to the notebook and 
# MAGIC look at the code in some more detail. Now let's 
# MAGIC see how to define the question-answer relevance 
# MAGIC feedback function in code. From TruLens eval, we will 
# MAGIC import the feedback class. Then we set up the different 
# MAGIC pieces of the question answer relevance function that we were just 
# MAGIC discussing. First up, we have the provider, that is 
# MAGIC OpenAI, GPT 3.5. And we set up this particular 
# MAGIC feedback function where the relevance score will also be augmented with the 
# MAGIC chain of thought reasoning, much like 
# MAGIC I showed in the slides. We give this 
# MAGIC feedback function a human-understandable name. We call it answer relevance. This will be 
# MAGIC show up later in the dashboard, 
# MAGIC making it easy for users to understand what the feedback function is 
# MAGIC setting up. Then we also will give the 
# MAGIC feedback function access to the input, that is the 
# MAGIC prompt, and the output, which is the prompt, and the output, which is 
# MAGIC the final response coming out of the RAG application. With 
# MAGIC this setup, later on in the notebook, we will 
# MAGIC see how to apply this feedback function on 
# MAGIC a set of records, get the evaluation scores for answer relevance 
# MAGIC as well as the chain of thought reasons for why for that 
# MAGIC particular answer that was the judged score to be appropriate 
# MAGIC for as part of the evaluation The next 
# MAGIC feedback function that we will go deep into is 
# MAGIC context relevance. Recall that context relevance is checking how good the retrieval 
# MAGIC process is. That is, given a query, we will look at each 
# MAGIC piece of retrieved context from the vector database and 
# MAGIC assess how relevant that piece of context is to the question that 
# MAGIC was asked. Let's look at a simple example. 
# MAGIC The question here or the prompt from the user is, how can altruism 
# MAGIC be beneficial in building a career? These are 
# MAGIC the two pieces of retrieved context. And after 
# MAGIC the evaluation with context relevance, each of these pieces 
# MAGIC of retrieved context gets a score between 0 and 1. You 
# MAGIC can see here the left context got a relevant 
# MAGIC score of 0.5. The right context got a relevant 
# MAGIC score of 0.7, so it was assessed 
# MAGIC to be more relevant to this particular query. 
# MAGIC And then the mean context relevant score is the average 
# MAGIC of the relevant scores of each of these 
# MAGIC retrieved pieces of context. That gets also reported out. Let's 
# MAGIC now look at the structure of the feedback function for 
# MAGIC context relevance. Various pieces of this structure are 
# MAGIC similar to the structure for answer relevance, which we reviewed a few 
# MAGIC minutes ago. There is a provider, that's OpenAI, and the 
# MAGIC feedback function makes use of that provider to implement the context-relevance feedback function. The 
# MAGIC differences are in the inputs to this particular 
# MAGIC feedback function. In addition to the user input or prompt, 
# MAGIC we also share with this feedback function a pointer to the retrieve contexts, that 
# MAGIC is, the intermediate results in the execution of 
# MAGIC the RAG application. We get back a score 
# MAGIC for each of the retrieved pieces of context, assessing 
# MAGIC how relevant or good that context 
# MAGIC is with respect to the query that was 
# MAGIC asked, and then we aggregate and average those scores 
# MAGIC across all the retrieved pieces of context to get the final 
# MAGIC score. Now you will notice that in the answer 
# MAGIC relevance feedback function, we had only made use of 
# MAGIC the original input, the prompt, and the final 
# MAGIC response from the RAG. In this feedback function, 
# MAGIC we are making use of the input or 
# MAGIC prompt from the user, as well as intermediate results, the set 
# MAGIC of retrieve contexts, to assess the quality of the retrieval. Between these 
# MAGIC two examples, the full power of feedback 
# MAGIC functions is leveraged by making use of inputs, outputs, and intermediate 
# MAGIC results of a RAG application to assess its quality. Now that we 
# MAGIC have the context selection set up, we are in a 
# MAGIC position to define the context relevance feedback function in code. You'll 
# MAGIC see that it's pretty much the code segment 
# MAGIC that I walked through on the slide. We are still using OpenAI 
# MAGIC as the provider, GPT 3.5 as the evaluation LLM. 
# MAGIC We are calling the question statement or context relevance feedback function. It gets 
# MAGIC the input prompt, the set of retrieved pieces of context, it 
# MAGIC runs the evaluation function on each of those 
# MAGIC retrieved pieces of context separately, gets a score for 
# MAGIC each of them, and then averages them to report a final 
# MAGIC aggregate score. Now, one additional variant 
# MAGIC that you can also use, 
# MAGIC if you like, is in addition to reporting a context-relevant score for 
# MAGIC each piece of retrieved context, you can also augment it with chain-of-thought 
# MAGIC reasoning so that the evaluation LLM provides not only 
# MAGIC a score, So that the evaluation LLM provides not only a 
# MAGIC score, but also a justification or explanation for its assessment score. 
# MAGIC And that can be done with QS relevance with chain of 
# MAGIC thought reasoning method. And if I give you a concrete example 
# MAGIC of this in action, you can see here's the 
# MAGIC question or the user prompt, how can altruism be 
# MAGIC beneficial in building a career? This is an 
# MAGIC example of a retrieved piece of context 
# MAGIC that takes out a chunk from Andrew's article 
# MAGIC on this topic. You can see 
# MAGIC the context relevance feedback function gives a score of 0.7 on a 
# MAGIC scale of 0 to 1 to this piece of retrieved context. 
# MAGIC And because we have also invoked the chain of 
# MAGIC thought reasoning on the evaluation LLM, it provides 
# MAGIC this justification for why the score is 0.7. Let 
# MAGIC me now show you the code snippet to set 
# MAGIC up the groundedness feedback function. We kick it off 
# MAGIC in much the same way as the previous feedback functions, leveraging LLM provider for 
# MAGIC evaluation, which is, if you recall, OpenAI GPT 3.5. Then we define 
# MAGIC the groundedness feedback function. This definition is structurally 
# MAGIC very similar to the definition for context relevance. The groundedness measure comes with chain of 
# MAGIC thought reasons justifying the scores, much 
# MAGIC like I discussed on the slides. We give it the name groundedness, 
# MAGIC which is easy to understand. And it 
# MAGIC gets access to the set of retrieved contexts in the RAG application, 
# MAGIC much like for context relevance, as well 
# MAGIC as the final output or response 
# MAGIC from the RAG. And then each 
# MAGIC sentence in the final response gets a grounded net 
# MAGIC score, and those are aggregated, averaged, to produce the final grounded net score 
# MAGIC for the full response. The context selection here is the same context selection 
# MAGIC that was used for setting up 
# MAGIC the context relevance feedback function. So if you recall, that 
# MAGIC just gets the set of retrieved pieces of context from 
# MAGIC the retrieval step of the RAG, and then can 
# MAGIC access each node within that list, recover the text of the 
# MAGIC context from that node, and proceed to work with that to 
# MAGIC do the context relevance as well as the groundedness 
# MAGIC evaluation. With that, we are now in a position 
# MAGIC to start executing the evaluation of the RAG 
# MAGIC application. We have set up all three feedback functions, 
# MAGIC answer relevance, context relevance, and groundedness. And all we 
# MAGIC need is an evaluation set on which we can 
# MAGIC run the application and the evaluations and see how they're doing and 
# MAGIC if there are opportunities to iterate and improve them further. Let's now look 
# MAGIC at the workflow to evaluate and 
# MAGIC iterate to improve LLM applications. We will start with the basic 
# MAGIC Llama Index RAG that 
# MAGIC we introduced in the previous lesson and which we have 
# MAGIC already evaluated with the TruLens RAG triad. We'll 
# MAGIC focus a bit on the failure modes related to the 
# MAGIC context size. Then we will iterate on that basic 
# MAGIC rag with an advanced rag technique, the Llama Index 
# MAGIC sentence window rag. Next, we will re-evaluate this new 
# MAGIC advanced rag with the TruLens RAG triad, focusing on these kinds of 
# MAGIC questions. Do we see improvements specifically in context relevance? 
# MAGIC What about the other metrics? The reason we focus on context relevance 
# MAGIC is that often failure modes arise because the context 
# MAGIC is too small. Once you increase the context up 
# MAGIC to a certain point, you might see improvements in context 
# MAGIC relevance. In addition, when context relevance goes up, often we 
# MAGIC find improvements in groundedness as well. Because the LLM in 
# MAGIC the completion step has enough relevant context to produce 
# MAGIC the summary. When it does not have enough relevant 
# MAGIC context, it tends to leverage its own internal knowledge 
# MAGIC from the pre-training data set to try to fill those gaps, which results in 
# MAGIC a loss of groundedness. Finally, we 
# MAGIC will experiment with different window sizes to figure out 
# MAGIC what window size results in the best evaluation metrics. Recall 
# MAGIC that if the window size is too small, there 
# MAGIC may not be enough relevant context to get 
# MAGIC a good score on context relevance and groundedness. If the window 
# MAGIC size becomes too big, on the other hand, irrelevant 
# MAGIC context can creep into the final response, resulting 
# MAGIC in not such great scores in 
# MAGIC groundedness or answer relevance. We walked through three examples of evaluations or 
# MAGIC feedback functions. Context relevance, answer relevance, and groundedness. In our 
# MAGIC notebook, all three were implemented with LLM evaluations. 
# MAGIC I do want to point out that feedback 
# MAGIC functions can be implemented in different ways. Often, 
# MAGIC we see practitioners starting out with ground truth 
# MAGIC evals, which can be expensive to collect, but nevertheless a 
# MAGIC good starting point. We also see people leverage humans to do 
# MAGIC evaluations. That's also helpful and meaningful, 
# MAGIC but hard to scale in practice. Ground truth evals, just 
# MAGIC to give you a concrete example, think of a summarization 
# MAGIC use case where there's a large 
# MAGIC passage and then the LLM produces a summary. 
# MAGIC A human expert would then give that summary a score indicating 
# MAGIC how good it is. This can be used for other 
# MAGIC kinds of use cases as well, such as chatbot-like use cases 
# MAGIC or even classification use cases. Human evals are similar 
# MAGIC in some ways to ground truth evals in that as the 
# MAGIC LLM produces an output or a RAG application produces 
# MAGIC an output, the human users of that application are going to 
# MAGIC provide a rating for that output, how good it is. 
# MAGIC The difference with ground truth evals is that these human users may 
# MAGIC not be as much of an expert in the topic as 
# MAGIC the ones who produce the curated ground truth 
# MAGIC evals. It's nevertheless a very meaningful evaluation. It'll 
# MAGIC scale a bit better than the ground truth 
# MAGIC evals, but our degree of confidence in it is lower. One 
# MAGIC very interesting result from the research literature is that if 
# MAGIC you ask a set of humans to rate 
# MAGIC a question, there's about 80% agreement. And interestingly enough, when you use 
# MAGIC LLMs for evaluation, the agreement between the LLM 
# MAGIC evaluation and the human evaluation is also about 
# MAGIC the 80 to 85% mark. So that suggests that 
# MAGIC LLM evaluations are quite comparable to human evaluations for 
# MAGIC the benchmark data sets to which they have been 
# MAGIC applied. So feedback functions provide us a way to scale up evaluations 
# MAGIC in a programmatic manner. In addition to the LLM evals 
# MAGIC that you have seen, feedback functions also provide can 
# MAGIC implement traditional NLP metrics such as rouge scores 
# MAGIC and blue scores. They can be helpful in certain scenarios, 
# MAGIC but one weakness that they have 
# MAGIC is that they are quite syntactic. 
# MAGIC They look for overlap between words in two pieces of text. 
# MAGIC So for example, if you have one piece of text that's referring 
# MAGIC to a river bank and the other to 
# MAGIC a financial bank, syntactically they might be viewed 
# MAGIC as similar, and these references might 
# MAGIC end up being viewed as similar references by a traditional 
# MAGIC NLP evaluation, whereas the surrounding context will get 
# MAGIC used to provide a more meaningful evaluation when 
# MAGIC you're using either large language models such as GPT-4 or medium-sized 
# MAGIC language models such as BERT models, and to perform your evaluation. 
# MAGIC While in the course we have given you three examples of feedback functions 
# MAGIC and evaluations, answer relevance, context relevance, and 
# MAGIC groundedness, TruLens provides a much broader set of evaluations 
# MAGIC to ensure that the apps that you're 
# MAGIC building are honest, harmless, and helpful. These are all 
# MAGIC available in the open source library, 
# MAGIC and we encourage you to play with them as 
# MAGIC you are working through the course and building your LLM 
# MAGIC applications. Now that we have set up all the feedback functions, 
# MAGIC we can set up an object 
# MAGIC to start recording, which will 
# MAGIC be used to record 
# MAGIC the execution of the application on various records. So you'll see here 
# MAGIC that we are importing this Tru Llama class, 
# MAGIC creating an object, Tru Recorder of 
# MAGIC this Tru Llama class. This is our integration of 
# MAGIC TruLens with Llama Index. It takes in the 
# MAGIC sentence-window-engine from Llama Index that we had created 
# MAGIC earlier, sets the app ID, and makes use of the 
# MAGIC three feedback functions of the RAG triad that we created earlier. This 
# MAGIC Tru recorder object will be used in a little bit to 
# MAGIC run the Llama Index application, as well as the evaluation of these feedback 
# MAGIC functions, and to record it all in a local database. 
# MAGIC Let us now load some evaluation questions. In this 
# MAGIC setup, the evaluation questions are set up already in this 
# MAGIC text file, and then we just execute this code snippet to load them 
# MAGIC in. Let's take a quick look at these 
# MAGIC questions that we will use for evaluation. You 
# MAGIC can see what are the keys to building a career in AI, and 
# MAGIC so on. And this file you can edit 
# MAGIC yourself and add your own questions that you might want to get answers 
# MAGIC from Andrew Ang. You can also append 
# MAGIC directly to the eval questions list in this way. Now let's 
# MAGIC take a look at the eval questions list, and you can see that 
# MAGIC this question has been added at the end. 
# MAGIC Go ahead and add your own questions. And now we have everything 
# MAGIC set up to get to the most exciting step in this notebook with this 
# MAGIC code snippet. We can execute the sentence window 
# MAGIC engine on each question in the list of eval questions 
# MAGIC that we just looked at. And then with 
# MAGIC Tru recorder, we are going to run each record against the RAG triad. We will 
# MAGIC record the prompts, responses, intermediate results, and the evaluation 
# MAGIC results in the true database. And you can see here as the 
# MAGIC execution of the steps are happening for each 
# MAGIC record, there is a hash that's an identifier 
# MAGIC for the record. As the record gets added, we 
# MAGIC have an indicator here that that step has 
# MAGIC executed effectively. In addition, the feedback results for answer 
# MAGIC relevance is done, and so on for context relevance, and 
# MAGIC so on. Now that we have the recording done, we can 
# MAGIC see the logs in the notebook by 
# MAGIC executing, by getting the records and feedback and 
# MAGIC executing this code snippet. And I don't want 
# MAGIC you to necessarily read through all of the information 
# MAGIC here. The main point I want to make is that you can see the 
# MAGIC depth of instrumentation and the application. A lot of information 
# MAGIC gets logged to the tru recorder, and this 
# MAGIC information are on prompts, responses, evaluation results, and so 
# MAGIC fault. And can be quite valuable to identify failure mods in the 
# MAGIC apps and to inform iteration and improvement of the apps. All 
# MAGIC of this information is valuable, flexible JSON format, so they can be 
# MAGIC exploded and consumed by downstream processes. Next up, 
# MAGIC let's look at some more human-readable format, for prompts, responses, and the 
# MAGIC feedback function evaluations. With these quotes stampedes, 
# MAGIC you can see that for each input prompt or question, 
# MAGIC we see the output and the respective scores for 
# MAGIC context relevance, groundedness, and answer relevance, and this is run for 
# MAGIC each and every entry in this list of 
# MAGIC questions and evaluations_questions.text. You can see here the last question is 
# MAGIC "How can I be successful in the AI?" Was 
# MAGIC the question that I manually embedded to 
# MAGIC that list have the end. Sometimes in running the evaluations, 
# MAGIC you might see a non that 
# MAGIC likely happen because of API key failure. You just 
# MAGIC want to rerun it to, and show that the execution successfully 
# MAGIC completes. I just showed you a 
# MAGIC record level view of the evaluations, the prompts, responses, and evaluations, 
# MAGIC let's now get an aggregate view in the 
# MAGIC leaderboard, which aggregates across all of these individual records and 
# MAGIC produces an average score across the 10 records in that database. So 
# MAGIC you can see here in the leaderboard, the aggregate 
# MAGIC view across all the 10 records. We had set 
# MAGIC the app ID to app 1. The average context 
# MAGIC relevance is 0.56. Similarly, their average scores for groundedness, answer 
# MAGIC relevance, and latency across all the 10 records of 
# MAGIC questions that were asked of the RAG application. It's 
# MAGIC useful to get this aggregate view to see how well your app 
# MAGIC is performing and at what level of latency and 
# MAGIC cost. In addition to the notebook interface, TrueLens also 
# MAGIC provides a local Streamlit app dashboard with which you can 
# MAGIC examine the applications that you're building, look at 
# MAGIC the evaluation results, drill down into record-level views to both get aggregate and detailed evaluation 
# MAGIC views into the performance of your app. 
# MAGIC So we can get the dashboard 
# MAGIC going with the True.run Dashboard method, and this 
# MAGIC sets up a local database at a certain URL. 
# MAGIC Now once I click on this, this might 
# MAGIC show up in some window, which is not within this frame. 
# MAGIC Let's take a few minutes to walk through this 
# MAGIC dashboard. You can see here the aggregate view of 
# MAGIC the app's performance. 11 records were processed 
# MAGIC by the app and evaluated. The average latency is 
# MAGIC 3.55 seconds. We have the total cost, the 
# MAGIC total number of tokens that were processed by 
# MAGIC the LLMs. And then scores for the RAG triad. For 
# MAGIC context relevance, it's 0.56. For groundedness, 0.86. And 
# MAGIC answer relevant is 0.92. We can select the app here to get 
# MAGIC a more detailed record-evel view of the evaluations. For each of the records, you 
# MAGIC can see that the user input, the prompt, the response, 
# MAGIC this metadata, the timestamp, and then scores for answer relevance, 
# MAGIC context relevance, and groundedness that have 
# MAGIC been recorded, along with latency, total number of tokens, and total cost. 
# MAGIC Let me pick a row in which the LLM indicates, 
# MAGIC evaluation indicates that the LLM, the RAG application has done well. Let's pick 
# MAGIC this row. Once we click on a row, we can scroll 
# MAGIC down and get a more detailed view of the different components of 
# MAGIC that row from the table. So the question here, the prompt 
# MAGIC was, what is the first step to becoming good at AI? The 
# MAGIC final response from the RAG was, is to 
# MAGIC learn foundational technical skills. Down here, you can 
# MAGIC see that the answer relevance was 
# MAGIC viewed to be one on a scale 
# MAGIC of zero to 1. It's a relevant, quite a relevant answer to the 
# MAGIC question that was asked. Up here, you can see that context 
# MAGIC relevance, the average context relevance score is 0.8. For the two 
# MAGIC pieces of context that were retrieved, Both of them individually got scores 
# MAGIC of 0.8. We can see the chain of 
# MAGIC thought reason for why the LLM evaluation 
# MAGIC gave a score of 0.8 to this particular response from 
# MAGIC the RAG in the retrieval step. And then down here, 
# MAGIC you can see the groundedness evaluations. So this was 
# MAGIC one of the clauses in the final answer. It got a score 
# MAGIC of one. And over here is the reason for that score. You 
# MAGIC can see this was the statement sentence, 
# MAGIC and the supporting evidence backs it up. And so 
# MAGIC it got a full score of one on a 
# MAGIC scale of 0 to 1, or a full score 
# MAGIC of 10 on a scale of zero to 10. So previously, the kind 
# MAGIC of reasoning and information we were talking about through slides 
# MAGIC and in the notebook, now you can see 
# MAGIC that quite neatly in this StreamLit local app that runs on 
# MAGIC your machine. You can also get a detailed 
# MAGIC view of the timeline, as well as get access to the full 
# MAGIC JSON object. Now let's look at an example where 
# MAGIC the rag did not do so well. So 
# MAGIC as I look through the evaluations, I 
# MAGIC see this row with a low groundedness score of 0.5. So let's click on that. That 
# MAGIC brings up this example. The question is how can altruism be beneficial in 
# MAGIC building a career? There's a response. 
# MAGIC If I scroll down to the groundedness evaluation, 
# MAGIC then both of the sentences in the final response have low groundedness 
# MAGIC score. Let's pick one of these and look at 
# MAGIC why the groundedness score is low. So you can see 
# MAGIC this, the overall response got broken down into four statements, and the 
# MAGIC top two were good, but the bottom two did not have good supporting 
# MAGIC evidence in the retrieved pieces of context. In particular, if you look 
# MAGIC at this last one, the final output from the LLM says, additionally, practicing 
# MAGIC altruism can contribute to personal fulfillment and a sense 
# MAGIC of purpose, which can enhance motivation and overall well-being, 
# MAGIC ultimately benefiting one's career success. While that might very well be 
# MAGIC the case, there was no supporting evidence found 
# MAGIC in the retrieved pieces of context to ground that statement. 
# MAGIC And that's why our evaluation gives this a low score. 
# MAGIC You can play around with the dashboard and explore 
# MAGIC some of these other examples where the LLM, the final RAG 
# MAGIC output, does not do so well to get 
# MAGIC a feeling for the kinds of failure modes 
# MAGIC that are quite common when you're using RAG applications. 
# MAGIC And some of these will get addressed as 
# MAGIC we go into the sessions on 
# MAGIC more advanced RAG techniques, which can do better in 
# MAGIC terms of addressing these failure modes. Lesson two 
# MAGIC is a wrap with that. In the next lesson, we will 
# MAGIC walk through the mechanism for sentence window-based retrieval 
# MAGIC and advanced RAG technique, and also show you how to 
# MAGIC evaluate the advanced technique leveraging the RAG triad and 
# MAGIC true lengths. 
# MAGIC
