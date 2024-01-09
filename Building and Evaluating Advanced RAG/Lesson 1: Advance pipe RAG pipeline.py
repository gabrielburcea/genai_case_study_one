# Databricks notebook source
"""Lesson 1: Advanced RAG Pipeline"""

import utils
​
import os
import openai
openai.api_key = utils.get_openai_api_key()

from llama_index import SimpleDirectoryReader
​
documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])
Basic RAG pipeline

from llama_index import Document
​
document = Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
​
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)

query_engine = index.as_query_engine()

response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))
Evaluation setup using TruLens

eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)

# You can try your own question:
new_question = "What is the right AI job for me?"
eval_questions.append(new_question)

print(eval_questions)

from trulens_eval import Tru
tru = Tru()
​
tru.reset_database()
For the classroom, we've written some of the code in helper functions inside a utils.py file.
You can view the utils.py file in the file directory by clicking on the "Jupyter" logo at the top of the notebook.
In later lessons, you'll get to work directly with the code that's currently wrapped inside these helper functions, to give you more options to customize your RAG pipeline.

from utils import get_prebuilt_trulens_recorder
​
tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])

records.head()

# launches on http://localhost:8501/
tru.run_dashboard()
Advanced RAG pipeline
1. Sentence Window retrieval

from llama_index.llms import OpenAI
​
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

from utils import build_sentence_window_index
​
sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

from utils import get_sentence_window_query_engine
​
sentence_window_engine = get_sentence_window_query_engine(sentence_index)

window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))

tru.reset_database()
​
tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)

for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

tru.get_leaderboard(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()
2. Auto-merging retrieval

from utils import build_automerging_index
​
automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)

from utils import get_automerging_query_engine
​
automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)

auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))

tru.reset_database()
​
tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")

for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()


# COMMAND ----------

# MAGIC %md
# MAGIC In this lesson, you'll get a full overview of 
# MAGIC how to set up both a basic and advanced RAG 
# MAGIC pipeline with Llama Index. We'll load in an evaluation benchmark and use 
# MAGIC TrueLens to define a set of metrics so that we can benchmark 
# MAGIC advanced RAG techniques against a baseline, or basic 
# MAGIC pipeline. In the next few lessons, we'll explore each 
# MAGIC lesson a little bit more in depth. Let's first walk through 
# MAGIC how a basic Retrieval Augmented Generation pipeline works, 
# MAGIC or a RAG pipeline. It consists of three different components, ingestion, 
# MAGIC retrieval, and synthesis. Going through 
# MAGIC the ingestion phase, we first load in 
# MAGIC a set of documents. For each document, we split it into a set of text 
# MAGIC chunks using a text splitter. Then for each chunk, we generate an embedding 
# MAGIC for that chunk using an embedding model. And then for each 
# MAGIC chunk with embedding, we offload it to an index, which is a view 
# MAGIC of a storage system such as a MacAndrew database. Once the data is 
# MAGIC stored within an index, we then perform retrieval 
# MAGIC against that index. First, we launch a user 
# MAGIC query against the index, and then we fetch 
# MAGIC the top K most similar chunks to the user query. Afterwards, we take these 
# MAGIC relevant chunks, combine it with the 
# MAGIC user query, and put it into the prompt window of 
# MAGIC the LLM in the synthesis phase. And this allows us to generate a final 
# MAGIC response. This notebook will walk you through how to set up 
# MAGIC a basic and advanced RAG pipeline with Llama Index. 
# MAGIC We will also use TruEra to help set up an 
# MAGIC evaluation benchmark so that we can measure improvements against the baseline. For 
# MAGIC this quick start, you will need an 
# MAGIC OpenAI API key. Note that for this lesson, we'll use a set of 
# MAGIC helper functions to get you set up and running quickly, 
# MAGIC and we'll do a deep dive into some of these sections in the future lessons. 
# MAGIC Next, we'll create a simple LLM application using 
# MAGIC Llama Index, which internally uses an OpenAI LLM. In 
# MAGIC terms of the data source, we'll use the How to Build a Career 
# MAGIC in AI PDF written by Andrew Wright. Note that you 
# MAGIC can also upload your own PDF file if you 
# MAGIC wish. And for this lesson, we encourage you to 
# MAGIC do so. Let's do some basic sanity checking of what the 
# MAGIC document consists of as well as the length 
# MAGIC of the document. We see that we have 
# MAGIC a list of documents. There's 41 elements in there. Each item in 
# MAGIC that list is a document object. And 
# MAGIC we'll also show a snippet of the text for a given 
# MAGIC document. Next, we'll merge these into a single document because it helps with 
# MAGIC overall text blending accuracy when using more advanced retrieval methods 
# MAGIC such as a sentence window retrieval 
# MAGIC as well as auto-merging retrieval. The 
# MAGIC next step here is to index these documents and we can do 
# MAGIC this with the vector store index within Llama 
# MAGIC Index. Next, we define a service context object 
# MAGIC which contains both the LLM we're going 
# MAGIC to use as well as the embedding model 
# MAGIC we're going to use. The LLM we're going to use is GPT-3.5-Turbo from OpenAI, and 
# MAGIC then the embedding model that we're going to use is the HuggingFace BGESmall model. These 
# MAGIC few steps show this ingestion process right here. We've loaded 
# MAGIC in documents, and then in one line, "VectorStoreIndexOfFromDocuments," we're 
# MAGIC doing the chunking, embedding, and indexing under the hood 
# MAGIC with the embedding model that you specified. Next, we obtain 
# MAGIC a query engine from this index that 
# MAGIC allows us to send user queries that do 
# MAGIC retrieval and synthesis against this data. Let's try out our first 
# MAGIC request. And the query is, what are steps to take when finding 
# MAGIC projects to build your experience? Let's find out. Start small and gradually 
# MAGIC increase the scope and complexity of your projects. Great, so it's working. 
# MAGIC So now you've set up the basic RAG 
# MAGIC pipeline. The next step is to set up 
# MAGIC some evaluations against this pipeline to understand how well it performs, and 
# MAGIC this will also provide the basis for defining our advanced retrieval methods of 
# MAGIC a sentence window retriever as well 
# MAGIC as an auto-merging retriever. In this section, we use TrueLens to initialize 
# MAGIC feedback functions. We initialize a helper function, get feedbacks, 
# MAGIC to return a list of feedback functions to 
# MAGIC evaluate our app. Here, we've created a RAG 
# MAGIC evaluation triad, which consists of pairwise comparisons between the 
# MAGIC query, response, and context. And so this really creates three different evaluation 
# MAGIC modules, answer relevance, context relevance, and groundedness. Answer relevance 
# MAGIC is, is the response relevant to the query. 
# MAGIC Context relevance is, is the retrieved context relevant to 
# MAGIC the query. And groundedness is, 
# MAGIC is the response supported 
# MAGIC by the context. We'll walk through how to set this 
# MAGIC up yourself in the next few notebooks. 
# MAGIC The first thing we need to do 
# MAGIC is to create set of questions on which does has her occupation. 
# MAGIC Here, we've pre-written the first 10, and we encourage you to add to the 
# MAGIC list. And now we have some evaluation questions. What are the keys 
# MAGIC to building a career in AI? How can teamwork contribute to 
# MAGIC success in AI? Etc. The first thing we need to do is 
# MAGIC to create a set of questions on which to 
# MAGIC test our application. Here, we've pre-written the first 10, 
# MAGIC but we encourage you to also add to this list. 
# MAGIC Here, we specify a fun new question, what is the 
# MAGIC right AI job for me? And we add it to the eval questions 
# MAGIC list. Now we can initialize the TrueLens modules to begin our 
# MAGIC evaluation process. We've initialized the TrueLens module and now we've reset the database. We 
# MAGIC can now initialize our evaluation modules. LLMs are growing as a standard mechanism 
# MAGIC for evaluating generative AI applications at scale. Rather than 
# MAGIC relying on expensive human evaluation or set benchmarks, 
# MAGIC LLMs allows us to evaluate our applications in a way that 
# MAGIC is custom to the domain in which we operate 
# MAGIC and dynamic to the changing demands for our 
# MAGIC application. Here we've pre-built a ShuLens recorder 
# MAGIC to use for this example. In the recorder, 
# MAGIC we've included the standard triad of evaluations for evaluating regs. Groundedness, context 
# MAGIC relevance, and answer relevance. We'll also specify an 
# MAGIC ID so that we can track this version of 
# MAGIC our app. As we experiment, we can track new 
# MAGIC versions by simply changing the app ID. Now we can 
# MAGIC run the query engine again with the 
# MAGIC TrueLens context. So what's happening here is that we're sending 
# MAGIC each query to our query engine. And in 
# MAGIC the background, the TrueLens recorder is evaluating each of our queries 
# MAGIC against these three metrics. If you see some 
# MAGIC warning messages, don't worry about it. Some of it is system dependent. 
# MAGIC Here we can see a list of queries 
# MAGIC as well as their associated responses. You 
# MAGIC can see the input, output, the record ID, tags, and 
# MAGIC more. You can also see the answer relevance, context relevance, and 
# MAGIC groundedness for each rub. In this dashboard, you can see your evaluation metrics like 
# MAGIC context relevance, answer relevance, and groundedness, 
# MAGIC as well as average latency, total cost, and more in the UI. 
# MAGIC Here, we see that the answer relevance and groundedness 
# MAGIC are decently high, but CloudTax relevance is pretty low. Now let's see 
# MAGIC if we can improve these metrics with 
# MAGIC more advanced retrieval techniques like sentence window 
# MAGIC retrieval as well as on emerging retrieval. The first advanced technique 
# MAGIC we'll talk about is sentence window retrieval. This 
# MAGIC works by embedding and retrieving single sentences, so more granular chunks. But after retrieval, 
# MAGIC the sentences are replaced with a 
# MAGIC larger window of sentences around the original retrieved sentence. 
# MAGIC The intuition is that this allows for 
# MAGIC the LLM to have more context for the information retrieved in 
# MAGIC order to better answer queries while still retrieving 
# MAGIC on more granular pieces of information. So ideally improving 
# MAGIC both retrieval as well as synthesis 
# MAGIC performance. Now let's take a look at how to 
# MAGIC set it up. First, we'll use opening IGBT 3.5 
# MAGIC Turbo. Next, we'll construct our sentence 
# MAGIC window index over the given document. 
# MAGIC Just a reminder that we have a helper 
# MAGIC function for constructing the sentence window index over the 
# MAGIC given document. and we'll do a deep dive in how this works under the hood in the 
# MAGIC next few lessons. Similar to before, we'll get a query 
# MAGIC engine from the sentence window index. And now that we've 
# MAGIC set this up, we can try 
# MAGIC running an example query. Here the question is, how 
# MAGIC do I get started on a personal project in AI? 
# MAGIC And we get back a response. Get started on a personal 
# MAGIC project in AI is first important to identify scope the 
# MAGIC project. Great. Similarly to before, let's try getting the TrueLens evaluation context and try benchmarking 
# MAGIC the results. So here, we import the True recorder 
# MAGIC sentence window, which is a pre-built True Lens recorder for the sentence 
# MAGIC window index. And now we'll run the sentence 
# MAGIC window retriever on top of these evaluation questions and 
# MAGIC then compare performance on the RAG triad of 
# MAGIC evaluation modules. Here we can see the responses 
# MAGIC come in as they're being run. Some examples of 
# MAGIC questions and responses. How can teamwork contribute to 
# MAGIC success in AI? Teamwork can contribute to success in AI by 
# MAGIC allowing individuals to leverage the expertise and insights of 
# MAGIC their colleagues. What's the importance of networking in AI? Networking is 
# MAGIC important in AI because it allows individuals to connect with 
# MAGIC others who have experience and knowledge in the field. 
# MAGIC Great. Now that we've run evaluations for two 
# MAGIC techniques, the basic RAG pipeline, as well as the 
# MAGIC sentence window retrieval pipeline, let's get a leaderboard of 
# MAGIC the results and see what's going on. Here, we 
# MAGIC see that general groundedness is 8 percentage points better 
# MAGIC than the baseline RAG pipeline. Answer relevance is more 
# MAGIC or less the same. Context relevance is also better for the sentence window 
# MAGIC primary engine. Latency is more or less the 
# MAGIC same, and the total cost is lower. Since the groundedness 
# MAGIC and context relevance are higher, but the total cost is lower, we 
# MAGIC can intuit that the sentence window retriever is actually 
# MAGIC giving us more relevant context and more 
# MAGIC efficiently as well. When we go back 
# MAGIC into the UI, we can see that we 
# MAGIC now have a comparison between the direct query engine and the baseline, as 
# MAGIC well as the sentence window. And we can 
# MAGIC see the metrics that we just saw in the notebook displayed in 
# MAGIC UI as well. The next advanced 
# MAGIC retrieval technique we'll talk about is the auto-merging 
# MAGIC retriever. Here we construct a hierarchy of larger parent nodes with smaller child nodes that 
# MAGIC reference the parent node. So for instance we might 
# MAGIC have a parent node of chunk size 512 tokens, and underneath there are four 
# MAGIC child nodes of chunk size 128 tokens that link to 
# MAGIC this parent node. The auto-merging retriever 
# MAGIC works by merging retrieved nodes into larger parent nodes, which means 
# MAGIC that during retrieval, if a parent actually has a 
# MAGIC majority of its children nodes retrieved, then we'll replace the 
# MAGIC children nodes with the parent node. So this allows us 
# MAGIC to hierarchically merge our retrieved nodes. The combination 
# MAGIC of all the child nodes is the same text as 
# MAGIC the parent node. Similarly to the sentence window 
# MAGIC retriever, in the next few lessons, we'll do a bit 
# MAGIC more of a deep dive on how it 
# MAGIC works. Here, we'll show you how to set it up 
# MAGIC with our helper functions. Here, we've built the auto merging index, 
# MAGIC again, using GPT 3.5 Turbo for the LLM, as 
# MAGIC well as the BGE model for the embedding model. We 
# MAGIC get the query engine from the 
# MAGIC auto-merging retriever. And let's try running an example query. How do I 
# MAGIC build a portfolio of AI projects? In the logs here, you actually see the merging 
# MAGIC process go on. We're merging nodes into a 
# MAGIC parent node to basically retrieve the parent node as opposed 
# MAGIC to the child node. To build a portfolio of AI 
# MAGIC projects, it is important to start with simple undertakings and gradually progress to 
# MAGIC more complex ones. Great, so we see that it's working. 
# MAGIC Now let's benchmark results with TrueLens. We get a pre-built TrueLens recorder on 
# MAGIC top of our auto merging retriever. We then run 
# MAGIC the auto merging retriever with TrueLens on top of 
# MAGIC our evaluation questions. Here for each question, you actually see 
# MAGIC the merging process going on, such as merging three nodes into the parent 
# MAGIC node for the first question. If 
# MAGIC we scroll down just a little bit, we see that for 
# MAGIC some of these other questions, we're also 
# MAGIC performing the merging process. Merging three nodes into a parent 
# MAGIC node, merging one node into a parent node. An example 
# MAGIC question response pair is, what is the importance of 
# MAGIC networking in AI? Networking is important in AI 
# MAGIC because it helps in building a strong professional 
# MAGIC networking community.
