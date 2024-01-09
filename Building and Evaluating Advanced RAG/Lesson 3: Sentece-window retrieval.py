# Databricks notebook source
"""Lesson 3: Sentence Window Retrieval"""

import warnings
warnings.filterwarnings('ignore')

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

from llama_index import Document
​
document = Document(text="\n\n".join([doc.text for doc in documents]))
Window-sentence retrieval setup

from llama_index.node_parser import SentenceWindowNodeParser
​
# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

text = "hello. how are you? I am fine!  "
​
nodes = node_parser.get_nodes_from_documents([Document(text=text)])

print([x.text for x in nodes])

print(nodes[1].metadata["window"])

text = "hello. foo bar. cat dog. mouse"
​
nodes = node_parser.get_nodes_from_documents([Document(text=text)])

print([x.text for x in nodes])

print(nodes[0].metadata["window"])
Building the index

from llama_index.llms import OpenAI
​
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

from llama_index import ServiceContext
​
sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    # embed_model="local:BAAI/bge-large-en-v1.5"
    node_parser=node_parser,
)

from llama_index import VectorStoreIndex
​
sentence_index = VectorStoreIndex.from_documents(
    [document], service_context=sentence_context
)

sentence_index.storage_context.persist(persist_dir="./sentence_index")
​

# This block of code is optional to check
# if an index file exist, then it will load it
# if not, it will rebuild it
​
import os
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import load_index_from_storage
​
if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(
        [document], service_context=sentence_context
    )
​
    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./sentence_index"),
        service_context=sentence_context
    )
Building the postprocessor

from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
​
postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

from llama_index.schema import NodeWithScore
from copy import deepcopy
​
scored_nodes = [NodeWithScore(node=x, score=1.0) for x in nodes]
nodes_old = [deepcopy(n) for n in nodes]

nodes_old[1].text

replaced_nodes = postproc.postprocess_nodes(scored_nodes)

print(replaced_nodes[1].text)
Adding a reranker

from llama_index.indices.postprocessor import SentenceTransformerRerank
​
# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)

from llama_index import QueryBundle
from llama_index.schema import TextNode, NodeWithScore
​
query = QueryBundle("I want a dog.")
​
scored_nodes = [
    NodeWithScore(node=TextNode(text="This is a cat"), score=0.6),
    NodeWithScore(node=TextNode(text="This is a dog"), score=0.4),
]

reranked_nodes = rerank.postprocess_nodes(
    scored_nodes, query_bundle=query
)

print([(x.text, x.score) for x in reranked_nodes])
Runing the query engine

sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=6, node_postprocessors=[postproc, rerank]
)

window_response = sentence_window_engine.query(
    "What are the keys to building a career in AI?"
)

from llama_index.response.notebook_utils import display_response
​
display_response(window_response)
Putting it all Together

import os
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage
​
​
def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )
​
    return sentence_index
​
​
def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
​
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

from llama_index.llms import OpenAI
​
index = build_sentence_window_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./sentence_index",
)
​

query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
​
TruLens Evaluation

eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

from trulens_eval import Tru
​
def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

from utils import get_prebuilt_trulens_recorder
​
from trulens_eval import Tru
​
Tru().reset_database()
Sentence window size = 1

sentence_index_1 = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=1,
    save_dir="sentence_index_1",
)

sentence_window_engine_1 = get_sentence_window_query_engine(
    sentence_index_1
)

tru_recorder_1 = get_prebuilt_trulens_recorder(
    sentence_window_engine_1,
    app_id='sentence window engine 1'
)

run_evals(eval_questions, tru_recorder_1, sentence_window_engine_1)

Tru().run_dashboard()

"""Note about the dataset of questions
Since this evaluation process takes a long time to run, the following file generated_questions.text contains one question (the one mentioned in the lecture video).
If you would like to explore other possible questions, feel free to explore the file directory by clicking on the "Jupyter" logo at the top right of this notebook. You'll see the following .text files:
generated_questions_01_05.text
generated_questions_06_10.text
generated_questions_11_15.text
generated_questions_16_20.text
generated_questions_21_24.text
Note that running an evaluation on more than one question can take some time, so we recommend choosing one of these files (with 5 questions each) to run and explore the results.
For evaluating a personal project, an eval set of 20 is reasonable.
For evaluating business applications, you may need a set of 100+ in order to cover all the use cases thoroughly.
Note that since API calls can sometimes fail, you may occasionally see null responses, and would want to re-run your evaluations. So running your evaluations in smaller batches can also help you save time and cost by only re-running the evaluation on the batches with issues."""

eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)
Sentence window size = 3

sentence_index_3 = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index_3",
)
sentence_window_engine_3 = get_sentence_window_query_engine(
    sentence_index_3
)
​
tru_recorder_3 = get_prebuilt_trulens_recorder(
    sentence_window_engine_3,
    app_id='sentence window engine 3'
)

run_evals(eval_questions, tru_recorder_3, sentence_window_engine_3)

Tru().run_dashboard()"""

# COMMAND ----------

# MAGIC %md
# MAGIC In this lesson, we'll do a deep dive into 
# MAGIC an advanced RAG technique, our sentence window retrieval method. In 
# MAGIC this method, we retrieve based on smaller 
# MAGIC sentences to better match the relevant context, and 
# MAGIC then synthesize based on an expanded context window 
# MAGIC around the sentence. Let's check out how to set it up. First, some context. 
# MAGIC The standard RAG pipeline uses the same text chunk for both 
# MAGIC embedding and synthesis. The issue is that embedding and synthesis. The 
# MAGIC issue is that embedding-based retrieval typically 
# MAGIC works well with smaller chunks, whereas the LLM 
# MAGIC needs more context and bigger chunks to synthesize a good answer. What 
# MAGIC sentence window retrieval does is decouple the two a bit. We 
# MAGIC first embed smaller chunks or sentences and store them in 
# MAGIC a vector database. We also add context of the sentences that 
# MAGIC occur before and after to each chunk. During retrieval, we 
# MAGIC retrieve the sentences that are most relevant to the question with 
# MAGIC a similarity search and then replace 
# MAGIC the sentence with a full surrounding context. This allows us 
# MAGIC to expand the context that's actually fed to 
# MAGIC the LLM in order to answer the question. This notebook will introduce 
# MAGIC the various components needed to construct 
# MAGIC a sentence window retriever with Llama 
# MAGIC Index. The various components will be covered in detail. 
# MAGIC At the end, Anupam will show you how to experiment with parameters and 
# MAGIC evaluation with TruEra. This is the same 
# MAGIC setup that you've used in the previous lessons, so make sure to install 
# MAGIC the relevant packages, such as Llama Index and 
# MAGIC Truelines eval. For this quick start, you'll need an open AI key 
# MAGIC similar to previous lessons. This open AI key is used 
# MAGIC for embeddings, LLMs, and also the evaluation piece. Now 
# MAGIC we set up and inspect our documents to use for iteration 
# MAGIC and experimentation. Similar to the first lesson, we 
# MAGIC encourage you to upload your own PDF file as well. 
# MAGIC As with before, we'll load in the How to Build 
# MAGIC a Career in AI eBook. It's the same document as before. So we 
# MAGIC see that it's a list of documents, there are 41 pages, 
# MAGIC the object schemas are document object, and here is 
# MAGIC some sample text from the first page. The next piece is, 
# MAGIC we'll merge these into a single document because it helps 
# MAGIC with overall text blending accuracy when using more advanced 
# MAGIC retrievers. Now let's set up the 
# MAGIC sentence window retrieval method, and we'll go through how to set 
# MAGIC this up more in depth. We'll start with a window size 
# MAGIC of 3 and a top K value of 6. First, we'll import what we 
# MAGIC call a sentence window node parser. The sentence window node parser is an object that 
# MAGIC will split a document into individual sentences and then augment each 
# MAGIC sentence chunk with the surrounding context around that sentence. 
# MAGIC Here we demonstrate how the node 
# MAGIC parser works with a small example. 
# MAGIC We see that our text, which 
# MAGIC has three sentences, gets split into three nodes. 
# MAGIC Each node contains a single sentence with the metadata 
# MAGIC containing a larger window around the sentence. We'll 
# MAGIC show what that metadata looks like for the second 
# MAGIC node right here. You see that this metadata contains 
# MAGIC the original sentence, but also the sentence that occurred 
# MAGIC before and after it. We encourage you to 
# MAGIC try out your own text too. For instance, let's try something 
# MAGIC like this. For this sample text, let's take a look at the 
# MAGIC surrounding metadata for the first node. Since the window size is 
# MAGIC 3, we have two additional adjacent nodes that 
# MAGIC occur in front, and of course none behind it because 
# MAGIC it's the first node. So we see that we have 
# MAGIC the original sentence, or hello, but also foobar 
# MAGIC and cat dog. The next step is actually build the index, and the 
# MAGIC first thing we'll do is a setup in 
# MAGIC LLM. In this case, we'll use OpenAI, specifically 
# MAGIC GPT-355 Turbo, with a temperature of 0.1. The next step is to 
# MAGIC set up a service context object, which, as a reminder, is 
# MAGIC a wrapper object that contains all the context needed for indexing, 
# MAGIC including the AL1, embedding model, and the node parser. Note 
# MAGIC that the embedding model that we specify is 
# MAGIC the "bge small model," and we actually download and 
# MAGIC run it locally from HuggingFace. This is a compact, 
# MAGIC fast, and accurate for its size embedding model. We can also 
# MAGIC use other embedding model. For instance, a related model is "bge 
# MAGIC large," which we have in the commented-out code 
# MAGIC below. The next step is the setup of VectorStoreIndex with the source document. 
# MAGIC Because we've defined the node parser as part 
# MAGIC of the service context, what this will do is it will 
# MAGIC take the source document, transform it 
# MAGIC into a series of sentences augmented with surrounding contexts, and embed 
# MAGIC it, and load it into the VectorStore. We can save the 
# MAGIC index to disk so that you can load it 
# MAGIC later without rebuilding it. If you've already built the index, saved 
# MAGIC it, and you don't want to rebuild it, here is a handy 
# MAGIC block of code that allows you to load the index from the 
# MAGIC existing file if it exists, otherwise, it will build it. 
# MAGIC The index is now built. The next step is to set up 
# MAGIC and run the query engine. First, what we'll do 
# MAGIC is we'll define what we call a metadata replacement post-processor. 
# MAGIC This takes a value stored in the metadata and replaces a node text 
# MAGIC with that value. And so this is done after retrieving 
# MAGIC the nodes and before sending the nodes to the outline. 
# MAGIC We'll first walk through how this works. Using 
# MAGIC the nodes we created with the sentence window node parser, we can 
# MAGIC test this post-processor. Note that we made a backup of 
# MAGIC the original nodes. Let's take a look at the second node again. Great. 
# MAGIC Now let's apply the post-processor on top of these nodes. If 
# MAGIC we now take a look at the text of the second 
# MAGIC node, we see that it's been replaced with a full context, including the sentences that occurred 
# MAGIC before and after the current node. The next step is to add 
# MAGIC the sentence transformer re-rank model. This takes the query and retrieve 
# MAGIC nodes and re-order the nodes in order relevance using a 
# MAGIC specialized model for the task. Generally, you will make the initial similarity top 
# MAGIC K larger, and then the re-ranker will rescore the nodes and 
# MAGIC return a smaller top N, so filter out a 
# MAGIC smaller set. An example of a re-ranker is bge-re-ranker 
# MAGIC based. This is a re-ranker based on the bge embeddings. 
# MAGIC This string represents the model's name from HuggingFace, and 
# MAGIC you can find more details on the model from HuggingFace. Let's take 
# MAGIC a look at how this re-ranker works. We'll input some toy data and then see 
# MAGIC how the re-ranker can actually re-rank the initial set of nodes 
# MAGIC to a new set of nodes. Let's assume the original query is I 
# MAGIC want a dog, and the initial set of score nodes is 
# MAGIC this is a cat with a score of 
# MAGIC 0.6, and then this is a dog with a score 
# MAGIC of 0.4. Intuitively, you would expect that the second node actually has a 
# MAGIC higher score, so it matches the query more. And so 
# MAGIC that's where the re-ranker can come in. Here, we see 
# MAGIC the re-ranker properly surfaces the known about dogs and gave it a high 
# MAGIC score of irrelevance. Now let's apply this to our actual query engine. 
# MAGIC As mentioned earlier, we want a larger similarity 
# MAGIC top K than the top N value we chose for the re-ranker. In 
# MAGIC order to give the re-ranker a fair chance at 
# MAGIC surfacing the proper information. We set the top K 
# MAGIC equal to 6 and top N equals to 
# MAGIC 2, which means that we first fetch the six most similar 
# MAGIC chunks using the sentence window retrieval, and then we filter 
# MAGIC for the top two most relevant chunks using the sentence re-ranker. Now that we 
# MAGIC have the full query engine set up, let's run 
# MAGIC through a basic example. Let's ask a question over this dataset. What are the 
# MAGIC keys to building a career in AI? And 
# MAGIC we get back to response. We see that 
# MAGIC the final response is that the keys to building a 
# MAGIC career in AI are learning foundational technical skills, working on projects, 
# MAGIC and finding a job. Now that we have the sentence window query engine 
# MAGIC in place, let's put everything together. We'll put a lot of code 
# MAGIC into this notebook cell, but note that this is essentially the 
# MAGIC same as the function in the utils.BAAI file. 
# MAGIC We have functions for building the sentence window index that we showed 
# MAGIC earlier in this notebook. It consists of being 
# MAGIC able to use the sentence window node parser to extract 
# MAGIC out sentences from documents and augment it 
# MAGIC with surrounding contexts. It contains setting up the sentence context 
# MAGIC or using the service context object. It also 
# MAGIC consists of setting up a vector sort index, using the source documents 
# MAGIC and the service context containing the LLM embedding 
# MAGIC model and node parser. The second part of this is 
# MAGIC actually getting the sentence window query entered, which we 
# MAGIC showed consists of getting the sentence 
# MAGIC window retriever, using the metadata replacement post processor to 
# MAGIC actually replace a node with the surrounding 
# MAGIC context, and then finally using a re-ranking module to filter 
# MAGIC for the top N results. We combine all of 
# MAGIC this using the as query intro module. Let's first call build 
# MAGIC sentence window index with the source document, them, as 
# MAGIC well as the save directory. And then let's call the 
# MAGIC second function to get the sentence 
# MAGIC window query engine. Great. Now you're ready to experiment 
# MAGIC with sentence window retrieval. In the next section, Audit Prompt 
# MAGIC will show you how to actually run evaluations using the 
# MAGIC sentence window retriever, so that you can evaluate the results 
# MAGIC and actually play around the parameters and see 
# MAGIC how that affects the performance of your engine. After running through these examples, we 
# MAGIC encourage you to add your own questions and then 
# MAGIC even define your own evaluation benchmarks just to play around 
# MAGIC with this and get a sense of how everything works. 
# MAGIC Thanks, Jerry. Now that you have set up the sentence window retriever, 
# MAGIC let's see how we can evaluate it with the 
# MAGIC RAG triad and compare its performance to the 
# MAGIC basic rag with experiment tracking. Let us now see how 
# MAGIC we can evaluate and iterate on the sentence 
# MAGIC window size parameter to make the right trade-offs between 
# MAGIC the evaluation metrics, or the quality of the app, and the 
# MAGIC cost of running the application and evaluation. We 
# MAGIC will gradually increase the sentence window size 
# MAGIC starting with 1, evaluate the successive app versions 
# MAGIC with TrueLens and the RAG triad, track experiments to pick the best 
# MAGIC sentence window size, and as we go through this exercise, we will 
# MAGIC want to note the trade-offs between token usage or cost. As we increase 
# MAGIC the window size, the token usage and cost will go up, as in many 
# MAGIC cases will context relevance. At the same time, increasing the window 
# MAGIC size in the beginning, we expect will improve context relevance 
# MAGIC and therefore will also indirectly improve 
# MAGIC groundedness. One of the reasons for that 
# MAGIC is when the retrieval step does not produce sufficiently 
# MAGIC relevant context, the LLM in the completion step will 
# MAGIC tend to fill in those gaps by leveraging 
# MAGIC its pre-existing knowledge from the pre-training stage rather than explicitly 
# MAGIC relying on the retrieved pieces of 
# MAGIC context. And this choice can result in lower groundedness scores because recall 
# MAGIC groundedness means components of the final 
# MAGIC response should be traceable back to 
# MAGIC the retrieved pieces of context. Consequently, what we expect is 
# MAGIC that as you keep increasing your sentence window size, 
# MAGIC context relevance will increase up 
# MAGIC to a certain point, as will groundedness, and then 
# MAGIC beyond that point, we will see context relevance 
# MAGIC either flatten out or decrease, and 
# MAGIC groundedness is likely going to follow a similar 
# MAGIC pattern as well. In addition, there is also 
# MAGIC a very interesting relationship between context relevance and 
# MAGIC groundedness that you can see in 
# MAGIC practice. When context relevance is low, groundedness tends to be low 
# MAGIC as well. This is because the LLM will usually try to 
# MAGIC fill in the gaps in the retrieved pieces of context by leveraging 
# MAGIC its knowledge from the pre-training stage. This results 
# MAGIC in a reduction in groundedness, even if the answers actually 
# MAGIC happen to be quite relevant. As 
# MAGIC context relevance increases, groundedness also tends to increase up 
# MAGIC to a certain point. But if the context size becomes 
# MAGIC too big, even if the context relevance is 
# MAGIC high, there could be a drop in the groundedness because the 
# MAGIC LLM can get overwhelmed with contexts that are 
# MAGIC too large and fall back on its pre-existing knowledge base from the 
# MAGIC training phase. Let us now experiment with the sentence 
# MAGIC window size. I will walk you through a notebook 
# MAGIC to load a few questions for evaluation and then gradually increase the 
# MAGIC sentence window size and observe the 
# MAGIC impact of that on the RAG triad evaluation metrics. 
# MAGIC First, we load a set of pre-generated evaluation questions. And you can see here 
# MAGIC some of these questions from this list. Next, we run the 
# MAGIC evaluations for each question in that preloaded set of evaluation questions. 
# MAGIC And then, with the true recorder object, we record the 
# MAGIC prompts, the responses, the intermediate results of the application, 
# MAGIC as well as the evaluation 
# MAGIC results, in the true database. Let's now adjust the 
# MAGIC sentence window size parameter and look at the impact of that 
# MAGIC on the different RAG triad evaluation metrics. We will first reset the 
# MAGIC true database. With this code snippet, we set the sentence window 
# MAGIC size to 1. You'll notice that in this instruction. Everything else is 
# MAGIC the same as before. Then we set 
# MAGIC the sentence window engine with the get sentence window query engine associated with 
# MAGIC this index. And next up, we are ready to 
# MAGIC set up the true recorder with the sentence window size set to 1. And 
# MAGIC this sets up the definition of all the feedback functions for the 
# MAGIC RAG triad, including answer relevance, context relevance, and groundedness. 
# MAGIC And now we have everything set 
# MAGIC up to run the evaluations for the setup 
# MAGIC with the sentence window size set to 1. 
# MAGIC Okay, that ran beautifully. Now let's look at it in the 
# MAGIC dashboard. You'll see that this instruction brings up a locally hosted StreamLit app, 
# MAGIC and you can click on the link to 
# MAGIC get to the StreamLit app. So the app leader board 
# MAGIC shows us the aggregate metrics for all the 21 records that we ran 
# MAGIC through and evaluated with TrueLens. The average latency here is 4.57 seconds. 
# MAGIC The total cost is about two cents. Total number of tokens 
# MAGIC processed is about 9,000. And you can see the 
# MAGIC evaluation metrics. The application does reasonably well in answer 
# MAGIC relevance and groundedness, but on context relevance, it's quite poor. 
# MAGIC Let's now drill down and look at 
# MAGIC the individual records that were processed by the application and evaluated. If 
# MAGIC I scroll to the right, I can see some examples where the 
# MAGIC application is not doing so well on these metrics. 
# MAGIC So let me pick this row, and then 
# MAGIC we can go deeper and examine how it's 
# MAGIC doing. So the question here is, in the context of project 
# MAGIC selection and execution, explain the difference between ready-fire and ready-fire-aim approaches. Provide 
# MAGIC examples where each approach might be more beneficial. 
# MAGIC You can see the overall response here in detail from the 
# MAGIC RAG. And then, if you scroll 
# MAGIC down, we can see the overall scores for groundedness, context relevance, 
# MAGIC and answer relevance. Two pieces of context were retrieved in this example. And for 
# MAGIC 1 of the pieces of retrieved 
# MAGIC context, context relevance is quite low. Let's 
# MAGIC drill down into that example and take a closer 
# MAGIC look. What you'll see here with this example is that the 
# MAGIC piece of context is quite small. Remember that we are using 
# MAGIC a sentence window of size 1, which means we have only added 
# MAGIC 1 sentence extra in the beginning and 1 sentence extra 
# MAGIC at the end around the retrieve piece of context. And 
# MAGIC that produces a fairly small piece of context 
# MAGIC that is missing out on important information that 
# MAGIC would make it relevant to the question that was asked. Similarly, 
# MAGIC if you look at groundedness, we will see that 
# MAGIC both of these pieces have retrieved the sentences. In 
# MAGIC the final summary, the groundedness scores are quite low. 
# MAGIC Let's pick the one with the higher groundedness 
# MAGIC score, which has a bit more justification. And 
# MAGIC if we look at this example, what we will see is 
# MAGIC there are a few sentences here in the beginning for which 
# MAGIC there is good supporting evidence in the retrieved piece of context. And so the score here 
# MAGIC is high. It's a score of 10 on a 
# MAGIC scale of 0 to 10. But then for these 
# MAGIC sentences down here, there wasn't supporting evidence, and therefore the groundedness score is 0. Let's take a concrete example. Maybe 
# MAGIC this one. It's saying it's often used in situations where the cost 
# MAGIC of execution is relatively low and where the 
# MAGIC ability to iterate and adapt quickly is more important than upfront planning. 
# MAGIC This does feel like a plausible piece of text that could be 
# MAGIC useful as part of the response to the question. However, it wasn't 
# MAGIC there in the retrieved piece of context, so 
# MAGIC it's not backed up by any supporting evidence 
# MAGIC in the retrieved context. This could possibly have been 
# MAGIC part of what the model had learned during its training phase, where either 
# MAGIC from the same document, Andrew's document here on 
# MAGIC career advice for AI, or some other source 
# MAGIC talking about the same topic, the model may 
# MAGIC have learned similar information. But it's not grounded in that 
# MAGIC it is not, the sentence is not supported by 
# MAGIC the retrieved piece of context in this particular instance. 
# MAGIC So this is a general issue when the sentence 
# MAGIC window is too small, that context relevance tends to be low. And as a 
# MAGIC consequence, groundedness also becomes low because the LLM starts 
# MAGIC making use of its pre-existing knowledge from its training phase to start answering 
# MAGIC questions instead of just relying on the supplied context. Now that 
# MAGIC I've shown you a failure mode with sentence 
# MAGIC windows set to one, I want to walk 
# MAGIC through a few more steps to 
# MAGIC see how the metrics improve as 
# MAGIC we change the sentence window size. For the purpose of 
# MAGIC going through the notebook quickly, I'm going to reload the evaluation questions, 
# MAGIC but in this instance, just set it to the one question where 
# MAGIC the model had problem, this particular question, which we just 
# MAGIC walked through with the sentence window size set at 1. And then 
# MAGIC I want to run this through with the sentence 
# MAGIC window size set to 3. This code snippet 
# MAGIC is going to set up the rag 
# MAGIC with sentence window size set up three, and also 
# MAGIC set up the true recorder for it. We now have 
# MAGIC the definition of the feedback function set up in addition 
# MAGIC to the RAG, with the sentence window set 
# MAGIC at size 3. Next up, we are going to 
# MAGIC run the evaluations with that eval for that particular 
# MAGIC evaluation question that we have looked through in 
# MAGIC some detail with the sentence window set through one, where we observe 
# MAGIC the failure mode that has run successfully. Let's now look at the results with 
# MAGIC sentence window engine set to three in the TruLens 
# MAGIC dashboard. You can see the results here, I ran 
# MAGIC it on the one record. That was the problematic record when 
# MAGIC we looked at sentence window size one. And 
# MAGIC you can see a huge increase in the context 
# MAGIC relevance. It went up from 0.57 to 0.9. Now if I select the 
# MAGIC app and look at this example in some 
# MAGIC more detail, let's now look at the same question that 
# MAGIC we looked at with sentence window set at one. Now we 
# MAGIC are at 3. Here's the full final response. Now if you look at the 
# MAGIC retrieved pieces of context, you'll notice that this particular piece 
# MAGIC of retrieved context is similar to the one 
# MAGIC that we had retrieved earlier with sentence window set at size 
# MAGIC 1. But now it has the expansion because of the 
# MAGIC bigger sentence window size. For this section, we'll see 
# MAGIC that this context got a context-relevant score of 0.9, 
# MAGIC which is higher than the score of 0.8 that the smaller context 
# MAGIC had gotten earlier. And this example shows that with an 
# MAGIC expansion in the sentence window size, even reasonably good 
# MAGIC pieces of retrieved context can get even better. Once the completion step is 
# MAGIC done with these significantly better pieces of context, the groundedness score goes 
# MAGIC up quite a bit. We'd see that by 
# MAGIC finding supporting evidence across these two pieces of highly 
# MAGIC relevant context, the groundedness score actually goes up all 
# MAGIC the way to one. So increasing the sentence 
# MAGIC window size from one to three led 
# MAGIC to a substantial improvement in the evaluation metrics of the RAG triad. 
# MAGIC Both groundedness and context relevance went up significantly, as 
# MAGIC did answer relevance. And now we can look 
# MAGIC at sentence window set to five. If you look 
# MAGIC at the metrics here, a couple of things to 
# MAGIC observe. One is the total tokens has gone up, 
# MAGIC and this could have an impact on the 
# MAGIC cost if you were to increase 
# MAGIC the number of records. So that's one of the trade-offs that 
# MAGIC I mentioned earlier. As you increase the sentence window size, it gets more 
# MAGIC expensive because more tokens are being processed by the LLMs during evaluation. 
# MAGIC The other thing to observe is that while context relevance, and answer relevance have 
# MAGIC remained flat, groundedness has actually dropped with the increase in the sentence 
# MAGIC window size. And this can happen after a certain point 
# MAGIC because, as the context size increases, the LLM can 
# MAGIC get overwhelmed in the completion step with too 
# MAGIC much information. And in the process of summarization, 
# MAGIC it can start introducing its own 
# MAGIC pre-existing knowledge instead of just the information in the 
# MAGIC retrieved pieces of context. So to wrap things up 
# MAGIC here, it turns out that as we gradually increase 
# MAGIC the sentence window size from 1 to 3 to 
# MAGIC 5, 3, the size of 3 is the best choice for us for 
# MAGIC this particular evaluation. And we see the increase in 
# MAGIC context relevance and answer relevance and groundedness as we go 
# MAGIC from one to 3, and then a reduction 
# MAGIC or degradation in the groundedness step with a further increase to a size of five. 
# MAGIC As you are playing with the notebook, we encourage you 
# MAGIC to rerun it with more records in these 
# MAGIC two steps. Examine the individual records which are causing problems for specific 
# MAGIC metrics like context relevance or groundedness, and get some intuition and build 
# MAGIC some intuition around why the failure modes are happening and what to do to 
# MAGIC address them, and in the next section, we 
# MAGIC will look at another advanced rack technique auto 
# MAGIC merging to address some of those 
# MAGIC failure modes. Irrelevant context can creep into 
# MAGIC the final response, resulting in not such great scores 
# MAGIC in groundedness or answer relevance. 
# MAGIC
