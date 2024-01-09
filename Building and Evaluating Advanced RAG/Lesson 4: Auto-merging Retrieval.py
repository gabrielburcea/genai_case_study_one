# Databricks notebook source
"""Lesson 4: Auto-merging Retrieval"""

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
Auto-merging retrieval setup

from llama_index import Document
​
document = Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index.node_parser import HierarchicalNodeParser
​
# create the hierarchical node parser w/ default settings
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

nodes = node_parser.get_nodes_from_documents([document])

from llama_index.node_parser import get_leaf_nodes
​
leaf_nodes = get_leaf_nodes(nodes)
print(leaf_nodes[30].text)

nodes_by_id = {node.node_id: node for node in nodes}
​
parent_node = nodes_by_id[leaf_nodes[30].parent_node.node_id]
print(parent_node.text)
Building the index

from llama_index.llms import OpenAI
​
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

from llama_index import ServiceContext
​
auto_merging_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)

from llama_index import VectorStoreIndex, StorageContext
​
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)
​
automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context, service_context=auto_merging_context
)
​
automerging_index.storage_context.persist(persist_dir="./merging_index")

# This block of code is optional to check
# if an index file exist, then it will load it
# if not, it will rebuild it
​
import os
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import load_index_from_storage
​
if not os.path.exists("./merging_index"):
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
​
    automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            service_context=auto_merging_context
        )
​
    automerging_index.storage_context.persist(persist_dir="./merging_index")
else:
    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./merging_index"),
        service_context=auto_merging_context
    )
​
Defining the retriever and running the query engine

from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
​
automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)
​
retriever = AutoMergingRetriever(
    automerging_retriever, 
    automerging_index.storage_context, 
    verbose=True
)
​
rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")
​
auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[rerank]
)

auto_merging_response = auto_merging_engine.query(
    "What is the importance of networking in AI?"
)

from llama_index.response.notebook_utils import display_response
​
display_response(auto_merging_response)
Putting it all Together

import os
​
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine
​
​
def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
​
    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index
​
​
def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

from llama_index.llms import OpenAI
​
index = build_automerging_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./merging_index",
)
​

query_engine = get_automerging_query_engine(index, similarity_top_k=6)
TruLens Evaluation

from trulens_eval import Tru
​
Tru().reset_database()
Two layers

auto_merging_index_0 = build_automerging_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_0",
    chunk_sizes=[2048,512],
)

auto_merging_engine_0 = get_automerging_query_engine(
    auto_merging_index_0,
    similarity_top_k=12,
    rerank_top_n=6,
)

from utils import get_prebuilt_trulens_recorder
​
tru_recorder = get_prebuilt_trulens_recorder(
    auto_merging_engine_0,
    app_id ='app_0'
)

eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

run_evals(eval_questions, tru_recorder, auto_merging_engine_0)

from trulens_eval import Tru
​
Tru().get_leaderboard(app_ids=[])

Tru().run_dashboard()
Three layers

auto_merging_index_1 = build_automerging_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_1",
    chunk_sizes=[2048,512,128],
)

auto_merging_engine_1 = get_automerging_query_engine(
    auto_merging_index_1,
    similarity_top_k=12,
    rerank_top_n=6,
)
​

tru_recorder = get_prebuilt_trulens_recorder(
    auto_merging_engine_1,
    app_id ='app_1'
)

run_evals(eval_questions, tru_recorder, auto_merging_engine_1)

from trulens_eval import Tru
​
Tru().get_leaderboard(app_ids=[])

Tru().run_dashboard()

# COMMAND ----------

# MAGIC %md
# MAGIC In this lesson, we'll do a deep dive into 
# MAGIC another advanced RAG technique, auto-merging. An issue with the naive approach 
# MAGIC is that you're retrieving a bunch of fragmented context chunks to 
# MAGIC put into the LLM context menu, and the fragmentation 
# MAGIC is worse the smaller your chunk size. Here, we use an 
# MAGIC auto-merging heuristic to merge smaller chunks into 
# MAGIC a bigger parent chunk to help ensure more coherent context. Let's check 
# MAGIC out how to set it up. In this section, we'll talk 
# MAGIC about auto-merging retrieval. An issue with the standard 
# MAGIC RAG pipeline is that you're retrieving a 
# MAGIC bunch of fragmented context chunks put into the LLM context window, 
# MAGIC and the fragmentation is worse the smaller your 
# MAGIC chunk size. For instance, you might get back two or 
# MAGIC more retrieved context chunks in roughly the same section, but there's 
# MAGIC actually no guarantees on the ordering of these 
# MAGIC chunks. This can potentially hamper the alum's ability to synthesize 
# MAGIC over this retrieved context within its context window. So what 
# MAGIC auto-merging retrieval does is the following. First, define 
# MAGIC a hierarchy of smaller chunks linking to bigger parent 
# MAGIC chunks, where each parent chunk can have some number of 
# MAGIC children. Second, during retrieval, if the set 
# MAGIC of smaller chunks linking to a parent chunk 
# MAGIC exceeds some percentage threshold, then we merge smaller chunks into the bigger 
# MAGIC parent chunk. So we retrieve the bigger parent 
# MAGIC chunk instead to help ensure more coherent context. Now let's check 
# MAGIC out how to set this up. This notebook will introduce 
# MAGIC the various components needed to construct an auto-merging retriever with Llama 
# MAGIC index. The various components will be 
# MAGIC covered in detail. And similar to the previous 
# MAGIC section, at the end, Adupam will show you 
# MAGIC how to experiment with parameters and evaluation with TruEra. Similar to before, 
# MAGIC we'll load in the OpenAI API 
# MAGIC key, and we'll load this using a convenience 
# MAGIC helper function in our utils file. As with the previous lessons, we'll also 
# MAGIC use the how to build a career in 
# MAGIC AI PDF. And as before, we also encourage you to try out 
# MAGIC your own PDF files as well. We load in 
# MAGIC 41 document objects, and we'll merge them into a single large document, 
# MAGIC which makes this more amenable for text blending with our advanced retrieval methods. 
# MAGIC Now we're ready to set up our auto-merging retriever. This 
# MAGIC will consist of a few different components, 
# MAGIC and the first step is to define what we call a 
# MAGIC hierarchical node parser. In order to use an 
# MAGIC auto-version retriever, we need to parse our nodes 
# MAGIC in a hierarchical fashion. This means that nodes are parsed in decreasing 
# MAGIC sizes and contain relationships to their parent 
# MAGIC node. Here we demonstrate how the node parser works with a 
# MAGIC small example. We create a toy 
# MAGIC parser with small chunk sizes to demonstrate. Note 
# MAGIC that the chunk sizes we use are 20, 48, 5, 12, 
# MAGIC and 128. You can change the chunk sizes to any sort of 
# MAGIC decreasing order that you'd like. Here we do it by a factor of four. 
# MAGIC Now let's get the set of nodes from the document. 
# MAGIC What this does is this actually returns all nodes. This returns 
# MAGIC all leaf nodes, intermediate nodes, as well as parent nodes. So 
# MAGIC there's going to be a decent amount of overlap 
# MAGIC of information and content between the leaf, 
# MAGIC intermediate, and parent nodes. If we only want to retrieve the leaf nodes, we can 
# MAGIC call a function within Llama index called "gat 
# MAGIC leaf nodes," and we can take a look 
# MAGIC at what that looks like. In this example, we 
# MAGIC call gat leaf nodes on the original set of nodes. And we take 
# MAGIC a look at the 31st node to look at 
# MAGIC the text. We see that the text trunk is actually fairly 
# MAGIC small. And this is an example of a leaf node, because a leaf node 
# MAGIC is the smallest chunk size of 128 tokens. Here's how 
# MAGIC you might go about strengthening your math background to figure out what's important 
# MAGIC to know, etc. Now that we've shown what a 
# MAGIC leaf node looks like, we can also explore 
# MAGIC the relationships. We can print the parent of the above node 
# MAGIC and observe that it's a larger chunk containing the text of 
# MAGIC the leaf node, but also more. More concretely, the parent 
# MAGIC node contains 512 tokens, while having four leaf nodes 
# MAGIC that contain 128 tokens. There's four leaf nodes because the chunk 
# MAGIC sizes are divided by a factor of four each time. This 
# MAGIC is an example of what the parent node of 
# MAGIC the 31st leaf node looks like. Now that we've 
# MAGIC shown you what the node hierarchy looks like, we 
# MAGIC can now construct our index. We'll use the OpenAI LLM 
# MAGIC and specifically GPT 3.5 Turbo. We'll also define a service context object containing 
# MAGIC the LLM embedding model and the hierarchical node parser. As with the 
# MAGIC previous notebooks, we'll use the "bge small en embedding model." The 
# MAGIC next step is to construct our index. 
# MAGIC The way the index works is 
# MAGIC that we actually construct a vector index on specifically the leaf 
# MAGIC nodes. All other intermediate and parent nodes are 
# MAGIC stored in a doc store and are retrieved dynamically during retrieval. 
# MAGIC But what we actually fetch during the initial top 
# MAGIC K embedding lookup is specifically the 
# MAGIC leaf nodes, and that's what we embed. You see 
# MAGIC in this code that we define a storage context 
# MAGIC object, which by default is initialized with an in-memory document store. And 
# MAGIC we call "storage_context.docstore.addDocuments "to add all nodes to this in-memory doc 
# MAGIC store. However, when we create our vector store index, called auto-merging index, 
# MAGIC right here, we only pass in the leaf 
# MAGIC nodes for vector indexing. This means that, specifically, the 
# MAGIC leaf nodes are embedded using the embedding model 
# MAGIC and also indexed. But we also pass in the storage context 
# MAGIC as well as the service context. And so the vector index 
# MAGIC does have knowledge of the underlying doc store that 
# MAGIC contains all the nodes. And finally, we persist 
# MAGIC this index. If you've already built this index 
# MAGIC and you want to load it from storage, you can 
# MAGIC just copy and paste this block of code, which 
# MAGIC will rebuild the index if it doesn't exist or 
# MAGIC load it from storage. The last step now that we've defined the 
# MAGIC auto-merging index is to set up the retriever and run the query engine. The 
# MAGIC auto-merging retriever is what controls the merging logic. If 
# MAGIC a majority of children nodes are retrieved for a given parent, 
# MAGIC they are swapped out for the parent instead. In order 
# MAGIC for this merging to work well, we set 
# MAGIC a large top-k for the leaf nodes. And remember, the leaf 
# MAGIC nodes also have a smaller chunk size of 
# MAGIC 128. In order to reduce token usage, we 
# MAGIC apply a re-ranker after the merging has taken 
# MAGIC place. For example, we might retrieve the top 12, merge and have a 
# MAGIC top 10, and then re-rank into a top 6. The top end 
# MAGIC for the re-ranker may seem larger, but remember that the base chunk 
# MAGIC size is only 128 tokens, and then the next parent above 
# MAGIC that is 512 tokens. We import a class 
# MAGIC called auto-merging retriever, and then we define a sentence transformer re-rank module. We combine 
# MAGIC both the auto-merging retriever and the re-rank module into our 
# MAGIC retriever query engine, which handles both retrieval and synthesis. Now that 
# MAGIC we've set this whole thing up end-to-end, let's actually test 
# MAGIC what is the importance of networking in AI 
# MAGIC as an example question, we get back a 
# MAGIC response. We see that it says networking 
# MAGIC is important in AI because it allows individuals 
# MAGIC to build a strong professional network and 
# MAGIC more. The next step is to 
# MAGIC put it all together. And we'll create two high-level functions, build auto-merging index, 
# MAGIC as well as get auto-merging query engine. And this 
# MAGIC basically captures all the steps that we just showed you in 
# MAGIC the first function, build auto-merging index. We'll use the hierarchical node parser 
# MAGIC to parse out the hierarchy of child 
# MAGIC to parent nodes we'll define the service context and we'll create a vector store index 
# MAGIC from the leaf nodes but also linking to the document store 
# MAGIC of all the nodes the second function, 
# MAGIC get auto-merging query engine, leverages our auto merging 
# MAGIC retriever which is able to dynamically merge leaf nodes 
# MAGIC into parent nodes and also use our re-rank 
# MAGIC module and then combine it with the overall retriever 
# MAGIC query engine. So we build the index using the build auto-merging 
# MAGIC index function using the original source document, the 
# MAGIC LLM set to GPT 3.5 turbo, as well 
# MAGIC as the merging index as a save directory. And 
# MAGIC then for the query engine, we call get auto 
# MAGIC merging query engine based on the index, as well 
# MAGIC as we set a similarity top K of equal to six. 
# MAGIC As a next step, Anupam will show you 
# MAGIC how to evaluate the auto-merging retriever and also iterate 
# MAGIC on parameters using TruEra. We encourage you to try out your own 
# MAGIC questions as well and also iterate on the parameters of 
# MAGIC auto-merging retrieval. For instance, what happens when you change the trunk 
# MAGIC sizes or the top K or the top N for the re-ranker? 
# MAGIC Play around with it and tell us what the results are. 
# MAGIC That was awesome, Jerry. Now that you have set up the auto-merging 
# MAGIC retriever, let's see how we can evaluate it with the RAG 
# MAGIC triad and compare its performance to the basic 
# MAGIC RAG with experiment tracking. Let's set up 
# MAGIC this auto-merging new index. You'll notice that it's two 
# MAGIC layers. The lowest layer chunk, the leaf nodes, will have a 
# MAGIC chunk size of 512, and the next layer up in 
# MAGIC the hierarchy is a chunk size of 2048, in the hierarchy is 
# MAGIC a chunk size of 2048, meaning that each parent will have 
# MAGIC four leaf nodes of 512 tokens each. The 
# MAGIC other pieces of setting this up are exactly the 
# MAGIC same as what Jerry has shown you earlier. 
# MAGIC One reason you may want to experiment with the two-layer auto-merging 
# MAGIC structure is that it's simpler. Less work 
# MAGIC is needed to create the index, as well as in the 
# MAGIC retrieval step, there is less work needed because all the third-layer 
# MAGIC checks go away. If it performs comparably well, then 
# MAGIC ideally we want to work with a simpler structure. 
# MAGIC Now that we have created the index with this two-layer 
# MAGIC auto-merging structure. Let's set up the auto-merging engine for this setup. I'm 
# MAGIC keeping the top K at the same value 
# MAGIC as before, which is 12. And the re-ranking step will also 
# MAGIC have the same and equal six. This will let us do a 
# MAGIC more direct head to head comparison between this application setup 
# MAGIC and the three-layer auto-merging hierarchy app that Jerry had set up earlier. 
# MAGIC Now let's set up the Tru Recorder with this 
# MAGIC auto-merging engine and we will give this an app ID of 
# MAGIC app 0. Let's now load some questions for evaluation 
# MAGIC from the generated questions.txt file that we have set up earlier, now we can 
# MAGIC define the running of these evaluation questions for each 
# MAGIC question in eval, we are going to set things up 
# MAGIC so that the Tru Recorder object, when invoked with 
# MAGIC the run evals, will record the prompts, responses, and the evaluation results, leveraging the 
# MAGIC query engine. Now that our evaluations have completed, let's take a 
# MAGIC look at the leaderboard. We can see that app 
# MAGIC 0 metrics here. Context relevance seems low. 
# MAGIC The other two metrics are better. This is with our two-level 
# MAGIC hierarchy with 512 as the leaf node chunk size and the parent 
# MAGIC being 2048 tokens, so four leaf nodes per parent node now 
# MAGIC we can run the true dashboard and take a look at the 
# MAGIC evaluation results at the record level at the 
# MAGIC next layer of detail. Let's examine the app leaderboard. You can see 
# MAGIC here that after processing 24 records, the context 
# MAGIC relevance at an aggregate level is quite low, although 
# MAGIC the app is doing better on answer relevance and 
# MAGIC groundedness. I can select the app. Let's now look at 
# MAGIC the individual records of app 0 and see how the evaluation 
# MAGIC scores are for the various 
# MAGIC records. You can scroll to the right here and look 
# MAGIC at the scores for answer relevance, context relevance, 
# MAGIC and groundedness. Let's pick one that has low context 
# MAGIC relevance. So here's one. If you click on it, you'll 
# MAGIC see the more detailed view down below. 
# MAGIC The question is discuss the importance of 
# MAGIC budgeting for resources and the successful execution of AI projects. On the right here 
# MAGIC is the response. And if you scroll down 
# MAGIC further, you can see a more 
# MAGIC detailed view for context relevance. There were six pieces of retrieve 
# MAGIC context. Each of them is scored to be particularly low in their evaluation 
# MAGIC scores. It's between 0 and 0.2. And if you pick 
# MAGIC any of them and click on it you can see that 
# MAGIC the response is not particularly relevant to the question that 
# MAGIC was asked. You can also scroll back up and explore some of the other records you can pick 
# MAGIC ones, for example, where the scores are good, 
# MAGIC like this one here, and explore how the application is doing 
# MAGIC on various questions, and where its strengths are, 
# MAGIC where its failure modes are, to build 
# MAGIC some intuition around what's working and what's not. Let's now compare the 
# MAGIC previous app to the auto-merging setup that 
# MAGIC Jerry introduced earlier. We will have three layers now in 
# MAGIC the hierarchy, starting with 128 tokens at the leaf node level 512, one 
# MAGIC layer up, and 2048 at the highest layer, so 
# MAGIC at each layer each parent has 
# MAGIC four children. Now let's set up the query engine 
# MAGIC for this app setup, the True Recorder, all identical steps 
# MAGIC as the one for the previous app. And finally, 
# MAGIC we are in a position to run the evaluations. 
# MAGIC Now that we have app 1 set up, we can take a 
# MAGIC quick look. And the total cost is also 
# MAGIC about half. And that's because, recall, this has three layers in 
# MAGIC the hierarchy and the chunk size is 128 tokens instead of the 512, 
# MAGIC which is the smallest leaf node token size for app 0. So that results 
# MAGIC in a cost reduction. Notice also that context relevance 
# MAGIC has increased by about 20%. And part of 
# MAGIC the reason that's happening is that And part of the reason 
# MAGIC that's happening is that the merging is likely happening a lot better with 
# MAGIC this new app setup. We can 
# MAGIC also drill down and look at app one in greater 
# MAGIC detail. Like before, we can look at individual records. Let's 
# MAGIC pick the same one that we looked at 
# MAGIC earlier, the tab 0. It's the question about the 
# MAGIC importance of budgeting. And now you can see context relevance is doing better. 
# MAGIC Groundedness is also considerably higher. And if we 
# MAGIC pick a sample example response here, you'll see that, in fact, 
# MAGIC it is talking very specifically about budgeting for resources. So there is 
# MAGIC improvement in this particular instance and also at an aggregate 
# MAGIC level. Let me now summarize some of the key 
# MAGIC takeaways from Lesson 4. We walked you through an approach to evaluate and iterate 
# MAGIC with the auto-retrieval advanced RAG technique. And in particular, 
# MAGIC we showed you how to iterate with different hierarchical 
# MAGIC structures, the number of levels, the 
# MAGIC number of child nodes, and chunk sizes. And 
# MAGIC for these different app versions, you could evaluate 
# MAGIC them with the RAG triad and track experiments to 
# MAGIC pick the best structure for your use case. One 
# MAGIC thing to notice is that not only are 
# MAGIC you getting the metrics associated with the RAG 
# MAGIC triad as part of the evaluation, but the drill down 
# MAGIC into the record level can help you gain 
# MAGIC intuition about hyperparameters that work best with certain doc types. For 
# MAGIC example, depending on the nature of the document, such 
# MAGIC as employment contracts versus invoices, you might find that 
# MAGIC different chunk sizes and hierarchical structures work best. Finally, one other 
# MAGIC thing to note is that auto-merging 
# MAGIC is complementary to sentence window retrieval. And one way to think about 
# MAGIC that is, let's say you have four 
# MAGIC child nodes of a parent. With auto-merging, you might 
# MAGIC find that child number one and child 
# MAGIC number four are very relevant to the query that was asked. 
# MAGIC And these then get merged under the auto-merging paradigm. In contrast, sentence 
# MAGIC windowing may not result in this kind of merging because they are not 
# MAGIC in a contiguous section of the text. That brings 
# MAGIC us to the end of Lesson 4. We 
# MAGIC have observed that with advanced rag techniques such 
# MAGIC as sentence windowing and auto-merging retrieval augmented with 
# MAGIC the power of evaluation, experiment tracking, and iteration, 
# MAGIC you can significantly improve your rag applications. Improve 
# MAGIC your RAG applications. In addition, while the course has focused on these two techniques 
# MAGIC and the associated RAG triad for evaluation, there are 
# MAGIC a number of other evaluations that you can play 
# MAGIC with in order to ensure that your LLM 
# MAGIC applications are honest, harmless, and helpful. This 
# MAGIC slide has a list of some of the ones that 
# MAGIC are available out of the box in TrueLens. 
# MAGIC We encourage you to go play with TrueLens, explore the 
# MAGIC notebooks, and take your learning to the next level. 
# MAGIC
