# Databricks notebook source
"""Retrieval
Retrieval is the centerpiece of our retrieval augmented generation (RAG) flow.

Let's get our vectorDB from before."""

# COMMAND ----------

"""Vectorstore retrieval"""

# COMMAND ----------

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# COMMAND ----------

#!pip install lark

# COMMAND ----------

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

# COMMAND ----------

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# COMMAND ----------

print(vectordb._collection.count())

# COMMAND ----------

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

# COMMAND ----------

question = "Tell me about all-white mushrooms with large fruiting bodies"

# COMMAND ----------

smalldb = Chroma.from_texts(texts, embedding=embedding)

# COMMAND ----------

smalldb.similarity_search(question, k=2)

# COMMAND ----------

smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)

# COMMAND ----------

"""Addressing Diversity: Maximum marginal relevance
Last class we introduced one problem: how to enforce diversity in the search results.

Maximum marginal relevance strives to achieve both relevance to the query and diversity among the results."""

# COMMAND ----------

question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)

# COMMAND ----------

docs_ss[0].page_content[:100]

# COMMAND ----------

docs_ss[1].page_content[:100]

# COMMAND ----------

"""Note the difference in results with MMR."""

# COMMAND ----------

docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)

# COMMAND ----------

docs_mmr[0].page_content[:100]

# COMMAND ----------

docs_mmr[1].page_content[:100]

# COMMAND ----------

"""Addressing Specificity: working with metadata
In last lecture, we showed that a question about the third lecture can include results from other lectures as well.

To address this, many vectorstores support operations on metadata.

metadata provides context for each embedded chunk."""

# COMMAND ----------

question = "what did they say about regression in the third lecture?"

# COMMAND ----------

docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)

# COMMAND ----------

for d in docs:
    print(d.metadata)

# COMMAND ----------

"""Addressing Specificity: working with metadata using self-query retriever
But we have an interesting challenge: we often want to infer the metadata from the query itself.

To address this, we can use SelfQueryRetriever, which uses an LLM to extract:

The query string to use for vector search
A metadata filter to pass in as well
Most vector databases support metadata filters, so this doesn't require any new databases or indexes."""



# COMMAND ----------

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# COMMAND ----------

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

# COMMAND ----------

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# COMMAND ----------

question = "what did they say about regression in the third lecture?"

# COMMAND ----------

"""You will receive a warning about predict_and_parse being deprecated the first time you executing the next line. This can be safely ignored."""

# COMMAND ----------

docs = retriever.get_relevant_documents(question)

# COMMAND ----------

for d in docs:
    print(d.metadata)

# COMMAND ----------

"""Additional tricks: compression
Another approach for improving the quality of retrieved docs is compression.

Information most relevant to a query may be buried in a document with a lot of irrelevant text.

Passing that full document through your application can lead to more expensive LLM calls and poorer responses.

Contextual compression is meant to fix this."""

# COMMAND ----------

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# COMMAND ----------

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# COMMAND ----------

# Wrap our vectorstore
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# COMMAND ----------

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

# COMMAND ----------

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

# COMMAND ----------

"""Combining various techniques"""

# COMMAND ----------

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

# COMMAND ----------

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

# COMMAND ----------

"""Other types of retrieval
It's worth noting that vectordb as not the only kind of tool to retrieve documents.

The LangChain retriever abstraction includes other ways to retrieve documents, such as TF-IDF or SVM."""

# COMMAND ----------

from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# COMMAND ----------

# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)


# COMMAND ----------

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

# COMMAND ----------

question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_svm[0]

# COMMAND ----------

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


