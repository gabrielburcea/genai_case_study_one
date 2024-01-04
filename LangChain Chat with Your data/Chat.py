# Databricks notebook source
"""Chat
Recall the overall workflow for retrieval augmented generation (RAG):
We discussed Document Loading and Splitting as well as Storage and Retrieval.

We then showed how Retrieval can be used for output generation in Q+A using RetrievalQA chain"""


# COMMAND ----------

import os
import openai
import sys
sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# COMMAND ----------

"""The code below was added to assign the openai LLM version filmed until it is deprecated, currently in Sept 2023. LLM responses can often vary, but the responses may be significantly different when using a different model version."""


import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

# COMMAND ----------

"""If you wish to experiment on LangChain plus platform:

Go to langchain plus platform and sign up
Create an api key from your account's settings
Use this api key in the code below"""

# COMMAND ----------

#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..."

# COMMAND ----------

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# COMMAND ----------

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

# COMMAND ----------

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)
llm.predict("Hello world!")

# COMMAND ----------

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "Is probability a class topic?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


result = qa_chain({"query": question})
result["result"]

# COMMAND ----------

"""Memory"""

# COMMAND ----------

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# COMMAND ----------

"""ConversationalRetrievalChain"""

# COMMAND ----------

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

# COMMAND ----------

question = "Is probability a class topic?"
result = qa({"question": question})

# COMMAND ----------

result['answer']

# COMMAND ----------

question = "why are those prerequesites needed?"
result = qa({"question": question})

# COMMAND ----------

result['answer']

# COMMAND ----------

"""Create a chatbot that works on your documents"""

# COMMAND ----------

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 


# COMMAND ----------

import panel as pn
import param

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file,"stuff", 4)
    
    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return 

# COMMAND ----------

"""Create a chatbot"""

cb = cbfs()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput( placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp) 

jpg_pane = pn.pane.Image( './img/convchain.jpg')

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2= pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources ),
)
tab3= pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4=pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
)
dashboard

# COMMAND ----------

"""Feel free to copy this code and modify it to add your own features. You can try alternate memory and retriever models by changing the configuration in load_db function and the convchain method. Panel and Param have many useful features and widgets you can use to extend the GUI."""

Acknowledgments
Panel based chatbot inspired by Sophia Yang, github

# COMMAND ----------

"""Transcript 
We're so close to having a functional chatbot. We 
started with loading documents, then we split them, then 
we created a vector store, we talked about different types of retrieval, 
we've shown that we can answer questions, but we 
just can't handle follow-up questions, we can't have a real conversation 
with it. The good news is, we're going to fix that in 
this lesson. Let's figure out how. 
We're now going to finish up by creating a 
question answering chatbot. What this is going to do is, it's going 
to look very similar to before, but we're going to add in this concept of 
chat history. And this is any previous conversations 
or messages that you've exchanged with the 
chain. 
What that's going to allow it to do, is it's going to allow it to 
take that chat history into context when it's trying to 
answer the question. So if you're asking a follow-up question, it'll know what 
you're talking about. 
An important thing to note here is that 
all the cool types of retrieval that we 
talked about up to this point, like self-query or compression or 
anything like that, you can absolutely use them here. All 
the components we talked about are very modular and can fit together 
nicely. We're just adding in this concept 
of chat history. 
Let's see what it looks like. First, as always, we're going to load our 
environment variables. 
If you have the platform set up, it might also be nice to 
turn it on from the beginning. 
There'll be a lot of cool things that we'll want 
to see what's going on under the hood. 
We're going to load our vector store that has 
all the embeddings for all the class materials. 
We can run through basic similarity search on the vector store. 
We can initialize the language model that we're 
going to use as our chatbot. And then, and 
this is all from before, which is why I'm skimming 
through it so quickly. We can initialize a prompt template, create a 
retrieval QA chain, and then pass in a question and 
get back a result. 
But now let's do more. Let's add some memory to it. So we're going to 
be working with conversation buffer memory. And what 
this does is it's just going to simply keep a list, a buffer of chat 
messages in history, and it's going to pass those along with the question 
to the chatbot every time. 
 
We're going to specify memory key, chat history. This is just going to line 
it up with an input variable on the prompt. And 
then we're going to specify return messages equal true. This is 
going to return the chat history as a 
list of messages as opposed to a single string. This is 
the simplest type of memory. For a more in-depth look at memory, go back to 
the first class that I taught with Andrew. We covered it 
in detail then. 
Let's now create a new type of chain, the 
conversational retrieval chain. We pass in the language model, we 
pass in the retriever, and we pass in memory. The 
conversational retrieval chain adds a new bit on top of the retrieval 
QA chain, not just memory. 
Specifically, what it adds is it adds a step 
that takes the history and the new question 
and condenses it into a stand-alone question to pass to 
the vector store to look up relevant documents. 
We'll take a look at this in the UI after we run it 
and see what effect it has. But for now, let's try it out. We can 
ask a question. This is without any history, and see 
the result we get back. And then we can ask a follow-up question to 
that answer. This is the same as before. So we're asking, is probability 
a class topic? We get some answer. 
The instructor assumes that students have basic 
understanding of probability and statistics. And 
then we ask, why are those prerequisites 
needed? We get back a result, and let's take a look at 
it. We get back an answer, and now we can see that the answer 
is referring to basic probability and statistics as prerequisites 
and expanding upon that, not getting 
confused with computer science as it 
had before. 
Let's take a look at what's going on under the hood in the UI. So here, 
we can already see that there's a bit 
more complexity. We can see that the input to the 
chain now has not only the question, but also chat 
history. Chat history is coming from the memory, and this gets 
applied before the chain is invoked and logged in this logging system. 
If we look at the trace, we can see 
that there's two separate things going on. There's first a call 
to an LLM, and then there's a call to the stuff 
documents chain. 
Let's take a look at the first call. We can see here a prompt with some instructions. 
Given the following conversation, a 
follow-up question, rephrase the follow-up question to 
be a stand-alone question. Here, we have the history 
from before. So we have the question we asked first, is probability a class 
topic? We then have the assistance answers. And then out here, 
we have the stand-alone question. What is the reason for 
requiring basic probability and statistics as prerequisites for the 
class? 
What happens is that stand-alone answer is then passed 
into the retriever, and we retrieve four documents 
or three documents or however many we specify. We 
then pass those documents to the stuff documents 
chain and try to answer the original question. So 
if we look into that, we can see that we have the system answer, 
use the following pieces of context to answer the user's question. 
We've got a bunch of context. And then we 
have the stand-alone question down below. And then we get 
an answer. And here's the answer that is relevant for the question at 
hand, which is about probability and statistics as prerequisites. 
 
This is a good time to pause and try out 
different options for this chain. You can pass in different prompt 
templates, not only for answering the question, 
but also for rephrasing that into a stand-alone question. You 
can try different types of memory, lots of different options to 
pull out here. After this, we're going to put it all together 
in a nice UI. There's going to be a lot of code for 
creating this UI, but this is the main important bit 
right here. Specifically, this is a 
full walkthrough of basically this whole class. So we're going 
to load a database and retriever chain. We're going to pass in 
a file. We're going to load it with the PDF loader. We're then going to load it 
into documents. We're going to split those documents. We're 
going to create some embeddings and put 
it in a vector store. 
 
We're then going to turn that vector store into a retriever. We're 
going to use similarity here with some "search_kwargs=k", which we're 
going to set equal to a parameter that we can 
pass in. And then we're going to create the conversational retrieval chain. One important 
thing to note here is that we're not 
passing in memory. We're going to manage memory externally for the 
convenience of the GUI below. That means that 
chat history will have to be managed outside the 
chain. 
We then have a lot more code here. We're not going to spend too 
much time on it, but pointing out that here we're passing 
in chat history into the chain. And again, that's because 
we don't have memory attached to it. And then here we're 
extending chat history with the result. We can then put it 
all together and run this to get a nice UI through which we can interact with 
our chatbot. 
Let's ask it a question. Who are the TAs? The 
TAs are Paul Baumstarck, Catie Chang. 
You'll notice here that there's a few tabs that we can also click 
on to see other things. So if we click on the database, we can 
see the last question we asked of the database, as well 
as the sources we got back from the lookup there. So these 
are the documents. These are after the splittings 
happened. These are each chunk that we've retrieved. We 
can see the chat history with the input 
and the output. And then there's also a place 
to configure it where you can upload files. 
We can also ask follow-ups. So let's ask, what are their majors? And we 
get back an answer about the previously mentioned TAs. So we 
can see that Paul is studying machine learning and computer vision, while 
Catie is actually a neuroscientist. 
This is basically the end of the class. So now's a great time to pause, ask 
it a bunch more questions, upload your own documents here, and 
enjoy this end-to-end question answering bot, complete with an 
amazing notebook UI. 
And that brings this class on LangChain, chat with your data, to an end. 
In this class, we've covered how to use 
LangChain to load data from a variety of 
document sources using LangChain's 80-plus different document 
loaders. From there, we split the documents into chunks and 
talk about a lot of the nuances that arrive when doing so. 
 
After that, we take those chunks, create embeddings for them, 
and put them into a vector store, showing how that 
easily enables semantic search. But we also talk about some of the 
downsides of semantic search and where it can 
fail on the certain edge cases that arise. 
The next thing that we cover is retrieval, 
maybe my favorite part of the class, where we talk about a lot of new 
and advanced and really fun retrieval algorithms for 
overcoming those edge cases. We combine that with 
LLMs in the next session, where we take 
those retrieved documents, we take the user question, 
we pass it to an LLM, and we generate an 
answer to the original question. But there's 
still one thing missing, which is the conversational 
aspect of it. 
And that's where we finish out the class, by 
creating a fully functioning end-to-end chatbot over your data. I've really enjoyed teaching 
this class. I hope you guys have enjoyed taking it. And I want to 
thank everyone in the open source who has contributed a lot of 
the things that make this class possible, like 
all the prompts and a lot of the functionality that you 
see. 
As you guys build with LangChain and discover 
new ways of doing things and new tips and techniques, 
I hope that you share what you'll learn on Twitter or even open 
up a PR in LangChain. 
It's a really fast-moving field, and it's an exciting 
time to be building. 
Look forward to seeing everything. 


"""
