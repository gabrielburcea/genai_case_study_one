# Databricks notebook source
'''Lab 4 - Cross-encoder re-ranking'''
from helper_utils import load_chroma, word_wrap, project_embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
embedding_function = SentenceTransformerEmbeddingFunction()
​
chroma_collection = load_chroma(filename='microsoft_annual_report_2022.pdf', collection_name='microsoft_annual_report_2022', embedding_function=embedding_function)
chroma_collection.count()
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
349
Re-ranking the long tail
query = "What has been the investment in research and development?"
results = chroma_collection.query(query_texts=query, n_results=10, include=['documents', 'embeddings'])
​
retrieved_documents = results['documents'][0]
​
for document in results['documents'][0]:
    print(word_wrap(document))
    print('')
• operating expenses increased $ 1. 5 billion or 14 % driven by
investments in gaming, search and news advertising, and windows
marketing. operating expenses research and development ( in millions,
except percentages ) 2022 2021 percentage change research and
development $ 24, 512 $ 20, 716 18 % as a percent of revenue 12 % 12 %
0ppt research and development expenses include payroll, employee
benefits, stock - based compensation expense, and other headcount -
related expenses associated with product development. research and
development expenses also include third - party development and
programming costs, localization costs incurred to translate software
for international markets, and the amortization of purchased software
code and services content. research and development expenses increased
$ 3. 8 billion or 18 % driven by investments in cloud engineering,
gaming, and linkedin. sales and marketing

competitive in local markets and enables us to continue to attract top
talent from across the world. we plan to continue to make significant
investments in a broad range of product research and development
activities, and as appropriate we will coordinate our research and
development across operating segments and leverage the results across
the company. in addition to our main research and development
operations, we also operate microsoft research. microsoft research is
one of the world ’ s largest corporate research organizations and works
in close collaboration with top universities around the world to
advance the state - of - the - art in computer science and a broad
range of other disciplines, providing us a unique perspective on future
trends and contributing to our innovation.

our success is based on our ability to create new and compelling
products, services, and experiences for our users, to initiate and
embrace disruptive technology trends, to enter new geographic and
product markets, and to drive broad adoption of our products and
services. we invest in a range of emerging technology trends and
breakthroughs that we believe offer significant opportunities to
deliver value to our customers and growth for the company. based on our
assessment of key technology trends, we maintain our long - term
commitment to research and development across a wide spectrum of
technologies, tools, and platforms spanning digital work and life
experiences, cloud computing, ai, devices, and operating systems. while
our main product research and development facilities are located in
redmond, washington, we also operate research and development
facilities in other parts of the u. s. and around the world. this
global approach helps us remain

when the world around us does well. that ’ s what i believe will lead
to widespread human progress and ultimately improve the lives of
everyone. there is no more powerful input than digital technology to
drive the world ’ s economic output. this is the core thesis for our
being as a company, but it ’ s not enough. as we drive global economic
growth, we must also commit to creating a more inclusive, equitable,
sustainable, and trusted future. support inclusive economic growth we
must ensure the growth we drive reaches every person, organization,
community, and country. this starts with increasing access to digital
skills. this year alone, more than 23 million people accessed digital
skills training as part of our global skills initiative.

also increased the number of identified partners in the black partner
growth initiative and continue to invest in the partner community
through the black channel partner alliance by supporting events focused
on business growth, accelerators, and mentorship. progress does not
undo the egregious injustices of the past or diminish those who
continue to live with inequity. we are committed to leveraging our
resources to help accelerate diversity and inclusion across our
ecosystem and to hold ourselves accountable to accelerate change – for
microsoft, and beyond. investing in digital skills the covid - 19
pandemic led to record unemployment, disrupting livelihoods of people
around the world. after helping over 30 million people in 249 countries
and territories with our global skills initiative, we introduced a new
initiative to support a more skills - based labor market, with greater
flexibility and accessible learning paths to develop the right skills

at times, we make select intellectual property broadly available at no
or low cost to achieve a strategic objective, such as promoting
industry standards, advancing interoperability, supporting societal and
/ or environmental efforts, or attracting and enabling our external
development community. our increasing engagement with open source
software will also cause us to license our intellectual property rights
broadly in certain situations. while it may be necessary in the future
to seek or renew licenses relating to various aspects of our products,
services, and business methods, we believe, based upon past experience
and industry practice, such licenses generally can be obtained on
commercially reasonable terms. we believe our continuing research and
product development are not materially dependent on any single license
or other agreement with a third party relating to the development of
our products. investing in the future

but generally include parts and labor over a period generally ranging
from 90 days to three years. for software warranties, we estimate the
costs to provide bug fixes, such as security patches, over the
estimated life of the software. we regularly reevaluate our estimates
to assess the adequacy of the recorded warranty liabilities and adjust
the amounts as necessary. research and development research and
development expenses include payroll, employee benefits, stock - based
compensation expense, and other headcount - related expenses associated
with product development. research and development expenses also
include third - party development and programming costs, localization
costs incurred to translate software for international markets, and the
amortization of purchased software code and services content. such
costs related to software development are included in research and
development expense until the point that technological feasibility is
reached, which for our

fiscal year 2021 was a year of both successes and challenges. while we
continued to make progress on several of our goals, with an overall
reduction in our combined scope 1 and scope 2 emissions, our scope 3
emissions increased, due in substantial part to significant global
datacenter expansions and growth in xbox sales and usage as a result of
the covid - 19 pandemic. despite these scope 3 increases, we will
continue to build the foundations and do the work to deliver on our
commitments, and help our customers and partners achieve theirs. we
have learned the impact of our work will not all be felt immediately,
and our experience highlights how progress won ’ t always be linear.
while fiscal year 2021 presented us with some new learnings, we also
made some great progress. a few examples that illuminate the diversity
of our work include : • we purchased the removal of 1. 4 million
metrics tons of carbon.

we protect our intellectual property investments in a variety of ways.
we work actively in the u. s. and internationally to ensure the
enforcement of copyright, trademark, trade secret, and other
protections that apply to our software and hardware products, services,
business plans, and branding. we are a leader among technology
companies in pursuing patents and currently have a portfolio of over
69, 000 u. s. and international patents issued and over 19, 000 pending
worldwide. while we employ much of our internally - developed
intellectual property exclusively in our products and services, we also
engage in outbound licensing of specific patented technologies that are
incorporated into licensees ’ products. from time to time, we enter
into broader cross - license agreements with other technology companies
covering entire groups of patents. we may also purchase or license
technology that we incorporate into our products and services.

15 corporate social responsibility commitment to sustainability we work
to ensure that technology is inclusive, trusted, and increases
sustainability. we are accelerating progress toward a more sustainable
future by reducing our environmental footprint, advancing research,
helping our customers build sustainable solutions, and advocating for
policies that benefit the environment. in january 2020, we announced a
bold commitment and detailed plan to be carbon negative by 2030, and to
remove from the environment by 2050 all the carbon we have emitted
since our founding in 1975. this included a commitment to invest $ 1
billion over four years in new technologies and innovative climate
solutions. we built on this pledge by adding commitments to be water
positive by 2030, zero waste by 2030, and to protect ecosystems by
developing a planetary computer. we also help our suppliers and
customers around the world use microsoft technology to reduce their own
carbon footprint.

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)
print("Scores:")
for score in scores:
    print(score)
Scores:
0.98693466
2.644579
-0.26802942
-10.73159
-7.7066045
-5.6469955
-4.297035
-10.933233
-7.0384283
-7.3246956
print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o+1)
New Ordering:
2
1
3
7
6
9
10
5
4
8
Re-ranking with Query Expansion
original_query = "What were the most important factors that contributed to increases in revenue?"
generated_queries = [
    "What were the major drivers of revenue growth?",
    "Were there any new product launches that contributed to the increase in revenue?",
    "Did any changes in pricing or promotions impact the revenue growth?",
    "What were the key market trends that facilitated the increase in revenue?",
    "Did any acquisitions or partnerships contribute to the revenue growth?"
]
queries = [original_query] + generated_queries
​
results = chroma_collection.query(query_texts=queries, n_results=10, include=['documents', 'embeddings'])
retrieved_documents = results['documents']
# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)
​
unique_documents = list(unique_documents)
pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])
scores = cross_encoder.predict(pairs)
​
print("Scores:")
for score in scores:
    print(score)
Scores:
-10.042844
-8.505108
-10.00014
-4.651892
-9.80788
-7.754099
-9.918428
-4.8184834
-7.9171767
-9.768024
-1.136996
-10.0839405
-9.357721
-11.0792675
-6.90209
-10.148884
-4.3417687
-5.141832
-3.7948632
-7.4906564
-5.274752
-3.7681527
-10.711212
print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)
New Ordering:
10
21
18
16
3
7
17
20
14
19
5
8
1
12
9
4
6
2
0
11
15
22
13
