# Databricks notebook source
"""
How Business Thinkers Can Start Building AI Plugins With Semantic Kernel


As LLMs continue to develop, finding ways to efficiently 
incorporate their capabilities into applications for many users 
has become increasingly urgent. 
I'm delighted today to introduce a course with one of the leading global 
application developers, Microsoft, 
which has been developing an SDK called Semantic 
Kernel that can enable you to rapidly use 
LLMs in your applications. 
Your instructor for this course is John Maeda, 
Vice President for Design and AI at Microsoft. 
John, welcome. 
Thank you, Andrew. 
So glad to be here. 
John's design work in tech and AI has touched so many lives. 
For example, his early work at MIT led to his co-creating 
Stretch, which is a wonderful programming 
tool for children. 
Later, John earned an MBA and shifted to the 
business and venture capital world, which is where we 
first crossed paths when John was with one of Coursera's 
investors. 
So, in this course, I understand you've 
married two of your favorite topics, teaching programming and 
accelerating businesses. 
So, this led to this course on LMs using Semantic Kernel. 
Well, I know that sounds a bit like a stretch, but I've 
been fascinated by the enabling power of computation. 
Computer science is something abstract, but as a kid, 
I touched a Commodore PET, PET. 
as a kid growing up in Chinatown of Seattle in a family-run 
tofu business where parents did just one dream for us, go 
to college and get out of Chinatown. 
That encouragement took me to MIT in my adulthood and down a 
conventional computer science path, 
but much later in life I discovered that it wasn't 
the technology that interests me the most, it 
was instead what you could do with the technology. 
And oftentimes it required exposing the 
technology to non technologists, whether children, designers, 
or artists, or even business people to find out how best to really use 
it. 
So I have to say Andrew, I was really inspired by 
your TED talk on AI. 
It really struck a chord because you described 
wanting to bring AI to a business of a pizza shop owner. 
That's a great vision and it's the kind of things I would have wanted to 
do for my parents. 
They're super busy. 
If AI could have helped them in their business because they've 
had a better life. 
I love that thought. 
So we're going to show how that pizza shop owner can use 
LLMs in their business. 
Yeah, thank you, John. 
I think the work that you and Microsoft are doing to make 
AI accessible to everyone is really fantastic. 
Can you say more about what you cover in this course? 
Absolutely. 
Well, you know, when you think about Symantec Kernel, 
it's kind of a fancy name, but just remember it's an 
open source toolkit. 
It's the brainchild of Microsoft's deputy CTO. 
Sam Scalace. 
Sam is the inventor of something called Google Docs, 
you may have heard of. 
There's two underlying concepts to this new wave of AI that 
I think matter the most. 
I like to tell people to stick their 
hands out in front of them. 
It's hard when you're on a remote, but stick your hands out in 
front of you, take your right hand, shake it, and call 
this the completion engine. 
Whoa, the completion engine. 
It can finish my sentence. 
And then take out your left hand, shake it a bit. 
The left hand is a similarity engine. 
It's like a magnet, it can sort of pull things out and find 
things that are similar that are something you 
wouldn't expect a computer program to find 
as similar at all. 
So the ability to compare meaning is very new, 
at least it feels to me, to some of you as well. 
And then this completion engine is the kind of wow. 
And so we sort of focus on the completion engine. 
oh my god it's so amazing, but the similarity and you 
need some love. 
So it's this combination of these two that are really 
making this wave feel different at least when 
I talk to businesses 
of all sizes and I think for small business as well this 
combination left hand right hand working together it's 
pretty cool. 
We see this in places like retrieval augmented generation so-called 
RAG the ability to connect context 
with completion but also if you don't have context and you 
try to complete something, 
you end up with hallucinations, because the completion 
engine is running on an empty stomach. 
So these two together, a combination of two hands, 
is pretty amazing. 
And in this course, we're going to try to make sure you don't stay 
sort of playing with your fingers, but actually 
coding with Symantec Kernel. 
You're going to use two AI plugins. 
Number one, a design thinking plugin, and number two, 
a business thinking plugin. 
So in this course, we're going to combine traditional 
business thinking with the power of LLMs to 
help make Andrew's pizza shop owner's business life 
a little better. 
In the process, you're going to learn about 
open source semantic kernel. 
You're going to start by building some semantic functions 
to summarize some text and do the regular 
things like chaining. 
Then we'll jump into design thinking. 
You're going to take customer feedback and stick it 
into the plugin and get it to do things and generate the 
magic of AI. 
And then we work from top down, 
we're going to apply SWOT analysis to the pizza 
shop using the business thinking AI plugin to 
find ways to improve cost and time efficiency. 
So moving on, we're going to play with vector memory, of 
course. 
Everyone loves that. 
And use our planner module to finish the 
meal to get you thinking about the future 
of this AI revolution. 
Boy, that's a very exciting set of topics. 
In addition to John teaching this, we'd also like to acknowledge 
the many people who've helped make this course possible. 
On the Microsoft side, we'd like to thank Sam Scalace, Umesh 
Madan, Devish Lucato, Evan Sharkey, Tim Laverty, 
Harleen Tind, Abby Harrison, Sean Caligari, Matthew Bolanas, 
and the entire Semantic Kernel community and team. 
On the Deep Learning.ai side, Jeff Ludwig, and Diala Ezzedine. 
So John, this sounds like a really exciting set of topics. 
Let's get started. 
"""
