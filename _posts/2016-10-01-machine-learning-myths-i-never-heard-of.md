---
layout: post
title: "9 Machine Learning Myths I've Never Heard"
date: 2016-09-27 00:00
tags: general
---
O'Reilly has a business model and part of that business model involves hype although they probably would call it "marketing". Recently, I've noticed that this "marketing" has promoted Machine Learning to Artificial Intelligence and Artificial Intelligence to Superintelligence. Everything has moved up a peg. It sells books. It sells conferences. It sells the infrastructure and frameworks of vendors at conferences.

Hype notwithstanding, I receive both their Data Science and Artificial Intelligence newsletters because they do sometimes point me to ideas and concepts I haven't been exposed to in my own corner of the Internet. And thus it came to pass that I read Professor Domingos' blog post on Medium, [Ten Myths About Machine Learning](https://medium.com/@pedromdd/ten-myths-about-machine-learning-d888b48334a3), and then promptly did a `#facepalm`.

The entire post was just...just...`#heavysigh`.

I'm going to preface this by saying that I have met Professor Domingos and have read some of his work...but that doesn't mean this was a good blog post. In this post, Professor Domingos addresses a number of myths about machine learning. As I read through these myths and the rebuttals, two things struck me almost every time: where does this myth come from and why is that even a rebuttal of the myth?

**Myth 1** Machine Learning is just summarizing data.

*Rebuttal* - "In reality, the main purpose of machine learning is to predict the future."

I have never heard anyone ever say "machine learning is just summarizing data" except that, at a certain level, it's absolutely true. Machine learning builds models of data. And if anyone is familiar with information theory, statistical pattern recognition, computational learning theory, these models *are* "summaries" of the data.

If he means that "machine learning is just calculating statistics like counts, means and percentages" then, well, no except that a mean and standard deviation can be used to fit a simple univariate model. So, yes, even simple aggregates--summaries--of the data are models and thus machine learning.

Now, let's turn to the rebuttal which holds that the main purpose of machine learning is to predict the future. There are several things necessary for this statement to be true. First, you have to limit yourself to *supervised* machine learning and ignore unsupervised machine learning altogether. This *might* also hold true for the third type of machine learning, reinforcement learning.

Even if we limit ourselves to *supervised* machine learning, which involves using data that has a feature we want to model, (and we will from now on because it appears that Professor Domingos did as well), such machine learning is not solely concerned about the future...it's concerned about the unknown. When you have a collection of images and wish to classify them as cats or dogs, that is not a prediction about the *future*. When you have email, and you wish to classify it as spam or "ham", that is not a prediction about the *future*.

Sometimes the unknown is the future. Generally speaking, when most people talk about predicting the future, whether it's GDP, stock market prices or sales...they are talking about *forecasting* and machine learning (broadly including all statistical models) has failed *miserably*. The most generous interpretation I can give of this statement is that it predicts a value or label that might be given or known in the future...but that's sophistry.

**Myth 2** Learning algorithms just discover correlations between pairs of events.

*Rebuttal* "They discover richer forms of knowledge."

If we concentrate on "pairs of events", I have to scratch my head. This one definitely falls into the "who believes that?" bucket. Really? There are people who believe that machine learning can only predict *pairs* of events?

Yes, machine learning algorithms do discover richer forms of knowledge. The interesting thing is that if we concentrate on the "just discover correlations" part of the "myth" the example used as a rebuttal makes no sense. He proffers "if a mole has [an] irregular shape and color and is growing, then it may be skin cancer" as an example of "richer forms of knowledge"...and in fact it is. But it's still just a correlation. One would be hard pressed to think of a world where the correlation was false and the rule was true.

**Myth 3** Machine learning can only discover correlations, not causal relationships.

*Rebuttal* - "In fact, one of the most popular types of machine learning consists of trying out different actions and observing their consequences — the essence of causal discovery."

I have to admit that he completely lost me on this one. It would be nice if he was more forthcoming about these mysterious algorithms he is referring to but I can make some educated guesses since, well, I've worked in this sandbox for a while.

First, wading into the philosophical debate on what causation actually is is a "strong move" as they used to say at LivingSocial. But I have literally no idea what algorithm he's talking about. If he's talking about the many thousands of A/B tests that LivingSocial, Facebook, Google, etc. are running on your behavior right now, I can assure you that the actual decisions are not made using machine learning.

"For example, an e-commerce site can try many different ways of presenting a product and choose the one that leads to the most purchases." While I agree that each different way of presenting the product might be the result of a machine learning algorithm, picking between them is not...unless all of a sudden Professor Domingo is counting tests of statistical significance as machine learning. They're made using statistics, developed and used long before the Internet Age. The statement only makes sense if Professor Domingos is now including hypothesis testing as machine learning. If so, wow.

Now, there *is* a machine learning algorithm that could do this: multi-armed bandit (MAB) optimization but I know of very few companies that have implemented it. It is a form of reinforcement learning. In this context, I'd be very careful about asserting that even MAB learns anything except correlation.

**Myth 4** Machine learning can’t predict previously unseen events, a.k.a. “black swans.”

*Rebuttal* - Professor Domingos rebuts this notion with the example of spam filters. Spam filters predict spam that they haven't seen before. He also makes some spurious claims about the housing bubble and resulting financial crisis.

Following *The Princess Bride*, I'm starting to think that when Professor Domingos says "machine learning", I don't think word means what he thinks it means. Or perhaps it's Humpty Dumpty in *Alice in Wonderland*, "When I use a word, it means just what I chose it to mean."

Machine learning works by applying algorithms to data that build a model (a representation) of the pattern they're going to predict. Again, this pattern need not be in the future *per se* but merely unknown..."what's wrong with this car?". Spam filters work because they have a representation of the patterns for "spam" and thus anything that matches that representation will be identified as spam even if the model hasn't seen that *specific* spam. However, if the features of spam change, the model *will* fail. Here are some "black swans" that spam filters have failed to predict:

* purposeful misspellings
* the use of l337 sp3@k
* the inclusion of random paragraphs of normal text.
* the transformation of spam into an embedded image.

In all such cases, new models of spam had to be built.

**Myth 5** The more data you have, the more likely you are to hallucinate patterns.

*Rebuttal* - I cannot easily summarize his rebuttal because he's all over the place on this one.

First, although I have not seen the myth in this form, I have seen similar myths. It normally occurs in Data Science. So this myth does actually circulate although I've never seen it applied to machine learning. We have to address what we mean by "more data".

Do we mean...for a given number of inputs, we have more *examples*? Do we mean that for a given number of examples, we have more *inputs*? Or both?

Machine learning is fundamentally inference from training data to the world. We can easily envision a situation where an initial training set concentrating on 10 features over 100 known terrorists, when expanded to the population at large (at NSA scale) might see an increase in false positives.

For a given model and a given set of inputs (features), there is a sweet spot in terms of the amount of data you can reasonably use and improve accuracy. This is the *bias/variance tradeoff*. But this does not handle the problem of whether your data was representative of the target problem to start with. So, yeah, I can easily believe that there are models made on unrepresentative data sets that don't scale.

Professor Domingos' example of connecting information is well taken. This is the case of "more data" meaning "more inputs" but if you add more inputs, you almost always need more examples of that data. This is easy to see if you have 100 examples with 10 features, then 100 features, then 1000 features. You really need more than 100 examples by the time you get to 1000 features.

And this leads to real problems: spurious correlations. The more feature you have, the more data you have, the more chance correlations you'll have. It's not about machine learning, it's about basic statistics...and it's not a myth.

**Myth 6** Machine learning ignores preexisting knowledge.

*Rebuttal* - Some do and some don't.

It's difficult to ascertain exactly what Professor Domingos is talking about here (again, a reference or three to actual algorithms would help). I'm familiar with the claim and it is often true but not always. I'm not sure that makes it a myth as much as an over generalization.

**Myth 7** The models computers learn are incomprehensible to humans.

*Rebuttal* Some are and some aren't.

Although I don't know anyone who believes the myth, the rebuttal is true. This is more an overgeneralization rather than a myth.

I would only add that all the "hot" algorithms these days: random forests, deep learning, etc. are *not* comprehensible. And at a certain scale, even recommender systems can be mysterious especially those based on collaborative filtering: they're just a form of missing value imputation.

**Myth 8** Simpler models are more accurate.

*Rebuttal* ???

First, the myth isn't really a myth, he just left off part of the entire claim. The full claim is that, a simpler model *with a lot of data* will often beat more complicated models with less data. This seems to be a jab at Peter Norvig's [The Unreasonable Effectiveness of Data](https://www.youtube.com/watch?v=yvDCzhbjYWs). Norvig is Director of Research at Google and used to be the Director of Search Quality. Norvig was making a claim about his experiences at Google using simple models with Google scale data. It's entirely possible that for the claim to be true, you need Google scale data. Domingos simply says, more complicated algorithms are worth it because the seemingly high overhead allows them to beat simpler models. But Professor Domingos seems only concerned with accuracy and there doesn't really seem to be anything in particular to back up his rebuttal.

Additionally, there are a whole host of other factors in the actual deployment of machine learning applications in real situations that Professor Domingos neglects. Machine Learning is not applied in business for its own sake. For many businesses, especially those just starting out, a simple algorithm (logistic regression, Naive Bayes) with 80% accuracy might be monetarily justifiable where the infrastructure for something more complicated might not be even if the accuracy is higher. In *applied* Machine Learning, there are trade-offs other than just accuracy. Starting out with simpler models can often make good business sense given a whole host of other factors: design, integration and debugging, data infrastructure, explanation to non-technical people and stakeholder buy-in.

**Myth 9** The patterns computers discover can be taken at face value.

*Rebuttal* Machine learning models are for prediction; only after certain assurances are made can we interpret them as *explanation*.

I think that Professor Domingos really wants to say "correlation is not causation" here but he's already ruled that out above because he says that models can find causation. If they can find causation, shouldn't they be taken at face value? This seems to be a general pattern throughout the blog post. If "some do, some don't" rebuts the myth then that's an acceptable answer. On the other hand, if "some do, some don't" would validate the myth or part of it, then it has to be *all* of machine learning.

**Myth 10** Machine learning will soon give rise to superhuman intelligence.

**Rebuttal** "No."

I have to agree with him on that. It's a ways off.

---

Overall, the blog post reminds me of a line that was repeated often in *The Giver*, a dystopian tale of the future where emotion was suppressed. Whenever someone said something smacking of emotion like "I love you", the canned response was "precision of language!" Like, you can't just refer to vague generalities. I feel like this is what Professor Domingos did in this post. It would be nice to go over some of the real problems/myths/hype that people hold that stand in the way of effectively bringing machine learning and data science to organizations of all sizes.

Cheers.
