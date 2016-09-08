---
layout: post
title: "Quick and Dirty NLP in Python"
date: 2016-09-06 00:00
tags: nlp, python, machine learning
---

Sometime ago I was looking for a quick and dirty example of the Bag of Words approach to NLP, preferably in Python. I guess my Google-Fu wasn't up to the task because I couldn't find one. I found the excellent `nltk` library but most of the examples are geared towards higher level computational linguistics. A Bag of Words model lets us use machine learning models we have lying around like kNN or k-means clustering rather than n-grams or max entropy models.

And so this example was born; I hope it's useful.

Basically I'm going to cover:

* Structured v. Unstructured Data.
* Bag of Words Model
* Counts, TF and TF-IDF
* Distance and Similarity Metrics
* Machine Learning with a Bag of Words
* Working with Raw Text
* Other Uses for Text

And although the "Dirty" part is mostly true, the "Quick" part isn't, except in a relative sense. There's nothing quick about NLP.

## Structured v. Unstructured Data

Some among the pendants will be quick to point out that prose is structured. Why yes, yes it is. But it's not a rectangular array of data--an table or Excel spreadsheet--and in this context that is what I mean by "structured." Almost all of our supervised and unsupervised learning algorithms expect such data and so the chief problem in applying those methods to prose is to get it into "the right shape". We want to turn the Complete Works of Shakespeare--or yesterday's twitter feed--into rows representing *documents*, columns representing *terms* and cells representing a number of some kind (we'll get more specific shortly).

It turns out that the most popular approach to solving this problem is to use the *Bag of Words* model of a document.

## Bag of Words

The basic concept involves doing something really simple but counter-intuitive. We understand that this sentence I am typing and you are reading has verbs, nouns and that the order is not arbitrary. There *are* NLP models that take advantage of that, identifying the nouns, verbs, the subordinate clauses, etc. We're going to toss everything aside you learned in Mr. Smith's English class and simply treat a document as a bag of words. We will completely ignore their order as if you shook the document really hard into a paper bag and all the words came off the page. And now you just start picking them up, one by one.

And do what?

## Existence, Counts, TF and TF-IDF

I did promise that this was going to be a Quick and Dirty guide...but "quick" and "dirty" are relative. The truth of the matter is that it's all grey and you start with the dark side of the spectrum and work your way towards the light, making trade offs along the way: implementation and maintenance complexity, competing priorities, incremental improvement.

One approach is to simply note, true or false, if each possible term is in the bag. Such an approach might involve scaning the dictionary and making note of each word, is it in the bag or not. We don't care how many times. Of course, it's easier--and more efficient--to simply count the unique terms in the bag and then assign "true" to each one. Anything that isn't true is assumed to be false. We'll be dealing with such *sparse* representations a lot. In Python, we can take the List of words, construct a Set from it and we have a Set of terms.

We will be using a very simplified set of documents for these exercises so we can get a feel for what is going on rather than being overwhelmed by details.

```python
bag_of_words = ["wet", "cold", "wet", "green", "leprechaun", "shamrock", "shamrock"]
terms = set( bag_of_words)
print terms
```

yields

```
set(['shamrock', 'leprechaun', 'cold', 'green', 'wet'])
```

The next level is to count the number of times each term appears. "is" might occur 20 times whereas "run" might occur 2 times. We just count them. It's easy to do in Python with a Counter.

```python
from collections import Counter

bag_of_words = ["wet", "cold", "wet", "green", "leprechaun", "shamrock", "shamrock"]
term_counts = Counter( bag_of_words)
print term_counts
```

yields

```
Counter({'shamrock': 2, 'wet': 2, 'leprechaun': 1, 'cold': 1, 'green': 1})
```

If you're working with Tweets, counting might be ok because the absolute number of words is fairly limited. However, if you're looking at something like emails, some are short and some are long and "is" will certainly appear fewer times in the shorter emails than the longer ones. Here we introduce the idea of (relative) *frequency*. You simply divide each count by the number of words in the document. We can do this by using a Counter then a Dictionary Comprehension.

```python
from collections import Counter

bag_of_words = ["wet", "cold", "wet", "green", "leprechaun", "shamrock", "shamrock"]
number_of_words = float( len( bag_of_words))
term_counts = Counter( bag_of_words)
term_frequencies = {t: c/number_of_words for (t, c) in term_counts.items()}
print term_frequencies
```

yields

```
{'shamrock': 0.2857142857142857, 'leprechaun': 0.14285714285714285, 'cold': 0.14285714285714285, 'green': 0.14285714285714285, 'wet': 0.2857142857142857}
```
This is technically known as *Term Frequency* or *TF*.

We will talk about "common" words a bit later ("a", "the", "and") but in a more general context, when you have a large collection of documents, some words will occur very frequently across all documents and so they don't help distinguish between different types of documents. For example, suppose you have a set of documents dealing with finance and fly fishing. "bank" will occur in both documents and so "bank" might not be very good at deciding if two documents are similar or if one is about finance or another is about fishing. In order to combat this problem, we can introduce *Inverse Document Frequency*

$idf(t, D) = log\frac{|D|}{|{d \in D: t \in d}|}$

Let's break this up into bits. The first function takes a list of words and returns a Dict of term frequencies:

```python
from __future__ import division
from math import log

def tf( d):
  number_of_words = len( d)
  term_counts = Counter( d)
  term_frequencies = {t: c/number_of_words for (t, c) in term_counts.items()}
  return term_frequencies
```

The next function takes a list of term frequency Dicts and returns a Dict of inverse document frequencies:

```python
def idf( tfs):
  terms = set([i for ls in map( lambda x: x.keys(), tfs) for i in ls])
  n = len( tfs)
  counts = {}
  for t in terms:
    count = 0
    for d in tfs:
      if t in d:
        count = count +1
    counts[ t] = count
  inverse_document_frequencies = {t: log(n/counts[t]) for t in terms}
  return inverse_document_frequencies
```
And now we can apply them to several documents about Ireland and Seattle:

```python
ds = [
   ["wet", "cold", "wet", "green", "leprechaun", "shamrock", "shamrock", "wet"],
   ["wet", "cold", "wet", "rocky", "hipster", "ipa", "evergreen"],
   ["wet", "cold", "wet", "cold", "guinness", "rocky", "leprechaun", "wool"],
   ["wet", "cold", "wet", "cold", "hipster", "rocky", "pacific", "salmon"],
   ["wet", "cold", "wet", "cold", "hipster", "rocky", "cold", "wool"]
]

tfs = [tf( d) for d in ds]
print tfs
idfs = idf( tfs)
print idfs
```
If we dig through the term frequencies:

```
[{'shamrock': 0.25, 'leprechaun': 0.125, 'cold': 0.125, 'green': 0.125, 'wet': 0.375},
 {'evergreen': 0.14285714285714285, 'ipa': 0.14285714285714285, 'rocky': 0.14285714285714285, 'hipster': 0.14285714285714285, 'wet': 0.2857142857142857, 'cold': 0.14285714285714285},
 {'rocky': 0.125, 'guinness': 0.125, 'wet': 0.25, 'leprechaun': 0.125, 'cold': 0.25, 'wool': 0.125},
 {'rocky': 0.125, 'salmon': 0.125, 'pacific': 0.125, 'hipster': 0.125, 'wet': 0.25, 'cold': 0.25},
 {'cold': 0.375, 'wool': 0.125, 'hipster': 0.125, 'rocky': 0.125, 'wet': 0.25}]
```

there isn't anything particularly new here. We can see that different words have different relative frequencies within their documents. However, when we look at IDF:

```
{'evergreen': 1.6094379124341003, 'ipa': 1.6094379124341003, 'rocky': 0.22314355131420976, 'guinness': 1.6094379124341003, 'leprechaun': 0.9162907318741551, 'salmon': 1.6094379124341003, 'pacific': 1.6094379124341003, 'shamrock': 1.6094379124341003, 'hipster': 0.5108256237659907, 'wet': 0.0, 'green': 1.6094379124341003, 'cold': 0.0, 'wool': 0.9162907318741551}
```

`wet` and `cold` appear in every document and so they end up with $idf=0$. When we multiply $tf \times idf$, they won't be counted. On the other hand, terms like `pacific`, `ipa`, `wool` and `salmon` appear in only one document each so their idf values are 1.61. They will up-weight their respective TFs. `hipster` appears relatively frequently across documents so it has a score of 0.51.

Create new tf-idf Dicts:

```python
tf_idfs  = [{t: tf * idfs[ t] for (t,tf) in d.items()} for d in tfs]
print tf_idfs
```

which yields:

```
[{'shamrock': 0.40235947810852507, 'leprechaun': 0.11453634148426939, 'cold': 0.0, 'green': 0.20117973905426254, 'wet': 0.0},
{'evergreen': 0.22991970177630003, 'ipa': 0.22991970177630003, 'rocky': 0.03187765018774425, 'hipster': 0.07297508910942724, 'wet': 0.0, 'cold': 0.0},
{'rocky': 0.02789294391427622, 'guinness': 0.20117973905426254, 'wet': 0.0, 'leprechaun': 0.11453634148426939, 'cold': 0.0, 'wool': 0.11453634148426939},
{'rocky': 0.02789294391427622, 'salmon': 0.20117973905426254, 'pacific': 0.20117973905426254, 'hipster': 0.06385320297074884, 'wet': 0.0, 'cold': 0.0},
{'cold': 0.0, 'wool': 0.11453634148426939, 'hipster': 0.06385320297074884, 'rocky': 0.02789294391427622, 'wet': 0.0}]
```

by comparing to the original term frequencies, you can see which terms were upweighted and which terms were downweighted. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is not uncontroversial from a theoretical point of view.

As I say, this is a continuum. You can certainly start out with the simplest thing and work your way forward. Why not just start with everything?

1. Simpler is always easier to debug, explain and remember, 5 months later, when you go digging through your own code or if you go off to found that start-up and the poor schmucks left behind have to read your code.
2. Every new transformation is a new opportunity for a coding error, a bad interaction with a different transformation, or another concept to explain to others.
3. You may be implementing this *de novo* in a production application that uses Ruby or Java that might not have all the libraries you want or there may be other restrictions (DevOps doesn't want to support R or Python in production).
4. Just because it sounds neat, it may not work/help.

## Distance and Similarity Metrics

Now we have a sparse vector of booleans, counts, term frequencies or term frequencies adjusted for document frequency, one for each document, that is, we finally have our "structured" data. What can we do with it?

In general, we can either try to do some kind of unsupervised learning or **clustering** with them or some kind of supervised learning or **classification** with them. We only need to be able to relate documents in this new representation to each other. We generally use some kind of *distance* or *similarity* measure to relate documents.

### Similarity

Starting with the simplest representation (documents represented by sets of terms), the simplest similarity measure is the **Jaccard Index**. The Jaccard Index is defined as the count of those terms the documents have in common divided by their combined vocabulary:

$Jaccard = \frac{A\cap B}{A \cup B}$

Python set operations have everything we need:

```python
d1 = set(tf_idfs[0].keys())
d2 = set(tf_idfs[1].keys())
print d1
print d2
intersection = d1.intersection( d2)
print intersection
union = d1.union( d2)
print union
jaccard = len( intersection)/len( union)
print jaccard
```
which will give us:

```
set(['shamrock', 'leprechaun', 'cold', 'green', 'wet'])
set(['evergreen', 'ipa', 'rocky', 'hipster', 'wet', 'cold'])
set(['cold', 'wet'])
set(['evergreen', 'ipa', 'rocky', 'shamrock', 'hipster', 'green', 'wet', 'leprechaun', 'cold'])
0.222222222222
```

Because the numerator is bounded by the denominator, we can say that document 1 is 22% similar to document 2, our goal would be to find the document that document 1 is most similar to:

```python
def jaccard(a, b):
  d1 = set(a.keys())
  d2 = set(b.keys())
  intersection = d1.intersection( d2)
  union = d1.union( d2)
  jaccard = len( intersection)/len( union)
  return jaccard
```

with

```
>>> jaccard( tf_idfs[0], tf_idfs[ 1])
0.2222222222222222
```

If we take document 1 and loop over the others, the most similar document is:

```python
for i, d in enumerate( tf_idfs[1:]):
  print i + 2, jaccard( tf_idfs[ 0], d)
```

```
>>> for i, d in enumerate( tf_idfs[1:]):
...   print i + 2, jaccard( tf_idfs[ 0], d)
...
2 0.222222222222
3 0.375
4 0.222222222222
5 0.25
```

Document 3 or

```
   ["wet", "cold", "wet", "cold", "guinness", "rocky", "leprechaun", "wool"],
```

That makes sense. Whew.

There are other similarity measures as well. The most popular is Cosine Similarity. Cosine Similarity is defined as:

$cos(a, b) = \frac{A}{||A||}\frac{B}{||B||}$

which is to say, it is the dot product of the normalized (unit vectors) of A and B. Normalization is straight-forward, we just need the length of the vector:

```python
from math import sqrt

def length( tf):
  return sqrt(sum([v**2 for v in tf.values()]))
```
which is the following for document 1:

```
>>> length( tf_idfs[0])
0.4642036304794556
```

If we divide the values through by the length, viola, normalized. I'm going to use the tfs instead of tf_idfs for this part of the exercise:

```python
unit_tfs = []
for tf in tfs:
  normalizer = length( tf)
  unit_tfs.append({t: v/normalizer for t, v in tf.items()})
```

Applying length again to a document, should yield 1:

```
>>> length( unit_tfs[ 0])
1.0
```

We need one last thing, a `dotproduct` function that understands sparse vectors:

```python
def dotproduct( a, b):
  result = 0.0
  for t in a.keys():
    if t in b:
      result = result + a[ t] * b[ t]
  return result
```

You should convince yourself that it doesn't matter if we loop over the terms in a or b above.

Now we can do our loop again, looking for documents similar to document 1:

```
>>> for i, d in enumerate( unit_tfs[1:]):
...     print i + 2, dotproduct( unit_tfs[ 0], d)
...
2 0.583333333333
3 0.649519052838
4 0.57735026919
5 0.5625
```

Again, Document 1 is most like Document 3.

There are other measures we can use as well:

1. [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
2. [Pearson's Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)
3. [Spearman's Rank Order Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

I have even used a "Fuzzy" Jaccard Index based on the Fuzzy AND (intersection) and OR (union) with TF as the degree of membership.

Because they measure the relatedness of documents in different ways, they all can lead to different results in real world use cases on real world documents. You will generally want to try all of them. The main obstacle is that most existing implementations (for example in `numpy`) do not handle sparse vectors so you will need to code your own versions. This isn't too difficult if you've gotten this far.

## Machine Learning with a Bag of Words

So our goal was to get unstructured data--from a Machine Learning point of view--into a structured format and now we've succeeded. Why kinds of Machine Learning can we do?

I will just say that I generally start with TF as the representation and cosine similarity as the relatedness measure and work from there, usually experimenting with relatedness measures and data preparation, which we have yet to discuss.

### Querying

We've already seen one example of with a more information retrieval bent. If I have a document A, I can use it as a query into a library of documents and find all the documents that are "like" it. This involves:

1. A precomputed representation of the library (usually called a *corpus*) as either Boolean, Counts, TF or TF-IDF. Generally, I suggest always starting with TF.
2. If Document A, the query, is a document from the corpus, then we only need to calculate the "relatedness" measure between Document A and every other document in the corpus. Not necessarily the most efficient thing. We could precompute this and depending on how often the corpus changes, we always have it.
3. If Document A, the query, is not a document from the corpus then we must transform it into whatever representation we have chosen (Boolean, Counts, TF, TF-IDF) and calculated our relatedness measure against the corpus, returning the top N documents.

### Clustering
Clustering involves the automatic grouping of items (documents, observations). It is an unsupervised Machine Learning algorithm. The most well known clustering algorithm is [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering). Essentially, you start out with a guess about the number of clusters your data has, k. The algorithm then simultaneously attempts to assign documents to the k clusters and determine what the k clusters are. This is an example of [Expectation Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm). If you require the cluster center to be an actual document rather than the "average" document, you can constrain the cluster centers to be documents, in which case you are doing k-medoids clustering.

The best measure of the effectiveness of a clustering is how successful it is application or in other machine learning algorithms. Since it is unsupervised, it is unlikely to recover "actual" classes unless you are very lucky. Still, clustering can be a useful tool. You may need to experiment with different representations, relatedness measures and raw data processing (below).

### Classification
Classification involves the building of a model using labeled data so that we can assign labels to data that doesn't have them. In the NLP context, we would have to label each document "Ireland" or "Seattle" and then we can build a classifier that will take a document that we haven't seen and it'll decide if the document should be labeled "Ireland" or "Seattle". These classes need not be topical. We could just as easily have had the classes be genres, reading level, or even era of composition.

Here we want to use an algorithm that works with the two things we have: a particular representation and a relatedness measure. [k Nearest Neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) fits that bill perfectly. kNN is not hugely different than the Querying problem when the Document A is not in the corpus. We transform Document A, the document we want to classify, into our representation (TF, say) and then calculate the relatedness measure (cosine similarity, for example) between it and all the other documents. We then take the k nearest (most similar) of the documents and determine the *majority* class label. If k = 5, we return 5 documents and if 4 are "Ireland" then we classify the document as "Ireland". If 3 are "Seattle", then we classify the document as "Seattle". For classification, k is normally chosen to be an odd number so there is a majority. Picking a good value for k requires a bit of background in model evaluation which we might cover in this blog at some point.

So that's basically it for working with text. You get your text into some structured representation and then apply some machine learning algorithm that uses distance or similarity as the main component and pick a metric. Some representations suggest other approaches. For example, term *frequencies* itself suggests a Bayesian approach might be used and indeed it is easy to adapt both the Binomial and Multinomial Naive Bayes Classifiers to NLP classification problems. Perhaps another blog post can cover those.

I have, however, until now, swept the hardest part under the rug. How did I get the List of words,

```python
words = ["wet", "cold", "wet", "cold", "guinness", "rocky", "leprechaun", "wool"]
```

?

And now we will find that *this* is the hard part.

## Working with Raw Text

How do we turn a tweet, a sonnet, a novel or a scientific article into a Bag of Words in the first place? At first glance, it seems easy enough...just break one whitespace. Well, that's easy enough for *English*, what about Chinese? What do we do about punctuation? In the old days, we simply removed it. What do we do in the Age of Emojis? Is ;) a word? Of course, now we have actual emoji names with :smile: is that different than smile or :)? Do we count all those "of" and "the" and "and"? Shouldn't "am", "be", "is" all be the same? What about "ur" and "you're"? Eek.

This is certainly where "dirty" comes in. Start simple and iterate with an eye to your particular use case.

These questions come up so often they have names:

1. Normalization - how are we going to treat capitalization and punctuation?
2. Tokenization - what in the text is a token (what I've been calling "term").
3. Stopwords - removing "noise" words that don't add semantic content: "of", "the", "an", etc.
4. Stemming - reducing words to their roots: "automate", "automation", "automatic" to "automat"
5. Lemmatization - sort of grammar backwards, turn "am", "are", "is" into "be" or "car", "car's", "cars" into "car".

Many of these are not trivial. What do we do in German with the word *Lebenversicherungsgesellschaftsangestellter* ("life insurance company employee")? Is that one token? On the other hand, some languages come pre "lemmatized"...in Irish, "is" is always "ta" in the present tense (although you might have problems with dialects where "ta me"--I am--becomes "taim").

This isn't necessarily the end. We already talked about the emoji "problem". So far, we've treated words as just plain tokens. However, I can imagine some applications where knowing that the word "bank" is used as a verb in some cases and a noun in other cases, would be very important. So we might need to calculate TF on Tuples of parts of speech and tokens. But it isn't where I would *start*. At the other end of the spectrum, the text might be "dirtier" than the typical string from a database. You may need to remove HTML tags, for example.

Unless the use case calls for it, at least when working with English text, a good starting point is normalization (remove all punctuation, reduce to lowercase), tokenize on whitespace, and remove stopwords. Assuming you have a document in memory, how do you do that in Python? I'm going to do it in a very generic way without any external libraries. This will get us to **Stopwords**. If you after getting that far, you might consider `nltk` for stemming and lemmatization.

### A Very Generic Approach

The general way focuses on stock functions available in most languages and publicly available lists of stopwords. I have rarely seen so much variation in function names as that seen in the seemingly simple task to convert a string to lower case.

In Python, the function is `lower()`. After converting a string to lowercase, we'll need to tokenize it by breaking on whitespace and removing punctuation.

For punctuation, we can simply use a regular expression to replace everything that isn't whitespace or a letter with the empty string:

```python
>>> import re
>>> re.sub( r'[^\w\s]', '', "This is. not! the way. it works?")
'This is not the way it works'
```

For actual tokenization, we can then split on whitespace:

```python
>>> re.split( r'\s', re.sub( r'[^\w\s]', '', "This is. not! the way. it works?"))
['This', 'is', 'not', 'the', 'way', 'it', 'works']
```

And now we are were we started. Such a List of words can be fed into `tf()`. Let's combine it into one function:

```python
import re

def preprocess_text( text):
  lowercased = text.lower()
  normalized = re.sub( r'[^\w\s]', '', lowercased)
  tokenized = re.split( r'\s', normalized)
  return tokenized

words = preprocess_text( "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way – in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.")

print Counter(words)
```
which gives:

```
Counter({'the': 14, 'of': 12, 'was': 11, 'it': 10, 'we': 4, 'all': 2, 'direct': 2, 'us': 2, 'in': 2, 'its': 2, 'before': 2, 'for': 2, 'had': 2, 'epoch': 2, 'going': 2, 'season': 2, 'were': 2, 'age': 2, 'times': 2, 'period': 2, '': 1, 'superlative': 1, 'being': 1, 'spring': 1, 'received': 1, 'some': 1, 'authorities': 1, 'best': 1, 'like': 1, 'darkness': 1, 'to': 1, 'comparison': 1, 'only': 1, 'everything': 1, 'way': 1, 'so': 1, 'hope': 1, 'good': 1, 'belief': 1, 'degree': 1, 'that': 1, 'far': 1, 'evil': 1, 'wisdom': 1, 'short': 1, 'worst': 1, 'nothing': 1, 'insisted': 1, 'noisiest': 1, 'present': 1, 'winter': 1, 'on': 1, 'heaven': 1, 'incredulity': 1, 'light': 1, 'or': 1, 'foolishness': 1, 'despair': 1, 'other': 1})
```
which shows the importance of removing stopwords.

### Removing Stopwords

Stopword lists are available for many languages.  You can try [English Stopwords](http://www.ranks.nl/stopwords) for example. Unfortunately, it's not in a very nice format for use by a computer program. A little typing and multiple cursor-fu later, you might have something like:

```python
en_stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
```

Of course, we already have a problem, we were so dirty that any place we had "weren't", we now have "werent" which wouldn't be so bad except that "she'll" is now "shell". You may get away with it, you may not.

There is a temptation to remove stopwords too soon. There isn't much overhead to doing all the work we did and then simply removing any key in the TF Dict that is a stopword. If we want to handle other, at least European, languages we can change our function `preprocess_text` to take a List of stopwords.

```python
def preprocess_text( text, stopwords):
  lowercased = text.lower()
  normalized = re.sub( r'[^\w\s]', '', lowercased)
  tokenized = re.split( r'\s', normalized)
  counts = Counter( tokenized)
  for t in counts.keys():
    if t in stopwords:
      del counts[ t]
  return counts

words = preprocess_text( "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way – in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.", en_stopwords)

print words
```

which gives us:

```
Counter({'direct': 2, 'us': 2, 'epoch': 2, 'going': 2, 'season': 2, 'age': 2, 'times': 2, 'period': 2, '': 1, 'superlative': 1, 'spring': 1, 'received': 1, 'authorities': 1, 'best': 1, 'like': 1, 'darkness': 1, 'comparison': 1, 'everything': 1, 'way': 1, 'hope': 1, 'good': 1, 'belief': 1, 'degree': 1, 'far': 1, 'evil': 1, 'wisdom': 1, 'short': 1, 'worst': 1, 'nothing': 1, 'insisted': 1, 'noisiest': 1, 'present': 1, 'winter': 1, 'heaven': 1, 'incredulity': 1, 'light': 1, 'foolishness': 1, 'despair': 1})
```
which is pretty flexible. We can use the keys for the Boolean representation, we can use the values for the Count representation and it's not difficult to transform it to the Frequency representation:

```python
def convert_to_tf( counts):
  n = sum( counts.values())
  return {t: v/n for t, v in counts.items()}

print convert_to_tf( words)
```

which gives us:

```
{'': 0.021739130434782608, 'superlative': 0.021739130434782608, 'spring': 0.021739130434782608, 'incredulity': 0.021739130434782608, 'direct': 0.043478260869565216, 'nothing': 0.021739130434782608, 'foolishness': 0.021739130434782608, 'best': 0.021739130434782608, 'darkness': 0.021739130434782608, 'everything': 0.021739130434782608, 'epoch': 0.043478260869565216, 'going': 0.043478260869565216, 'way': 0.021739130434782608, 'hope': 0.021739130434782608, 'good': 0.021739130434782608, 'belief': 0.021739130434782608, 'degree': 0.021739130434782608, 'comparison': 0.021739130434782608, 'far': 0.021739130434782608, 'season': 0.043478260869565216, 'evil': 0.021739130434782608, 'wisdom': 0.021739130434782608, 'heaven': 0.021739130434782608, 'worst': 0.021739130434782608, 'authorities': 0.021739130434782608, 'insisted': 0.021739130434782608, 'present': 0.021739130434782608, 'winter': 0.021739130434782608, 'received': 0.021739130434782608, 'short': 0.021739130434782608, 'like': 0.021739130434782608, 'light': 0.021739130434782608, 'age': 0.043478260869565216, 'us': 0.043478260869565216, 'times': 0.043478260869565216, 'despair': 0.021739130434782608, 'period': 0.043478260869565216, 'noisiest': 0.021739130434782608}
```

I do notice one bug in `preprocess_text`. Our normalization allows for there to be the empty string as a possible token. We can revise the code to get rid of that with a `del counts['']` at the right spot.

### Using `nltk`

Python actually has a library for processing natural language called `nltk`. If you able to use Python, using `nltk` has two advantages. First, there are functions designed specifically tokenize languages. Second, it includes functions for the more complicated transformations we might want to apply such as stemming and lemmatization. Of course, you have to make sure that if you develop something in Python, you can deploy it in your environment.

I suggest reviewing the relevant sections of their book, [Natural Language Processing in Python](http://www.nltk.org/book/), especially the chapter on [Processing Raw Text](http://www.nltk.org/book/ch03.html) which talks about a stemmer and a basic version of lemmatization.

### Productionization

Actually putting such a system into production can lead to a whole slew of questions:

1. Do you need to recover the actual document in anyway? If so, you may need to add metadata to your representation with a document name or id so that you can link back to the original item.
2. If you can compute representations ahead of time or even relatedness ahead of time, where are you going to store it? Since Postgres, for example, stories JSON, storing the Dict as a JSON document is very easy. If you use a different database, you might not have as easy a time.
3. Do you have one function that does everything or a bunch of functions that can be chained together? If you know that you're always going to use normalization, tokenization, stopwords, TF and cosine similarity, you can have one function that simply takes a string and returns the normalized (unit) sparse vector of terms. However, especially in the experimentation stage, you might just return the List of words so you can experiment with Boolean, Counts, TF or TF-IDF. Or you might just return TF so you can experiment with cosine similarity and jaccard index.
4. If you are developing in Python, can you deploy in Python? Developing a whiz-bang model in Python that needs to be deployed in Ruby is a waste of time if you can't build the same production model in Ruby.  There are a lot of options, though:
  1. Using only what is available in Ruby, see how well a simple model works. Our simple Python model used functions that would definitely be available in Ruby.
  2. See if Ruby has the specialized libraries. Ruby may have the specialized libraries. In general, I suggest starting *and deploying* simple and then iterating. If you've already done this, then if an exploratory model with stemming works better, see if Ruby has a Porter stemmer.
  3. See if you can set up a separate service. In the dawning age of Microservices, an NLP *Python* microservice might be feasible.
  4. Can you set up a Python backend process and precalculate. For things you can do upfront, like clustering. You might simply be able to run a Python job every night rather than relying on a real time or any time algorithm in Ruby.

## Other Uses For text

Using the actual words/terms in a document are not the only ways to use text in a classification problem. For some problems, using the metadata (author, genre, data of publication, etc.) is the real goal and generating features from the text can augment this. An example might be parsing the text and determining the reading level of the text based on the vocabulary. There are other inventive ways to generate and include metadata about text from text in your models.

## Summary

Exactly how quick this was is debatable. NLP is not particularly easy but it is tractable. It helps to have clear goals in mind and start simple, adding complexity as it is required and to the extent you, your team and infrastructure can support it. As for the dirty, that'll ultimately be up to you, the use case and your environmental constraints. I will try to cover some of the more nuanced topics at a later time.
