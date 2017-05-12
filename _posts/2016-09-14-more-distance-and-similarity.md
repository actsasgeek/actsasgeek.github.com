---
layout: post
title: "More Distance and Similarity for NLP"
date: 2016-09-14 00:00
tags: nlp, python, machine learning
---

Last time, I talked about "Quick and Dirty NLP" which wasn't particularly "quick" (unless you consider the other options) but definitely had a spectrum of "dirty".

This time I want to fill in a few of the measures and metrics I left as exercises for the reader. As a reminder, we cannot generally use the the functions in `numpy` or `scipy` because we are dealing with a sparse vector representation of our term presences/counts/frequencies, Dicts in Python. For the remainder of this blog post, I'm going to concentrate on Term Frequencies.

## Fuzzy Jaccard Index
I have no idea if Jaccard was actually hirsute or not...this has to do with crisp versus fuzzy.

We discussed the Jaccard Index last time. It is a measure (ratio) of the terms two documents have in common to the terms those two documents use:

$$Jaccard = \frac{|A \cap B|}{|A \cup B|}$$

As with all such things, if that works for you. Perfect. Don't start out with complex things, either to implement or to explain to non-technical people (and do not discount the latter. If people do not understand something they are far less likely to support or trust it).

From the Jaccard Index point of view, the following table is sufficient to calculate the index:

| Term     | Document A | Document B |
|----------|------------|------------|
| cold     | 1          | 1          |
| wet      | 1          | 1          |
| hipster  | 0          | 1          |
| green    | 1          | 1          |
| guinness | 1          | 0          |

<br/>

because we are only noting the *presence* of terms. After all, we are talking about sets of words and words are either in the set or not.

However, this is the Classical Logic point of view. In the [Fuzzy Logic](https://en.wikipedia.org/wiki/Fuzzy_logic) point of view (horribly named), we can distinguish between degrees of membership. If we're talking about "Long Rivers" then all the rivers of the world belong to this set, just to differing degrees. This is what makes it fuzzy. On the other hand, with a crisp set, we must come up with a threshold that acts like a bouncer at an exclusive nightclub, this river is in the set; this river is out.

To my thinking, if cold appears 27% of the time and green appears 2% of the time, they shouldn't necessarily get the same weight. So what we would want to do is apply Fuzzy Sets instead of crisp ones. One way of hacking this is to use Term Frequency (TF):

| Term     | Document A | Document B |
|----------|------------|------------|
| cold     | 0.27       | 0.39       |
| wet      | 0.03       | 0.20       |
| hipster  | 0          | 0.15       |
| green    | 0.15       | 0.26       |
| guinness | 0.25       | 0          |

<br/>

If we count the term frequency as being the degree of membership. Then we can use fuzzy operators for intersection and union to calculate a Fuzzy Jaccard Index.

### Fuzzy operators

Intersection can be seen as logical AND. Is the element in both sets? Using 1 to denote presence and 0 to denote absence (a classical *characteristic function*), we have the following:

{% highlight pre %}
AND(1, 1) == true
AND(1, 0) == false
{% endhighlight %}

Since true and false are often equated with 1 and 0 respectively, we can switch out AND for MIN and get the same result:

{% highlight pre %}
MIN(1, 1) = 1
MIN(1, 0) = 0
{% endhighlight %}

So if we want a Fuzzy Intersection, following this train of thought, we are going to use MIN (there are many alternatives).

Similarly, Union can be thought of as OR:

{% highlight pre %}
OR(1, 1) == true
OR(1, 0) == true
OR(0, 0) == false
{% endhighlight %}

And we can think of this as MAX:

{% highlight pre %}
MAX(1, 1) = 1
MAX(1, 0) = 1
MAX(0, 0) = 0
{% endhighlight %}

If we look at the formula for the Jaccard Index, it uses the size of a set. If we converted the set representation to 1's and 0's, then the size of the set is simply the sum of the values. We can do the same thing to calculate the size of intersection by using MIN then SUM. And now if we substitute our crisp set memberships for fuzzy set memberships, we can calculate a Fuzzy Jaccard Index.

In code,

{% highlight python %}
def fuzzy_jaccard( a, b):
  fuzzy_intersection = sum([min( a[ t], b.get( t, 0.0)) for t in a.keys()])
  crisp_union = set(a.keys()).union( set( b.keys()))
  fuzzy_union = sum([max( a.get( t, 0.0), b.get( t, 0.0)) for t in crisp_union])
  return fuzzy_intersection / fuzzy_union
{% endhighlight %}

Here are some TFs from last time:

{% highlight python %}
tfs = [{'shamrock': 0.25, 'leprechaun': 0.125, 'cold': 0.125, 'green': 0.125, 'wet': 0.375},
 {'evergreen': 0.14285714285714285, 'ipa': 0.14285714285714285, 'rocky': 0.14285714285714285, 'hipster': 0.14285714285714285, 'wet': 0.2857142857142857, 'cold': 0.14285714285714285},
 {'rocky': 0.125, 'guinness': 0.125, 'wet': 0.25, 'leprechaun': 0.125, 'cold': 0.25, 'wool': 0.125},
 {'rocky': 0.125, 'salmon': 0.125, 'pacific': 0.125, 'hipster': 0.125, 'wet': 0.25, 'cold': 0.25},
 {'cold': 0.375, 'wool': 0.125, 'hipster': 0.125, 'rocky': 0.125, 'wet': 0.25}]
{% endhighlight %}

And the Jaccard Index from last time:

{% highlight python %}
def jaccard(a, b):
  d1 = set(a.keys())
  d2 = set(b.keys())
  intersection = d1.intersection( d2)
  union = d1.union( d2)
  jaccard = len( intersection)/len( union)
  return jaccard
{% endhighlight %}

Let's compare them:

{% highlight python %}
>>> jaccard( tfs[0], tfs[0])
1.0
>>> fuzzy_jaccard( tfs[0], tfs[0])
1.0
{% endhighlight %}

Nice. A document should always be the same as itself. Here are a few more comparisons. When comparing Document 1 (index 0) with Document 2 (index 1), the fuzzy jaccard is a bit higher. Conversely, the fuzzy jaccard is a bit lower when comparing to Document 3 (index 2).

{% highlight python %}
>>> jaccard( tfs[0], tfs[1])
0.2222222222222222
>>> fuzzy_jaccard( tfs[0], tfs[1])
0.25842696629213485
>>> jaccard( tfs[0], tfs[2])
0.375
>>> fuzzy_jaccard( tfs[0], tfs[2])
0.3333333333333333
{% endhighlight %}

Although the difference does seem large here. It can make a difference to some applications. Like everything else YMMV (your mileage may vary).

## Euclidean Distance

Euclidean distance is the workhorse of distance metrics and while it has its flaws, many of them (such as features of different magnitudes), don't show up in this application. Euclidean distance is the square root of the sum of squared differences between the two vectors. Strictly speaking, because we are often only *comparing* distances in these kinds of applications, we need not take the final square root but we will in this case:

{% highlight python %}
import math
def euclidean_distance( a, b):
  all_terms = set(a.keys()).union( set( b.keys()))
  return math.sqrt(sum([(a.get( t, 0.0) - b.get( t, 0.0))**2 for t in all_terms]))
{% endhighlight %}

We can check our function using a well known result:

{% highlight python %}
>>> euclidean_distance({"a": 3, "b": 0}, {"a": 0, "b": 4})
5.0
{% endhighlight %}

## Correlations

And this is where it gets, well, [supercalifragilisticexpialidocious](https://www.youtube.com/watch?v=tRFHXMQP-QU). Mainly because all the calculations involving correlations are combinations of means, variances, covariances, etc. and they really want regular vectors, not sparse ones. Those zeros count!

Below we will use the 2nd definition of Pearson's, after the first re-arranging, from Wikipedia:

{% highlight python %}
def pearsons_correlation( a, b):
  all_terms = set(a.keys()).union( set( b.keys()))
  n = len( all_terms)
  sum_ab = sum([a.get(t, 0) * b.get(t, 0) for t in all_terms])
  sum_a  = sum(a.values())
  sum_b  = sum(b.values())
  sum_a2 = sum([v**2 for v in a.values()])
  sum_b2 = sum([v**2 for v in b.values()])
  numerator = n * sum_ab - sum_a * sum_b
  denominator = math.sqrt(n * sum_a2 - sum_a**2) * math.sqrt(n * sum_b2 - sum_b**2)
  return numerator / denominator
{% endhighlight %}

There is also a single pass formulation if efficiency is required. We can test the result here:

```
>>> pearsons_correlation({"a": 1, "b": 2, "c": 3}, {"b": 1, "c": 2})
1.0000000000000002
```

Not bad.

And now it gets complicated (as if it weren't before!). *Spearman's* Correlation Coefficient is the Pearson's Correlation Coefficient of the *ranks* of the values. Pearson's tests for a *linear* correlation between the two vectors whereas Spearman's looks to see if there is a *monotonic* relationship between the two vectors. The distinction might be important for your application.

But in a sparse representation, the most frequent count is probably going to be zero so a lot of terms all get the same rank. This actually applies Pearson's as well and is part of the reason for introducing the Fuzzy Jaccard Index. One reasonable variation would be to look at the correlation (Pearson's or Spearman's) of the words that two documents *do* have in common rather than all the terms encompassed by the two documents. Nevertheless we're going to forge through with the latter.

We will need to create a representation of the *non* sparse vector for each document, for the combined vocabulary and this time we can use `scipy`

1. Determine all terms.
2. Put them into a canonical ordering (sort them!).
3. Create a vector of term frequencies.

Because the result of #3 is a regular vector of data with a paired ordering. We can use the `scipy` implementation of `spearmans`.

{% highlight python %}
def convert_to_non_sparse(a, b):
  all_terms = list( set(a.keys()).union( set( b.keys())))
  all_terms.sort()
  new_a = [a.get( t, 0.0) for t in all_terms]
  new_b = [b.get( t, 0.0) for t in all_terms]
  return new_a, new_b
{% endhighlight %}

Now that we have non-sparse representations, we can use `spearmanr`:

{% highlight python %}
import scipy.stats as stats

def spearmans_correlation( a, b):
  new_a, new_b = convert_to_non_sparse( a, b)
  rho, _ = stats.spearmanr( new_a, new_b)
  return rho
{% endhighlight %}

Note that we could have done something similar to use `scipy`'s Pearson's correlation coefficient.

Working with the correlation coefficients is interesting because, unlike the other measures and metrics, they can be negative. That is, the word distributions of two documents can be negatively related.

## Summary

And that fills in the distance metric (Euclidean) and correlations (Pearson's, Spearman's) that we noted before but did not describe as well as introducing a Fuzzy Jaccard Index.

Cheers.
