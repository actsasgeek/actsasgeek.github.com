---
layout: post
title:  "Simple Natural Language Processing - Part I"
date:   2014-12-22 21:16
tags: general
---

This last semester (Fall 2014) I searched in vain for a simple, self-contained, of any sort of natural language processing
example that I could show my Data Science students. Next year, I'll show them this series of blog posts.

## Text as Structured Data

I spent most of the semester talking about rectangular arrays---tables---of data. Whether these are CSV files, TSV files,
Excel spreadsheets or some other proprietary format, or even RDBMS tables, they are the bread-and-butter of most data
analysis. Data like this generally goes by the name of "structured data" and while some people get upset with that particular
label ("All data is structured! *Romeo and Juliet* is clearly structured."), what is generally meant by the term "structured data"
is that the data is a table with rows as observations, columns as variables and cells as values. Most statistical techniques and machine learning algorithms require the data to be so structured. The problem, of course, is that text---like the text of this blog---is not structured 
like that.

Nevertheless, text *is* structured. If we examine the previous paragraph, the sentences are in a certain order. The words
within the sentences are in a certain order. It would seem altogether foolish and not a little bit surprising if we thought
we could get anywhere by assuming that the order did not matter. But that is exactly what the **bag of words model** for
natural language processing (NLP) does. If you could shake each paragraph of this page so that all the words fell to the bottom 
and then picked them up, one by one, and counted how many times each word appeared, you would end up with what we thought we lacked:
observations or rows (each paragraph), variables or columns (each word) and cells (the count of how many times each word appears
in the paragraph).

## Bag of Words Metrics

The values don't have to be counts. There are quite a number of different metrics. For example, you might record the mere presence
or absence of a word (usually encoded as 0 or 1) depending on whether or not a word from the *dictionary* appeared in the paragraph. Here
dictionary means "the universe of words that appeared in any of the paragraphs" (I could type an example but then it would appear in this
paragraph, we'd have a paradox and the world would end). For example, if the word "this" appears in the paragraph, we record a "1" and if "this" 
does not appear in the paragraph, we record "0". The values are thus **boolean**.

We already noted we could record counts (**absolute frequency**). For example, "this" has a count of 2 for the first paragraph; 2 for the second
paragraph and 1 for the third paragraph. This approach is slightly problematic because it favors longer paragraphs because they will have larger
counts in general and this merely a function of how I decided to break my paragraphs.

Therefore, there is the third approach which involves dividing the counts through by the total count of words in the paragraph and thus
calculate **relative frequency** for each word. There are even more elaborate approaches than this but we will stick with the boolean and
relative frequency approaches for now.

In NLP problems, the general term for a unit of an observation is a **document**. So if we were analyzing tweets, each not more than
140 characters, each tweet would be a document. If we were analyzing the works of William Shakespeare, each play, every sonnet, would also
be an individual document. And more generally, we do not refer to words but to **terms**.

## Ongoing Example

For an ongoing example, that would allow us to investigate different problems and different techniques, I picked the []() data set of moving ratings. While the original purpose of the data set is for supervised learning (predicting positive and negative reviews), we can simply ignore the labels in order to investigate unsupervised learning algorithms.

## Data Preparation

Regardless of whether we want to do supervised or unsupervised learning on the movie ratings, we're going to do the same basic preparation.
