---
layout: post
title: "Review: Doing Data Science with Clojure (talk)"
date: 2016-09-27 00:00
tags: data science, clojure, conferences
---
I often joke that when I go to data science conferences, I want to be a data scientist and when I go to software engineering conferences, I want to be a software engineer. I find it interesting when they intersect. And so it is with [Doing Data Science with Clojure: The Ugly, the Sad, the Joyful](https://www.youtube.com/watch?v=PSTSO8K80U4).

For my tastes, there was too much hand waving in this talk. The library [huri](http://github.com/sbelak/huri) is undocumented and the improvements to [Gorilla REPL](http://gorilla-repl.org) have not been submitted upstream. There is this tension between a good conference talk and a blog post and, well, let's just say I applaud Simon Belak for getting up there.

## The Clojure Bit.

I really like programming with Clojure. Well, I like *software engineering* with Clojure. I like to think about a problem and then figure out how to compose Clojure's core functionality to solve it. I would be lying if it weren't hairy a few months later and you can't quite understand the higher order mess you've assembled but that's not Clojure's fault *per se*. Seemingly always in retrospect, I think to make function smaller, to give names to semantic units and to document data structures.

But I don't like doing that kind of "hammock time" with *programming* in Data Science...I want to save the hammock time for the Data Science! So I much prefer something quicker like Python and recently I've stopped treating Python like the red-headed step-child of my polyglotry (is that a word?).

That doesn't mean Clojure hasn't taught me a lot about how to do nicer programming in a language like Python. And if you really want to do some interesting FP, there's [Hy](https://github.com/hylang/hy) (although be careful, Hy has some sharp edges inherited from Python).

But I digress...in short, I would have dismissed this talk out of hand except that the first five minutes are gold. Data Science gold. Not the "I just found the secret to our company's future in a pile of data" gold, *actually doing data science* gold.

## 5 Key Points

**Point 1.** Your data should be managed in such a way that you can provide answers, ideally, in 2 minutes or less--no more than 20 minutes. You want to be able to influence discussions as they happen. Of course, that doesn't mean you can build a personalization model in 2 minutes but you should be able to calculate things like counts, sums, percentages on the fly, as they come up in discussions. This means that discussions will stay factually focused and not rely on "folksy wisdom", until you get back to them, next week. You can answer questions immediately.

  This goes to the Analytics part of Data Science (or exploratory Data Science). This was sorely missing at one place where I worked where there were at least  definitions of "revenue". Elsewhere, after I made an analytics database, I actually experienced how powerful this really is. Although you should temper your new found powers by reminding yourself that [you're not the most important function in your company](http://analyticsmadeskeezy.com/2012/11/05/check-yo-self-5-things-you-should-know-about-data-science-author-note/).

**Point 2.** Here's another plug for "your data isn't that big". Most data fits in RAM on one machine and if it does not, consider reducing it so that the key variables and values do so that you can answer questions in 2 minutes or less. Here is the canonical blog post on the topic, [Don't Use Hadoop -- Your Data Isn't that Big](https://www.chrisstucchio.com/blog/2013/hadoop_hatred.html).

**Point 3.** No Throwaways. Don't think "Oh, I just need to answer this question." Those questions get asked repeatedly and when the answer gets out into the world, you need to know where it came from. Everything should be reproducible, have provenance and revision management. At minimum, an email that answers "give me that number" should include the query in the footer.

**Point 4.** Distributions, think in Distributions. Averages lose detail and the interesting characteristics of the data. This is one advantage to working in something like Jupyter notebook. Because in many ways, Clojure is a special beast, this talk was about Gorilla REPL which gives you some of the features of Jupyter Notebook for Clojure. Beaker Notebook is trying to do the same for all languages (at once!). Note that Jupyter Notebooks is undergoing a revision/re-write on the front end to JupyterLab with even more features.

**Point 5.** Sharing Results. It's not enough to think about coming up with the result but you have to think about sharing results, is there one definitive source for a calculation? Is it persistent, searchable, reproducible? Is it always current? Don't send files over email. Include the methodology...the code (it answers, "how was this calculated?").

## Can we do this with Jupyter?

Looking at his lessons, obviously they're not limited to Clojure. Since I use Python and Jupyter for Data Science, is it possible to apply these lessons to those tools? Let's check each point, one by one.

1. This isn't limited to Clojure. You need an analytics database separate from production. It needs to be fast and if you do have big data, it needs to include well chosen data and aggregates/roll-ups.

2. Ditto, not limited to Clojure.

3. We can certainly do this in Python. Jupyter helps towards provenance and reproducibility. Github or a similar service can handle revision management). I do sometimes experience problems understanding diffs. I have started to look into different representations of notebooks so that diffs to explode.

4. Jupyter Notebook, Pandas, Matplotlib...no problem.

5. The last point is interesting. I have a notebook that I want to share it. How might one do it? I don't think the current Jupyter distributed notebook approach really answers this use case. Here's one strategy...

* Make the notebooks available on a simple web server in HTML format.
* You can execute notebooks from the command line.

```
jupyter nbconvert --to notebook --execute mynotebook.ipynb
```
Additionally, notebooks can be converted to other formats:

```
jupyter nbconvert mynotebook.ipynb
```

will automatically convert the notebook to an HTML form. You could skip this step but it is probably better to have the executed version of the notebook around.

One could run a simple Flask website that executes the notebooks and provides them automatically.

This is easiest if you run it inside a VPN, of course, because you don't necessarily need to worry about security. As long as you trust uploading random notebooks and executing them (basically, that the notebooks execute correctly and don't do anything malicious) and you don't mind who browses them, this is probably pretty easy. It becomes a lot more complicated if you need general security and permissions.

Adding search is an interesting addition. Most likely you would need to do something like read the contents of each generated file into elasticsearch or something similar which would need to be running along side the web server.

What would the MVP be?

1. upload notebooks
2. process on a timer that executes notebooks and converts them (probably once when uploaded and then at X:XX every day)
3. dynamically generate index.html of notebooks.

What could you add in Version 1.0?

1. add username/password protection.
2. add elastic search indexing and search page.

Analytical databases, Jupyter notebooks and microservices, oh my.

Cheers.
