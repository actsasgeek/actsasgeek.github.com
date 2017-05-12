---
layout: post
title: "Deploying Linear Regression"
date: 2015-01-28 00:00
tags: linear regression, R, Python, Clojure
---

The other day I found myself with, what I thought, was a simple task: read a file of training data, use it to compute a linear regression model, read a file of raw data, use the model to predict and write the predictions back out to a file. I wanted to do this in Python and, naturally, I googled "linear regression Python". This did not result in the cut-and-paste solution I was looking for. However, there were breadcrumbs and eventually I hacked together a solution. This blog post is an extended discussion of the various options available to solve this problem in Python, R and Clojure.

## Linear Regression

While linear regression--and its cousin, logistic regression--may not be getting all the press these days, they are simple algorithms that can often get you quite far especially if you have a "lot of data". This has been discussed both by Norvig [The Unreasonable Effectiveness of Data](https://www.youtube.com/watch?v=yvDCzhbjYWs) and Rajaraman [More Data Usually Beats Better Models](http://anand.typepad.com/datawocky/2008/03/more-data-usual.html). Additionally, if your goal is understanding rather than prediction (or in addition to prediction), the latest and greatest of the machine learning world cannot help you. Random forests, deep learning neural networks, support vector machines are all black boxes into which you cannot peer.

Today we are not going to get into how to intepret a linear or logistic regression or how to pick features...I'm assuming that the data science part is done. We're simply interested in putting the model into production with a very simple notion of "in production". We're opting to retrain the model and create batch predictions. We could just as easily train the model and store it in a separate step then retrieve it and do just-in-time predictions.

## Running a linear regression in Python

As I mentioned, I googled "linear regression Python" and got a few hits. The first was [Basic Linear Regressions in Python](http://jmduke.com/posts/basic-linear-regressions-in-python/) which suggests using [pandas]() and [numpy](). `numpy` is the standard numerical library for Python. It along with [scipy]() are *de rigeur* libraries for any data scientist using Python. `pandas` is a library for Python that adds R's DataFrame data structure to Python's numpy enriched repetoire. np.polyfit( x, y, 1). R's DataFrame has nice abstractions and if you must use both Python and R, having it in both languages is a godsend. This, at least, leads to an answer covering how to read the training and raw data from file and how to write the predictions to file:

{% highlight python %}
import pandas as pd

training_data = pd.read_csv( "training_data.tsv", sep="\t") # header=True is the default
raw_data  pd.read_csv( "raw_data.tsv", sep="\t")

# miracle occurs

prediction.to_csv( "predictions.tsv", sep="\t", index=False)
{% endhighlight %}

I am not immediately fond of the `np.polyfit` approach. In the code Justin provides, \\(x\\) and \\(y\\) are just vectors and we have multiple variables, \\(X\\). Ugh, my ADD kicks in...next search result, please!

## Scikit.Learn

[scikit.learn](http://scikit-learn.org) is a nice machine learning package for Python which includes [linear regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). Reading the documentation, I can see that I can use the `fit(X, y[, n_jobs])` method to build a model and `predict(X)` to apply it and get predictions. Great! How do I get the DataFrame I have to be X and y? Not sure. Next search result please...

## Statsmodels

[statsmodels](http://statsmodels.sourceforge.net) is also a nice *statistics* modeling package for Python which includes [linear regression](http://statsmodels.sourceforge.net/devel/regression.html). If you don't have a statistic background, the documentation here might be a bit tougher. You have to know that the most basic linear regression is also called "ordinary least squares" or OLS which is the name of the function/module (it's sometimes not entirely clear in Python) you need to use. Their minimal example is:

{% highlight python %}
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
dat = sm.datasets.get_rdataset("Guerry", "HistData").data

# Fit regression model (using the natural log of one of the regressors)
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

# Inspect the results
print results.summary()
{% endhighlight %}

Most of this is not useful to me. I don't want to read an existing dataset and I don't want to print out the summary of the results. What *is* interesting to me is this line:

{% highlight python %}
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
{% endhighlight %}
```

The formula string appears to have the same format as the formula DSL in R. That's really cool. What is this black magic?

## Patsy

The [patsy](http://patsy.readthedocs.org/en/latest/) package lets you specify R style formulas over `pandas` DataFrames. `statsmodels` uses the `patsy` library to provide R-style formula notation for *its* library. But this interesting bit comes in the `statsmodels` documentation section that describes using using patsy with modules that do not yet support `patsy` directly. The code fragment shows:

{% highlight python %}
import patsy
f = 'Lottery ~ Literacy * Wealth'
y,X = patsy.dmatrices(f, df, return_type='dataframe')
print y[:5]
print X[:5]
{% endhighlight %}

Let's keep that in mind for later.

## The Main Event

Combining these pieces, it's now possible to solve our original problem: reading files, building a model and making predictions, and writing those predictions back out. We need only replace `# miracle occurs` with code that trains the model, makes predictions and adds the predictions to the raw data before saving it:

{% highlight python %}
import pandas as pd
import statsmodels.formula.api as smf

training_data = pd.read_csv( "training_data.tsv", sep="\t")
raw_data = pd.read_csv( "raw_data.tsv", sep="\t")

model = smf.ols( "y ~ x1 + x2 + x3 + x4", data = training_data).fit()
predictions = model.predict( raw_data)
raw_data[ "predictions"] = pd.Series( predictions)

raw_data.to_csv( "predictions.tsv", sep="\t", index=False)
{% endhighlight %}

This solves the original problem so, yay.

## New tricks for scikit-learn

Using `patsy` we can work with scikit-learn using formulas as well. the `dmatrices` function returns `y,X` as either `dataframe` or `matrix`. This won't help us with the raw data, however, because we need to specify an outcome variable, \\(y\\). A quick google for "pandas dataframe to matrix" reveals that Pandas' DataFrame object has a `as_matrix` function which we can use.  The only "gotcha" is that `patsy` returns a *design matrix* that includes an intercept column of all ones. This means we need to fit a model without an intercept (because the coefficient on an implicit column of all ones is exactly what the intercept is) and we will need to pad our raw data with ones as well when we make predictions. The scikit-learn version of the above code is as follows:

{% highlight python %}
import pandas as pd
import patsy
from sklearn.linear_model import LinearRegression

training_data = pd.read_csv( "training_data.tsv", sep="\t")
raw_data = pd.read_csv( "raw_data.tsv", sep="\t")

y, X = patsy.dmatrices( "y ~ x1 + x2 + x3 + x4", training_data, return_type="matrix")
model = LinearRegression(fit_intercept=False).fit( X, y)

raw_data.insert(0, "Intercept", 1)
predictions = model.predict( raw_data.as_matrix())

raw_data[ "predictions"] = pd.Series( predictions.transpose()[0,])
del raw_data[ "Intercept"]

raw_data.to_csv( "predictions.tsv", sep="\t", index=False)
{% endhighlight %}

It takes a few more lines because the use of `patsy` is not baked in, but this illustrates that `patsy` can be used with other libraries that either expect a DataFrame or Numpy `matrix/array` type. However, the extra steps required to deal directly with the design matrix and creating a model with an explicit intercept term may not be worth the ability to use patsy to specify a formula. If you did this often, you could certain write helper functions and subclasses to ease the pain. The array of array type for the `predict` function is weird but that's statsmodel's return type for predict.

## R

Although my problem at the time required me to use Python, I do sometimes need to use R and wondered how the same thing might be accomplished in that language. Most of it was straight forward as `read.csv` and `write.csv` are well-known. The linear model itself is also not difficult to figure out. The killer was finding an example of using the result of `lm()` to predict values on *other* data. The `fit` returned by `lm()` has a field called `predicted` but those are the predicted values for the training data. Googling again for "r apply lm to new data" resulted in a hit for the `predict` function. Otherwise, everything is very similar to the Python except that as R is a statistical language, most of the libraries we need are already imported for us.

The code is as follows:

{% highlight r %}
training_data <- read.csv( "training_data.tsv", sep="\t") # header=TRUE is the default
raw_data <- read.csv( "raw_data.tsv", sep="\t")

fit <- lm( y ~ x1 + x2 + x3 + x4, data=training_data)
predictions <- predict( fit, raw_data)
raw_data[ "predictions"] <- predictions

write.table( raw_data, "predictions.tsv", sep="\t", quote=FALSE, row.names=FALSE)
{% endhighlight %}

Very easy as one would expect for a domain focused language.

## Clojure

As noted in the introduction, linear regression is a good starting place when you want to add "smarts" to an application. These applications do not always have the ability to call out to R but have to use the language they're written in like Python, Ruby, Java or...Clojure. Of course, if your host language is Python, this problem is always solved above. Most of my day-to-day engineering is done in Clojure and so I wondered how I would accomplish this same task in Clojure. Basically it amounts to hoping that your language has a machine learning or statistical library and finding it. For Clojure, this is [Incanter](http://incanter.org). Otherwise, you might have to explore other options (for example, generate the model offline, pickle it and unpickle it to use it in production. At that point, prediction just a vector dot product of the coefficients and the data).

Incanter can do a lot more than just linear regression. It aspires to being a Lisp-based statistical language (DSL) with the functionality of R but embedded and available to the larger Clojure ecosystem. Since the use of a REPL (read-print-eval-loop) is so central to Lisp, interactive statistical and machine learning applications would seem like a natural fit for a language like Clojure. However, in this instance, we are strictly interested in the *non* interactive properties of the library.

While Incanter takes quite a few ideas from R, it would have been nicer if Incanter had kept the nomenclature the same as in R. For example, R's `DataFrame` is Incanter's `dataset`. Ick. There also appear to be some holes in Incanter's implementations. There is linear regression and Bayesian linear regression but no logistic regression. In fact there is no GLM implementation. Weird. Let's stick with vanilla OLS for now. For the others, [Incanter and the GLM](http://www.ccri.com/2010/02/17/incanter-and-the-glm/) looks promising.

What follows is more of a REPL session (I used [LightTable](http://lighttable.com)) than a function you can call, keeping in the spirit of the examples so far. Ideally, all of these would be functions of three parameters (training file, raw data, predictions file) that wrap our investigations into a reusable, callable bundle of intelligence (or into an even more general refactoring). The Clojure version of our solution is:

{% highlight clojure %}
(ns playground
	(:require [incanter.core :refer :all]
					  [incanter.io :as io]
					  [incanter.stats :as stats]))

(def training-data (io/read-dataset "training_data.tsv" :delim \tab :header true))
(def raw-data (io/read-dataset "raw_data.tsv" :delim \tab :header true))

(def y (sel training-data :cols "y"))
(def X (sel training-data :cols ["x1" "x2" "x3" "x4"]))

(def model (stats/linear-model (to-vect y) (to-matrix X)))

(def predictions (map #(stats/predict model %) (to-matrix raw-data)))

(def intermediate (add-column "y" predictions raw-data))
{% endhighlight %}

Incanter is not as well documented as either R or Python's statistics libraries which is unfortunate. It takes quite a bit more time to piece things together and several trips to the [API documentation](http://liebke.github.io/incanter/core-api.html) and [Stackoverflow](http://stackoverflow.com/questions/tagged/incanter).

## Summary

Only R worked right out of the box. This is not surprising since the task is clearly in R's wheelhouse. Getting Python (scikit-learn and statsmodels) and Clojure to work correctly was a huge pain. First, while I am a fan of dynamically typed languages, in this particular case, it was not always clear what each function expected and how you might get it. Secondly, and this is probably more important, all of the examples of linear regression were simple single variable problems. Thus some necessary steps were missing (transpose for example). Almost none of the examples use the prediction functions.

On the other hand, it's nice to see that, all things considered, this isn't much more difficult than any other specialized task in a general purpose programming language. If you have an application to which you plan to add some "smarts", it is much more likely that you started out with a general purpose programming language like Python or Clojure. In addition to discussing the interpetation of linear regression, in the future we might discuss other engineering aspects of fielding models such as serialization of coefficients and mixed language environments (build and serialize the model in R then deserialize and apply the model in Clojure).

Cheers.
