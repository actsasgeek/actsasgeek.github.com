---
layout: post
title:  "Beginnings"
date:   2014-12-22 20:28
tags: general
---

After several failed attempts at keeping a blog, I've decided to dip my toe in the pool again. Hopefully this time
I will be able to keep it going and not have to delete the moribund corpse a few months or years hence.

I'm blogging mostly to clarify my own thinking about various topics that interest me: data science (whatever that is),
software engineering, computer science, statistics, programming languages, etc. Since I teach Artificial Intelligence
and Data Science, topics will invariablly come from there. Since I work as a software engineer, that will almost surely
provide fodder for blog posts as well. Hopefully the topics will be interesting enough that I'll keep hacking away at it.

Since this is the first blog post, I need to test some things out...there will invariably be some Clojure. That is the
language that I use most during the day:

```clojure
(defn my-juxt [x fs]
  (mapv (fn [f] (f x)) fs))
```

And some Python and R, since they are the co-*lingua franca* of data science:

```python
def my_juxt( x, fs):
  return [f( x)] for f in fs]
```

and

```r
my.juxt <- function( x, fs) {
  results <- c()
  for ( f in fs) {
    results <- c( results, f( x))
  }
  return( results)
}
```

I should mention that I only know enough R to get around. Interestingly enough all three languages support functions
as first class values.

Finally, there will be mathematical notation such as $$x^2$$ and:

$$a^2 + b^2 = c^2$$

neat, huh?