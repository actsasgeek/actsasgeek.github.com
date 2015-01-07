---
layout: post
title: "Genetic Algorithm - Part 1"
date: 2015-01-07 00:00
tags: genetic algorithm, evolutionary computation, optimization
---

The [Genetic Algorithm](http://en.wikipedia.org/wiki/Genetic_algorithm) (GA) was developed in the 1970's by [John Holland](http://en.wikipedia.org/wiki/John_Henry_Holland) at the University of Michigan. The algorithm involves simulating an abstract model of evolution and can be used for a wide range of optimization problems. Although for this blog post I'm only implementing the basic GA, I will discuss various options along the way and implement them in future blog posts.

# Abstract Evolution

There are several key aspects of evolution (as it was understood at the time) that are captured in the GA:

1. A fitness function converts the genotype into a phenotype and assigns a fitness value to the individual.
2. Each individual is represented by a binary string that encodes the genotype.
3. Fit individuals produce more offspring.
4. During sexual reproduction, there is the crossover of genetic material.
5. Genes are subject to random mutations.

\#1 and \#2 are related so we'll take them together:

## Fitness Function and Binary Encoding

The fitness function is a mathematical function or simulation that takes quantitative and qualitative arguments and produces some kind of score for the individual. The simplest such function might be something like "max-ones" that simply counts the number of positions in a bit-string that are "1". In this case, there is no difference between the genotype and phenotype.

A slight more complicated case might involve something like minimize $$\sum x^2$$ (to convert a minimization problem into a fitness value, we can calculate fitness as $$f'=\frac{1}{1+f})$$. In problems dealing with numeric values, the values have to be encoded into bits just as they are in computer memory. And just like computer memory, you have to set limits on the sizes of numbers that can be represented. For example, we may decide that for this particular problem we will only encode 3-bit positive integers. In that case, if we have four x's, we can encode the phenotype [3 4 1 2] as 3 = 011, 4 = 100, 1 = 001, and 2 = 010 or "011100001010". This is the genotype.

Most of the time we are going the other way, though. We have the genetic material of an individual such as "010111000101", we get the phenotype: [2 7 0 5] and calculate the fitness: $$f(x) = 2^2 + 7^2 + 0^2 + 5^2=78$$ and $$f'(x)=\frac{1}{1 + 78} = 0.012658228)$$ (We can see here that if $$f(x)=0$$ then $$f'(x) = 1$$, the maximum).

![Genetic algorithms breeding Skynet](http://imgs.xkcd.com/comics/genetic_algorithms.png)

Image from [XKCD](http://xkcd.com/534/)

## Survival of the Fittest

The basic idea of "survival of the fittest" is that individuals with some advantage are better able to survive, reproduce and thus pass on that advantage to their offspring. At the level of the population, this can be thought of as a stochastic process. There is a population of individuals with some degree of genetic diversity facing seemingly random obstacles: finding food, surviving to adulthood, enduring environmental changes. That genetic diversity confers relative advantages and disadvantages but dumb luck also plays a role. Surivors mate with other survivors and pass on their genes.

In the original GA, we model this process by randomly selecting individuals from the population...but not *uniformatly* at random. Instead, the probability of an individual being selected is proportionate to the relative fitness of the individual in the overall fitness of the population. For example, suppose there are only four individuals: A, B, C and D with fitnesses 3, 10, 4 and 5. The total fitness of the population is $$3+10+4+5=22$$. The *relative* fitnesses of the individuals are...A: 3/22 = 0.14, B: 10/22 = 0.45, C: 4/22 = 0.18 and D: 5/22 = 0.23. B is the most fit and thus has the highest probability of being drawn (45%).

## Crossover and Mutation

So far we have an abstract representation of an individual (genotype) that encodes a candidate solution (phenotype) to a problem (the fitness function). We also have a means of randomly selecting individuals proportionate to their fitness. So if, at random, we pick individuals with genes "011100001010" and "010111000101", how do we get "children"? We apply the crossover operator and the mutation operator.

For the crossover operator, we first identify a random crossover point somewhere on the genome, for example, locus 7. Using this point, we split each genome into two substrings. The first individual (let's call it "dad"), "011100001010", becomes "0111000" and "01010". And the second individual (you guessed it, "mom"), "010111000101", becomes "0101110" and "00101". We then recombine the strings using the first substring of dad and the second substring of mom: "0111000" + "00101" = "011100000101" and then the first substring of mom and the second substring of dad: "0101110" + "01010" = "010111001010". This leaves us with two new children "011100000101" and "010111001010".

Once we have the new children, we apply the mutation operator. The mutation operator randomly applies a "bit flip" to some locus on the genome. For example, the second child might have a bit flip at location 2 which changes the "1" to a "0": "010111001010" becomes "000111001010".

# Combining Selection, Crossover and Mutation

The genetic algorithm involves combining all of the aforementioned operators and re-applying them over a large population of candidate solutions for many generations. The outline of the process is as follows:

1. Pick a bit string genomic representation for your problem.
2. Write a fitness function that will appropriately decode the genotype and return a fitness value.
3. Pick parameter values for the genetic algorithm including:
	1. n - the size of the population.
	2. N - the number of generations to run the simulation.
	3. c - the probability of crossover.
	4. m - the probability of mutation.
3. Create an initial population of size n of random individuals.
4. Run the genetic algorithm for N generations:
	1. Pick N/2 sets of parents.
	2. For each set of parents apply the crossover and mutation operators.
	3. Completely replace the previous generation with the children.

Step 1 requires you to think about your problem and how you could encode it in a bit string. While this can be quite a challenge (determining the likely range of values for each variable, figuring out the size of the bit encoding necessary), don't run out and start using this code for problems you have laying around. In future blog posts we'll see that the bit encoding is largely unnecessary.

Step 2 requires you to write a function that does the decoding and then calculates the fitness of the individual. This can be either a mathematical function or a simulation of some kind. Note that it doesn't make much sense to use the genetic algorithm for problems that have analytical solutions or can be solved by other, deterministic means. The chief benefit of the GA is in optimizing problems that cannot be solved classically or perhaps have fitness values that can't be expressed exactly.

Step 3 is the usual parameter picking step. Both n and N are problem specific. If you end up with individuals having very long bit strings and a complicated problem, you may need a lot of them and have to run the algorithm for a long time. The probability of crossover is usually set in the 80%-90% range. This means that sometimes (10-20%) parents are just copied into the next generation. The probability of mutation is trickier. This is because this mutation rate can be expressed either as the probability of a specific gene flipping (in which case it should be very small) or the probability of a individual undergoing a mutation (in which case it should be higher) at a single random locus. The overall rate should generally work out to be about 1%.

Step 4 is simply our implementation of the algorithm. For this blog post, we're going to implement it in [Clojure](http://clojure.org/). Clojure is a functional programming language of the Lisp family that runs on the Java Virtual Machine (JVM). For the kinds of things we'll be doing, the ability to use functions as values will be very useful.

![Recipes evolved by genetic algorithms](http://imgs.xkcd.com/comics/recipes.png)

Image from [XKCD](http://xkcd.com/720/)

## Implementation

At a high level, we want a function that takes a problem configuration as an argument and returns something useful. The solution would be nice but what is the solution? The genetic algorithm is a stochastic algorithm that is normally used on difficult optimization problems, if we run the algorithm for 500 generations, the "best" solution seen might occur in generation 347 but lost because of a mutation in generation 348 and never seen again. So "the solution" is not necessarily the most fit individual in the final population, it would be the most fit seen over the entire run.

There are a host of other things we might return. For example, we might return statistics over the course of the run so that we can diagnose the performance of the genetic algorithm on our particular problem. And since we don't know if a better solution was only 50 generations away, some checkpointing would be useful as well. We'll leave those to another blog post. Perhaps even a web API to monitor the progress of the algorithm. Parallelization? Later. We're going to start very simple and work our way up to bigger and better things. The code for this blog post can be found here [dawkins](https://github.com/actsasgeek/dawkins) under a branch `2015-01-07`. Let's start.

If you have `git` and `lein` installed you can do:

{% highlight bash %}
$ git clone https://github.com/actsasgeek/dawkins.git
$ cd dawkins
$ git checkout -b 2015-01-07
$ lein deps
$ lein repl
user=>
user=> (require '[dawkins.genetic-algorithm :refer [genetic-algorithm]])
nil
{% endhighlight %}

Here's a quick explanation of the code you just loaded.

To make an individual, we need to be able to generate random bits of a certain length. We can also evaluate the fitness of an individual's genome when it's created and store it. For Clojure, the likely candidate is a map with keys `:genome` and `:fitness`. Although the genome can literally be a String, we're going to use a Vector of bits here.

{% highlight clojure %}
(defn rand-bit
	"Returns a random bit 0 or 1 with a 50/50 probability."
  []
  (if (< (rand) 0.5) 0 1))

(defn random-genome 
	"Given a genome of length k, returns a vector represenation of random bits"
  [k]
  (into [] (repeatedly k rand-bit)))

(defn random-individual 
	"Generate a random individual given a genome of length k and a fitness function f defined over that genome
   as a map with keys :genome and :fitness"
  [k f]
  (let [genome (random-genome k)]
    {:genome genome
     :fitness (f genome)}))
{% endhighlight %}

We can make our initial population by generating n random individuals. We will store the population in a Vector because we'll need to access individuals by index later:

{% highlight clojure %}
(defn random-population 
	"Generate a random n-sized population of individuals with a genome of length k and fitness f(genome)."
  [n k f]
  (into [] (repeatedly n #(random-individual k f))))
{% endhighlight %}

Let's start with the basic genetic algorithm operations: mutation and crossover. The mutation operator that we're going to implement takes a mutation rate and genome as parameters and flips a single bit in the genome with probability *mutation-rate*.

{% highlight clojure %}
(defn flip-bit 
	"For bit = 0, returns 1; 1, 0."
  [bit]
  (if (= bit 0) 1 0))

(defn mutate 
 	"This is the basic mutation operator for the Genetic Algorithm. It is but one alternative and other implementations are possible. For a 
  specified mutation-rate, pick a locus at random and flip the bit."
  [mutation-rate genome]
  (if (< (rand) mutation-rate)
    (let [mutation-point (rand-int (count genome))]
      (into [] (concat (take mutation-point genome) [(-> genome (nth mutation-point) flip-bit)] (drop (inc mutation-point) genome))))
    genome))
{% endhighlight %}

The crossover operator involves taking the two parents who are selected and with probability *crossover-rate* recombining their genomes. We're going to implement single-point crossover which means that the genomes will be recombined as we discussed above...at a single point. Note that it is possible for this to be a no-op and that's not really a problem, it just means that our effective *crossover-rate* is slightly smaller than what we explicitly set:

{% highlight clojure %}
(defn cross 
	"Given two genomes, the 'front' and the 'back' and a point of crossover, append the point number
  of genes from the front genome to the back genome with the point number of genes removed."
  [front-genome back-genome point]
  (concat (take point front-genome) (drop point back-genome)))

(defn crossover 
	"This is the basic one-point crossover operator for the Genetic Algorithm. It is but one alternative and other implentations are possible. For a 
 specified crossover-rate, either create children as the cross of the two parents or return the two parents unchanged. If actual crossover occurs,
 the 'children' will consist of two new genomes the first being the 'front' part of the dad and 'back' part of the mom and the second being
 the front part of the mom and back part of the dad. See the cross function."
  [dad-genome mom-genome crossover-rate]
  (if (< (rand) crossover-rate)
    (let [point (rand-int (count dad-genome))]
      [(cross dad-genome mom-genome point) (cross mom-genome dad-genome point)])
    [dad-genome mom-genome]))
{% endhighlight %}

We can now combine these two operators into a single breed function:

{% highlight clojure %}
(defn breed 
	"This function combines the crossover and mutation operators (given two parents) and creates new, fitness-evaluated individuals."
  [dad mom fitness mutation-rate crossover-rate]
  (let [[son dau] (crossover (:genome dad) (:genome mom) crossover-rate)
        son (mutate mutation-rate son)
        dau (mutate mutation-rate dau)]
    [{:genome son :fitness (fitness son)} {:genome dau :fitness (fitness dau)}]))
{% endhighlight %}

In order to breed two individuals, we have to select them. This implementation uses roulette wheel proportionate selection. The first function calculates the probability distribution based on the fitness values of the individuals in the population. The second function selects an individual (as an index) given that vector of probabilities. We can then use that index to pull the actual individual out of the population.

{% highlight clojure %}
(defn calculate-sampling-probabilities 
	"Roulette Wheel Selection requires that we calculate the size of the 'slots' in the roulette wheel to be proportionate to the
 relative fitness of each individual. If all individuals had the same fitness, then all individuals would have equal probabilities
 of being selected. With unequal fitness, an individual with a greater than average fitness will have a greater than average probability
 of being selected and an individual with a smaller than average fitness will have a smaller than average probability of being selected."
  [population]
  (let [total-fitness (reduce + (map :fitness population))]
   (mapv #(double (/ (:fitness %) total-fitness)) population)))

(defn roulette-wheel-selection 
	"This function implements roulette wheel selection. It is but one alternative and other implementations are possible. It takes a 
 precalculated probability distribution over the individuals as an argument. See calculate-sampling-probabilities.
 
 The basic idea is that individuals are selected with a probability proportionate to their relative fitness."
  [probabilities]
  (loop [mark (rand) sampled-index 0 probabilities probabilities]
    (if (empty? probabilities)
      sampled-index
      (let [current (first probabilities)]
        (if (< mark current)
         sampled-index
         (recur (- mark current) (inc sampled-index) (rest probabilities)))))))
{% endhighlight %}

We can bring all of the previous functions together to select two parents and, probabilistically, apply crossover and mutation to create two children:

{% highlight clojure %}
(defn pair-off-and-breed 
	"This is the main function for creating two (possibly) new individuals from two parents. It selects two parents at random from the population
 using roulette wheel selection and then applies one-point crossover with probability crossover-rate and single locus mutation with probability
 mutation-rate. The individuals are returned in a vector as a tuple with their genomes evaluated for fitness."
  [population probabilities fitness mutation-rate crossover-rate]
  (let [dad (population (roulette-wheel-selection probabilities))
        mom (population (roulette-wheel-selection probabilities))]
    (breed dad mom fitness mutation-rate crossover-rate)))
{% endhighlight %}

There isn't much more to do. We need to repeat this function n / 2 times to create the next generation of the genetic algorithm:

{% highlight clojure %}
(defn make-next-generation 
	"Creates the next generation for the genetic algorithm."
  [population n fitness mutation-rate crossover-rate]
  (let [probabilities (calculate-sampling-probabilities population)]
    (into [] (flatten (repeatedly (/ n 2) #(pair-off-and-breed population probabilities fitness mutation-rate crossover-rate))))))
{% endhighlight %}

As we previously noted, we aren't going to start off collecting much in the way of information or statistics, just the best individual of a generation:

{% highlight clojure %}
(defn compare-fitness 
	"Compares two individuals a and b using their :fitness values and returns the one with the larger fitness."
  [a b]
  (if (< (:fitness a) (:fitness b))
    b
    a))

(defn statistics 
	"Returns statistics about the population. Currently, it returns only the fittest individual."
  [population n]
  (reduce compare-fitness population))
{% endhighlight %}

We now get to the main function, `genetic-algorithm` that takes a problem and returns the best solution over the entire simulation:

{% highlight clojure %}
(defn genetic-algorithm 
	"Applies the genetic algorithm to the problem and returns the best individual generated. The problem is a map that contains the keys:
	 max-generations: the number of generations to run the GA simulation
	 n - the size of the population
	 k - the length of the individual genome
	 fitness - a function that takes a Vector representation of the genome, decodes it appropriately and as needed and returns a scalar fitness
	 	value where larger values are better
	 mutation-rate - the probability of a single locus mutation
	 crossover-rate - the probability that two parents will generate children that are some combination of their genomes."
  [problem]
  (let [{:keys [max-generations n k fitness mutation-rate crossover-rate]} problem
        initial-population (random-population n k fitness)
        solution (statistics initial-population n)
        _ (println "Initial best:" solution)]
   (loop [current-generation initial-population solution solution generation-no 0]
     (if (= generation-no max-generations)
       solution
       (let [next-generation (make-next-generation current-generation n fitness mutation-rate crossover-rate)
             candidate (statistics next-generation n)]
             (recur next-generation (compare-fitness solution candidate) (inc generation-no)))))))
{% endhighlight %}

For now, the function prints out the best of the initial population so that we can see that things are improving. We'll do something a bit better in the next blog post.

## Running the Algorithm

With the code implemented (and REPL tested), we can try it out. The easiest test is "max-ones". Each individual will be 100 bits and the fitness function will calculate the number of bits set to "1". We will run the simulation for 500 generations, the number of individuals (n) will be 100, the length of the genome (k) will be 100, the mutation-rate will be 0.05 and the crossover-rate will be 0.9. In the REPL:

{% highlight clojure %}
$ (def problem {
	:max-generations 500
	:n 100
	:k 100
	:fitness (fn [genes] (reduce + genes))
	:mutation-rate 0.05
	:crossover-rate 0.9
})
{% endhighlight %}

We can now call `genetic-algorithm` on our problem:

{% highlight clojure %}
$ (genetic-algorithm problem)
{% endhighlight %}

which yields:
{% highlight clojure %}
Initial best: {:genome [1 1 0 1 0 0 1 1 0 0 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0 1 0 1 0 0 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 0 0 0 0 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 1 1 0 1 0 1 1 0], :fitness 64}
=> {:genome (1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0), :fitness 86}
{% endhighlight %}

The best individual of the initial population has a fitness of 64 (64 bits set to 1). This is about what we should expect. The genome of each individual is a sequence of 100 bernoulli trials with a 50% probability of "success". The mean fitness should be around 50. Based on the central limit theorem, we can guess that fitness is approximately normally distributed around 50 and because we took the best individual, we're looking at the right hand side of the distribution.

When the algorithm is complete, the fitness is 86. You might be surprised that it's not 100. While we'll look at the issues in more detail in later blog posts, the population size is probably a bit too small relative to the genome size. Put a different way, the genome has $$2^{100}$$ possibilities and we started with about $$2^7$$ of them assuming no duplications. It's pretty amazing that this algorithm, fueled by three simple operators (selection, crossover and mutation), improved the fitness of the best individual in the population by 22 points or 34%.

Note: you may get a different answer since this is a stochastic algorithm.

## Future Posts

The genetic algorithm is a pretty amazing technique that simulates an abstract model of evolution in order to solve optimization problems. What's even better is that it has practical applications. The genetic algorithm has been used in a [lot of different applications](http://en.wikipedia.org/wiki/List_of_genetic_algorithm_applications) but one of the more interesting has been the evolution of an antenna for NASA's [Space Technology 5 Mission](http://idesign.ucsc.edu/projects/evo_antenna.html).

Future blog posts will continue the discussion of this interesting topic. We'll look at it from a variety directions: setting parameters, formulating problems and fitness functions, monitoring, testing, and different implemenations of the basic operators. We will also see how the code becomes a scaffold for applying different kinds of evolutionary computation such as evolutionary programming (by using finite state machines) or genetic programming (by using abstract syntax trees).

Cheers!