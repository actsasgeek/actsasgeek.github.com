---
layout: post
title:  "Ecological Bin Packing"
date:   2015-03-10 20:28
tags: general, algorithms, dynamic programming
---
The name of this [Programming Challenge](http://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=38) is a double entendre.

```
1  3  4
3  9  2
8  4  2
```

Each row represents a glass color (for example, bottles made of brown, green, or clear glass) and each column represents a bin. This implies the first bin has 1 brown bottle, 3 green bottles and 8 clear bottles. The goal is to move all the bottles so that each bin contains only one color of glass but only by moving the minimum number of bottles.

## A Greedy Approach

Looking first a brown bottles, one option would be to move all the bottles to the first bin. This would require moving 3 bottles from bin 2 and 4 bottles from bin 3 or 7 bottles total. On the other hand, if we moved all the bottles to bin 2, we would move the 1 bottle from bin 1 and 4 bottles from bin 3 or 5 bottles total. Our final option would be to move the 1 bottle from bin 1 and 3 bottles from bin 2 to bin 3 which would be or 4 bottles total.

To see all the possibilities we can transform our table of bottle counts, to a table of bottle moves by replacing each bin count with the number of bottles that would need to be moved to the bin instead. We already did this for the first row and this is what it looks like for the current problem:

```
7   5   4
11  5   12
6   10  12
```

Looking at the this table, we can see that the minimum number of moves is accomplished when we move all the bottles in the 1st row to the 3rd bin, all the bottles in the 2nd row to the 2nd bin, and all the bottles in the 3rd row to the 1st bin. This leaves 15 moves total. Can we generalize this into an algorithm that works for all kinds of counts? For more than 3 rows?

A little bit of thought reveals a flaw: what if all three colors have a minimum at the same bin? We can't move all the bottles to the same bin because they need to be in different bins. Maybe we can salvage our [greedy algorithm](http://en.wikipedia.org/wiki/Greedy_algorithm) by making that a special case? Let's see.

If more than one glass color did have the same bin number for its largest count, we would be unable to move both colors to the same bin. Consider this setup:

```
5   7   9
2   6   7
3   4   5
```

If we replace the number of bottles with the number of moves, we have:

```
16  14  12
13  9   8
9   8   7
```

which shows that we definitely want to move each color to bin 3. So how do we break the tie? We look at the bottle with the fewest moves which would be the clear glass (row 3). If we move all of them to bin 3, that's 7 moves total. Continuing with our greedy approach, now that bin 3 is out, it looks like bin 2 is the best for the green glass (row 2). All that is left for the brown glass is bin 1. Our total number of moves is 16 + 9 + 7 = 32. Is that the best we can do? Let's look at all the possibilities:


16 + 9 + 7 = 32

16 + 8 + 8 = 32

14 + 13 + 7 = 34

14 + 8 + 9 = 31

12 + 9 + 9 = 30

12 + 13 + 8 = 33

Turns out that the following has fewer total moves:

```
-    -    12
-    9    -
9    -    -
```

which suggests that the greedy approach is bound to fail. Now, the Programming Challenge only asks us to solve problems involving three bins and three bottle colors, which means it would be trivial to hard code the equations above in terms of matrix indices and return smallest value and the bins used in the solution but what if we had more colors of glass? And what if we add newspaper, cardboard, cans and milk cartons to the problem? Our six possibilities with 3 rows would increase exponentially. 2 colors has 2 candidate solutions to check. 3 colors has 3 x 2 = 6 candidates. 4 colors is 4 x 6 = 24 candidates. 5 colors is 5 x 24 = 120 candidates...not good and believe it or not, we simplified.

## Decomposition of the Problem

Still there might be a generalizable approach. Looking at our matrix of moves again, if we isolate the bins for row 1, then there are two possibilities for rows 2 and 3.

```
16  -    -   |  -    14   -   |  -    -    12
-   9    8   |  13   -    8   |  13   9    -
-   8    7   |  9    -    7   |  9    8    -
```

Ultimately, the bin to pick for row 1 depends on the number of bottle moves there plus the sum of the main diagonal or the sum of the off diagonal of the sub problem formed by the 2x2 matrix of rows 2 and 3. We can easily and always solve such problems as the minimum of either row 1, column 1 plus row 2, column 2 or the sum of row 1, column 2 and row 2, column 1. It would appear that this is a base case.

Using this base case, can we use this same approach with a set of four bottle colors and bins? We start with the following problem:

```
6  8  3  2
9  1  8  8
3  7  2  5
3  4  7  1
```

and transform it into move counts:

```
13  11  16  17
17  25  18  18
14  10  15  12
12  11  8   14
```

This matrix can be decomposed into four 3 bin problems:

```
13  -   -   -  | -   11  -   -  | -   -   16  -  | -   -   -   17
-   25  18  18 | 17  -   18  18 | 17  25  -   18 | 17  25  18  -
-   10  15  12 | 14  -   15  12 | 14  10  -   12 | 14  10  15  -  
-   11  8   14 | 12  -   8   14 | 12  11  -   14 | 12  11  8   -
```

And we already showed above how a three bin problem can be decomposed into three two bin problems. So it looks like this approach would work.

## Dynamic Programming

As it turns out, this is an example of [Dynamic Programming](http://en.wikipedia.org/wiki/Dynamic_programming). The use of "programming" in this context refers to the word's older meaning as "optimization". It is especially useful when greedy solutions will not work. Other examples of such problems include efficient multiplication of long chains of matrices and laying out text on a page. All such solutions generally have three main components: a base case that can be solved directly, an inductive step that is often recursive and memoization. The last component will become apparent later but you can easily see that it is not enough to calculate the fewest number of moves...you must return the bin numbers that accomplish it. This will generally require some kind of record keeping of candidate solutions as they are built up. This record keeping is called memoization.

## Implementation

Now that we've figured it all out, let's implement the algorithm in Clojure.

The firt function we'll need is one that converts from number of bottles to number of moves:

{% highlight clojure %}
(defn bottles-to-moves
  [bottle-matrix]
  (let [convert-row (fn [row] (let [total (apply + row)
                                    subtract-from-total (fn [x] (- total x))]
                                (mapv subtract-from-total row)))]
    (mapv convert-row bottle-matrix)))
{% endhighlight %}

We can also use a function that simplies access to our matrix which is a Vector of Vectors:

{% highlight clojure %}
(defn access [matrix r c]
  ((matrix r) c))
{% endhighlight %}

As we previously noted, it's easy to write a hard-coded solution for the case of 2 bins and 2 colors of glass:

{% highlight clojure %}
(defn tally-moves-for-2-bins
  [moves-matrix bottle-indices bin-indices]
  (let [main-sum       (+ (access moves-matrix (bottle-indices 0) (bin-indices 0))
                          (access moves-matrix (bottle-indices 1) (bin-indices 1)))
        off-sum        (+ (access moves-matrix (bottle-indices 0) (bin-indices 1))
                          (access moves-matrix (bottle-indices 1) (bin-indices 0)))
        switch-indices (fn [indices] (into [] (concat (drop 1 indices) (take 1 indices))))]
    [[bin-indices main-sum] [(switch-indices bin-indices) off-sum]]))
{% endhighlight %}

Notice, however, the memoization in the last line.

For efficiency, we'll always use the full matrix but we'll work on a subset of it as described by a vector of indices. When we concentrate on a particular row and column of the NxN move matrix, we extract the number of moves and use the other rows and columns to construct the different (N-1)x(N-1) sub-problems. We need a function that generates those sets of sub-indices.

{% highlight clojure %}
(defn calculate-subindices
  "Given a list of n indices, calculate all n-1 subindices"
  [indices]
    (let [drop-nth (fn [xs n] (into [] (concat (subvec xs 0 n) (subvec xs (inc n) (count xs)))))]
      (mapv #(drop-nth indices %) (range 0 (count indices)))))
{% endhighlight %}

We know have almost everything we need. We start with bottle index 0 which generates N possible subproblems (one for each bin). We recurse on that problem and generate (N-1) sub-subproblems from bottle 1. Recursing on those sub-subproblems, creates (N-2) sub-sub-subproblems...and so on until we reach the case of 2x2. We write a function that calculates moves for N bins and a function to call it that calculates move totals:


{% highlight clojure %}
(declare calculate-move-totals)

(defn tally-moves-for-n-bins* [moves-matrix bottle-indices current-bin other-bin-indices]
  (let [current-bottle                (bottle-indices 0)
        next-bottles                  (subvec bottle-indices 1)
        current-total                 (access moves-matrix current-bottle current-bin)
        intermediate-results          (calculate-move-totals moves-matrix next-bottles other-bin-indices)
        tally-intermediate-results    (fn [[indices total]] [(concat [current-bin] indices) (+ current-total total)])]
    (mapv tally-intermediate-results intermediate-results)))

(defn calculate-move-totals
  [moves-matrix bottle-indices bin-indices]
  (if (= (count bin-indices) 2)
    (tally-moves-for-2-bins moves-matrix bottle-indices bin-indices)
    (let [bin-indices-subsets            (calculate-subindices bin-indices)
          tally-moves-for-n-bins         (partial tally-moves-for-n-bins* moves-matrix bottle-indices)]
        (mapcat tally-moves-for-n-bins bin-indices bin-indices-subsets))))
{% endhighlight %}

Finally, we wrap the whole thing up into an `ecological-bin-packing` function that returns all the possible solutions with the best one first...afterall, the best solution may mean moving a bottle containing questionnable contents...best to leave it in its bin.

{% highlight clojure %}
(defn ecological-bin-packing [bottles-matrix]
   (let [bin-count   (count bottles-matrix)
        indices      (into [] (range 0 bin-count))
        moves-matrix (bottles-to-moves bottles-matrix)
        tallies      (calculate-move-totals moves-matrix indices indices)]
    (sort-by second tallies)))
{% endhighlight %}

Here are a few results. For the first 2x2 problem, it doesn't matter how you sort the bottles:

```
user=> (ecological-bin-packing [[1 2] [3 4]])
([[0 1] 5] [[1 0] 5])
```

Here's one of the 3x3 problems from above:

```
user=> (ecological-bin-packing [[5 7 9] [2 6 7] [3 4 5]])
([(2 1 0) 30] [(1 2 0) 31] [(0 1 2) 32] [(0 2 1) 32] [(2 0 1) 33] [(1 0 2) 34])
```

Finally, here's a 4x4 problem:

```
user=> (ecological-bin-packing [[6 8 3 2] [9 1 8 8] [3 7 2 5] [3 4 7 1]])
([(1 0 3 2) 48] [(0 3 1 2) 49] [(1 3 0 2) 51] [(3 0 1 2) 52] [(1 2 3 0) 53] [(0 2 3 1) 54] [(0 2 1 3) 55] [(1 3 2 0) 56] [(2 0 3 1) 56] [(2 3 1 0) 56] [(0 3 2 1) 57] [(1 0 2 3) 57] [(1 2 0 3) 57] [(2 0 1 3) 57] [(3 2 1 0) 57] [(0 1 3 2) 58] [(2 3 0 1) 59] [(3 0 2 1) 60] [(3 2 0 1) 60] [(3 1 0 2) 64] [(2 1 3 0) 65] [(0 1 2 3) 67] [(2 1 0 3) 69] [(3 1 2 0) 69])
```

Just for fun, here's a solution in Python as well. I changed some of the names of functions/arguments between languages.

{% highlight python %}
def translate_bottles_to_moves( bottle_matrix):
  result = []
  for row in bottle_matrix:
    total_bottles = sum( row)
    moves = [total_bottles - n for n in row]
    result.append( moves)
  return result

def solve_2_by_2( full_matrix, row_indices, column_indices):
  main_diagonal_sum = full_matrix[ row_indices[ 0]][column_indices[ 0]] + full_matrix[ row_indices[ 1]][column_indices[ 1]]
  off_diagonal_sum = full_matrix[ row_indices[ 0]][column_indices[ 1]] + full_matrix[ row_indices[ 1]][column_indices[ 0]]
  return [(column_indices, main_diagonal_sum), (column_indices[1:2] + column_indices[ 0:1], off_diagonal_sum)]

def calculate_subindices( indices):
  results = []
  for i in xrange( len( indices)):
    removed = indices[0:i] + indices[i+1:]
    results.append( removed)
  return results

def calculate_total_moves( full_matrix, row_indices, column_indices):
  if len( row_indices) == 2:
    return solve_2_by_2( full_matrix, row_indices, column_indices)

  current_row = row_indices[ 0]
  next_rows = row_indices[ 1:]

  column_indices_subsets = calculate_subindices( column_indices)

  results = []
  for current_column, other_column_indices in zip( column_indices, column_indices_subsets):
    current_total = full_matrix[ current_row][ current_column]
    intermediate_results = calculate_total_moves( full_matrix, next_rows, other_column_indices)
    current_results = [([current_column] + indices, total + current_total) for indices, total in intermediate_results]
    for result in current_results:
      results.append( result)
  return results

def ecological_bin_packing( matrix):
  bins = len( matrix)
  moves = translate_bottles_to_moves( matrix)
  all_results = calculate_total_moves( moves, range( 0, bins), range( 0, bins))
  best_to_worst = sorted( all_results, key=lambda x: x[ 1])
  return best_to_worst
{% endhighlight %}

Cheers.
