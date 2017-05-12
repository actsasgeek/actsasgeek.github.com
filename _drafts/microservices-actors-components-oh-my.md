---
layout: post
title: "Microservices, Actors, Components. Oh, my"
date: 2017-05-05 00:00
tags: musings
---

> "It is better to have 100 functions operate on one data structure than 10 functions on 10 data structures." —Alan Perlis

1. Alan Perlis's maxim--oft quoted in the Clojure community--is not a sufficient foundation for good software engineering in a functional programming language. By software engineering, I mean things like larger patterns of code organization, coupling/decoupling or communication patterns, testing, etc.

> "Remember, it doesn't take an awful lot of skill to write a program that a computer can understand. The skill is in writing programs that humans understand." --'Uncle' Bob Martin

2. Without these larger principles, a codebase is difficult to understand, modify and deploy.

> "An accounting application should not look like a Rails app or a Django app but like an accounting application." -- 'Uncle' Bob Martin

3. I have never seen a Rails app that didn't look exactly the same. Most web application based frameworks generate identical directory structures making projects completely indistinguishable from each other.

4. Microservices have arisen in antithesis to the Monolith but I don't think we've decomposed the pros and cons of microservices sufficiently to understand how we might get different mixes of pros and cons without actually deploying microservices.

5. I believe one of the biggest pros in microservices is code decoupling and this can be achieved without implementing microservices but by using Actors.
