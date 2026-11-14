# My Write-up for the assignment 5 - alignment

## Problem (prompting_baseline)

### (a)
Results are below:

    RESULTS OF ONLY QUESTION PROMPTING
    Total questions: 1319
    Fully correct: 0
    Only formatting correct: 219
    Incorrect: 1100
    RESULTS OF ZERO-SHOT PROMPTING
    Total questions: 1319
    Fully correct: 0
    Only formatting correct: 792
    Incorrect: 527
    RESULTS OF FEW-SHOT PROMPTING
    Total questions: 1319
    Fully correct: 217
    Only formatting correct: 1046
    Incorrect: 56

It's very surprising few-shot prompting helps this much in terms of correctness, when I viewed the answer that were formatted correctly but answer reward was 0, I saw some interesting pattern of repeated thinking patterns from the model that are related to math problems but not to that problem at all. 

### (b)

Apparently (maybe this is not the case for bigger models) we need to show some examples to the model for the model to understand the idea and use it in new examples as shown on the results above. Few-shot and zero-shot prompting definitely improved the formatting in my opinion and allowed the model to also reason that may have helped it to solve some problems it wasn't able to solve with just answering in a few tokens. It may have also allowed to model to remember it's Math data inside it's pretraining set by showing these examples and increasing familarity to reduce randomness and gibberish that I saw in question only prompting.
