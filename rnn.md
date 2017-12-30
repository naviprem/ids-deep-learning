# Recurrent Neural Networks

Humans don’t start their thinking from scratch every second. As you read this essay, you understand 
each word based on your understanding of previous words. You don’t throw everything away and start 
thinking from scratch again. Your thoughts have persistence.

Traditional neural networks can’t do this, and it seems like a major shortcoming. For example, 
imagine you want to classify what kind of event is happening at every point in a movie. It’s unclear 
how a traditional neural network could use its reasoning about previous events in the film to inform 
later ones.

Recurrent neural networks address this issue. They are networks with loops in them, allowing 
information to persist.


# The Problem of Long-Term Dependencies

One of the appeals of RNNs is the idea that they might be able to connect previous information to 
the present task, such as using previous video frames might inform the understanding of the present 
frame. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. For example, 
consider a language model trying to predict the next word based on the previous ones. If we are 
trying to predict the last word in “the clouds are in the sky,” we don’t need any further context – 
it’s pretty obvious the next word is going to be sky. In such cases, where the gap between the 
relevant information and the place that it’s needed is small, RNNs can learn to use the past 
information.

But there are also cases where we need more context. Consider trying to predict the last word in 
the text “I grew up in France… I speak fluent French.” Recent information suggests that the next 
word is probably the name of a language, but if we want to narrow down which language, we need the 
context of France, from further back. It’s entirely possible for the gap between the relevant 
information and the point where it is needed to become very large.

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

Thankfully, LSTMs don’t have this problem!



## Reference:

### Understanding LSTM Networks

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


### The Unreasonable Effectiveness of Recurrent Neural Networks

[http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)


### How to Visualize Your Recurrent Neural Network with Attention in Keras

[https://medium.com/datalogue/attention-in-keras-1892773a4f22](https://medium.com/datalogue/attention-in-keras-1892773a4f22)
