# Deep Learning Course Notes

## Week 1
### Introduction
* AI - impact of AI analagous to impact of Electricity 
* Specialization outline
    * Basic Deep Learning
    * Hyperparameters tuning, regularization, optimization
    * Building a project
    * CNN
    * RNN
    * Shallow learning vs Deep learning

### Neural networks intuition
* E.g. housing price prediction
* Consider each node as a ReLU (Rectified Linear Unit)..can sometimes use non-linear function too, depending on the problem in hand
![Housing Price Prediction](images/neural-networks-intuition.png)
* Applications in supervised learning

### Supervise Learning with NN
* Lots of famous applications
* Using standard neural networks
    * Real Estate (house price prediction)
    * Online ads based on user info - predict click
* Using Convolution Neural Network (CNN) 
    * Image recognition
* Using Recurrent Neural Network (RNN)
    * Speech recognition
    * Language translation
* Hybrid
    * autonomous driving - decide direction based on the image, position of other cars on road - custom NN
* Supervised learning
    * Structured Data
    * Unstructured Data
        * Audio
        * Image
    * Short term value creation of NN - much in the structured data. although it is immensely useful in unstructured data as well

### Why deep learning is taking off?
* Data availability vs performance graph
![Why deep learning is taking off](images/why-deep-learning-is-taking-off.png)
* With smaller data sets, performance of NN doesn't necessarily increase as we scale up the size of NN.
* Algorithmic innovation drove innovation and efficiency in computation
* sigmoid function vs ReLU
    * Slow learning rate (gradient is nearly 0) on either ends of the sigmoid curve
    * ReLU - better learning rate
* Building a neural network is more of a iterative approach - Idea --> Code --> Experiment --> Idea --> Code...

### Interview with [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton)
* Inspired from brain science.. background in Physiology and Physics.. then some psycology..then carpentry.. then AI
* Co-invented back propagation algorithm
* Graph representation <--> Feature vector
* Boltzman machine - Geoffrey considers this as his best - Restrcited Boltzman machine 
* Deep Belief Nets
* ReLU - somewhat equivalent to stacked logistic units
* Backprop and Brain - Geoffrey is working on a paper..is brain really using backprop? referred to brain plasticity in connection to how brain learns
* Some more advanced concepts that stick to my brain :-(
* He works with Google Brain team too
* Discriminative learning - refers to supervised learning. In initial days, lot of focus on unsupervised learning as that is how brain is believed to learn. Recent advancements in deep learning with supervised learning, especially in the last ten years or so really pushed the focus in a different direction. unsupervised learning is still largely unsolved problem. 
* Advice to people who wants to get into deep learning - trust your intuitions 
* Google Brain residents? 
* Geoffrey did the first MOOC on deep learning on Coursera
* Thoughts - as symbolic expression vs vectorized ops
* Haven't heard about him before this talk. After reading few articles, I realized why he is rightly referred as Godfather of AI/Deep Learning. Respect!