# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

Notes from course-2 of the [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning) in Coursera

- [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](#improving-deep-neural-networks-hyperparameter-tuning-regularization-and-optimization)
  - [Week 1 - Setting up your ML application](#week-1---setting-up-your-ml-application)
    - [Choosing train/dev/test sets](#choosing-traindevtest-sets)
    - [Bias and Variance](#bias-and-variance)
    - [Basic recipe for Machine Learning](#basic-recipe-for-machine-learning)
    - [Regularization](#regularization)
    - [Why regularization reduces overfitting?](#why-regularization-reduces-overfitting)
  - [References](#references)

## Week 1 - Setting up your ML application

### Choosing train/dev/test sets

- Applied ML is highly iterative..remember idea==>code==>experiment cycle.
- Many different applications..but the intuition from one application may not transfer to other application due to differences in data
- Trends in choosing the train/dev/test datasets. dev set is nothing but cross-validation set.
- Typical split used to be: 60-20-20
- For some big data applications, the ratio would be something in the order of 98:1:1 or even lesser.
- Mismatched train/test distributions
  - E.g cat classifier
  - training set - cat pictures from webpages
  - dev/test set - cat pictures from user uploaded images
  - make sure that dev/test set come from the same distribution
- Sometimes it might be okay to not have a test set. (some people may call train/dev set as train/test set.) Why test set is needed? - to eliminate bias and avoid overfitting to the training set.

### Bias and Variance

- Difficulties in plotting decision boundary in higher dimensions
- Find the error in train set and dev set. If the error on dev set is significantly higher than the train set, that signals some level of overfitting to the train set. High Variance.
- If the training set error itself is high, then classifier is not fitting the train set properly. Underfitting. High Bias.
- Some examples of diff train and dev set errors and their meaning.

| Train set error | dev set error | Bias | Variance | Fitting    |
| --------------- | ------------- | ---- | -------- | ---------- |
| 1 %             | 11%           | Low  | High     | over       |
| 15%             | 16%           | High | Low      | under      |
| 15%             | 30%           | High | High     | wrong      |
| 0.5%            | 1%            | Low  | Low      | just right |

- Assumption: human error is ~0%
- Optional (Bayes) error
- Example of high bias and high variance - mix of underfitting and overfitting
- ![High Bias High Variance](images/high-bias-high-variance.png)

### Basic recipe for Machine Learning

- Systemic way to tune the performance
- After training the algorithm, ask some questions
  - High bias(on train set)? - retry using bigger network, more iterations, or different NN architecture
  - High variance(on dev set)? - retry with more data, regularization, or different architecture
- Bias-Variance tradeoff - in the previous days, we used to tradeoff one for other. Regularization may some times add some bias when trying to reduce the variance.

### Regularization

- To prevent overfitting by reducing variance
- Objective of logistic regression: minimize cost function $J(w, b) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$
- Add the regularization factor to the cost ($\lambda$ - is the regularization parameter)
- $J(w, b) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \parallel w\parallel^2$
- "Frobenius norm" - sum of square of elements of a matrix (not confuse with L2 norm in linear algebra)
- Why only $W$ is regularized, but not $b$? $W$ is often a multi-dimension matrix, whereas $b$ is of smaller dimension vector. Experiements show that regularizing the bias vector adds only very little.
- Types
  - L1 regularization, w will end up sparse, this type not used often.
  - L2 regularization - commonly used.
- $dw = dw + \frac {\lambda}{m}w^{[l]}$, regularize $\partial w$ during backpropagation
- Aka weight decay - why? $W$ is eventually multiplied by $(1 - \frac{L\lambda}{m})$ when we update $W$ after backpropagation.

### Why regularization reduces overfitting?

- Trying to explain in my own words. Both the cost function and the weights are penalized by the regularization factor. High value of $\lambda$ will bring $W$ closer to 0, because of weight decay (remember $W$ gets multiplied by $(1 - \frac{L\lambda}{m})$ in the update phase). Neurons with near-zero weights in the hidden units have very little to no effect on the result. That makes the pipeline more linear and thus reduces the complex non-linear fit in the decision boundary curve.
- ![Why-regularization-reduces-overfitting](images/why-regularization-reduces-overfitting.png)
- More intuition..this time using $\tanh$ function. If weights are low, then $Z$ will also be significantly low, and ends up linear as seen in the $\tanh$ curve.
- ![regularize-tanh](images/reduce-over-fitting-in-tanh.png)

## References

- [Enabling private emails in github commits](https://stackoverflow.com/questions/43378060/meaning-of-the-github-message-push-declined-due-to-email-privacy-restrictions)