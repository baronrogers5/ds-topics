### The problem with tanh

If a kernel wanted to be at 0.75, tanh will push the value to 1, which causes information loss.
The values of the kernels are pushed far apart, which makes the network not converge.

### Backpropagating Losses in a weird layman understanding

Suppose conceptually a GPU can house a batch size of 16.
Then, conceptually we can have 16 such models with each of the 16 images placed on a model.
The output losses are calculated for each model using categorical-crossentropy for classification.
```python 
categorical-crossentropy = [-gt*np.log(pv) for gt,pv in zip(ground_truths, predicted_val)]
# actually categorical crossentropy is: softmax + crossentropy loss

categorical-crossentropy = -np.log(predicted_val[np.where(ground_truths == 1)[0]])
```
 
Then the losses are averaged.

Loss func = f(all kernel values, ...)
For a kernel value x3, if we change x3 slightly to x3', how does our loss change ?
If delta(x3) is positive and the loss increases, it means we are going up the cliff and we need to come down.
If delta(x3) is negative and the loss reduces, it means we need to keep reducing delta(x3)

### Signmoid Loss function

Sigmoid loss function sqishes the values between to 0 or 1. So it is a positive facing act. func.
Formula - f(x) = 1 / (1 + np.exp(-x))

### Small Learning Rate reasons

Well two important ones:
1. The partial derivative loss calculated against a parameter/weight is still large enough that it cannot
be directly used to update the weight. eg. The weights are generally initialized from a normal distribution 
with a mean = 0 and stddev = 1. But the loss in the first iteration may be close to 2-3 (even higher).
2. To perform partial derivative we consider every other weight constant, wrt loss, but this is only possible
if we update the weight with a very small number that does not change the intricate relationship it has with all the 
other weights.
  

### Food for thoughts

Run an Exp: to identify the fact that a trained model has all its weights as a normal distribution and an untrained model
initialized from a normal distribution does not infact have the same.

Loss funtion shape changes after every iteration, because the weights change, and the loss fn (weights, batchnorm params, dropout, 
weight decay, relu)

Saddle point and plateau regions where theta is 0, can be solved using momentum.

The momentum formula is the exact formula for calculating exponentially moving weighted averages.
 
### Weight Initialization reasons

The Kaiming He policy for relu networks: 
is to keep the variance of a layer = 2 / n
```python
w[l] = np.random.randn(shape) * np.sqrt(2 / n[l - 1])
```
To prevent vanishing and exploding gradients. If our weight inititializations are very similar to identity matrices.
Suppose, they are just slightly above 1, say,
```python
w[1] = np.array([1.5, 0], [0, 1.5])
```

and if our activation is a linear funtion where, g(z) = a = z.
Then our prefinal layer would have a weight of w^(l-1).
and since w is 1.5 the values will explode.

Similarly, if w is slightly less than 1, then, the gradients will exponentially vanish.

### Gradient Descent with Momentum

It almost always works faster than the normal gradient descent. The idea is to compute an exponentially weighted 
moving average.
If we use a very high learning rate with SGD, there is a chance that we may diverge, so we stick with a smaller value of lr.
Derivative imparts acceleration and the momentum term imparts velocity.
What I mean by this is, Momentum helps take less vertically diverging steps and faster steps in the horizontal direction,
which is where our velocity is right now, but the gradient update knows ex actly which direction to go, which may end up being 
opposite to the momentum velocity, and thus it accelerates in the opposite direction, the result may take some time
before the momentum velocity reduces and starts moving in the opposite direction.

```python
momentum = beta*old_momentum + (1 - beta)*grad_update
new_weight = old_weight - alpha*momentum
```
There are now two hyperparameters, alpha and beta, the most common value of beta = 0.9

> Momentum is an approx average over 1 / (1 - beta) previous values, so with a very high value of 
beta, we average over a larger number of points and get smoother representations (not always desired)  


### One Cycle Policy

#### Range Test

The idea is to start with a small LR, which we may not even start actual training with, like (1e-4, or 1e-5),
and increase the lr after each mini-batch until the loss starts exploding.

#### LR Warm up and Annealing strategies

These srategies are used in the one-cycle policy.
When our weights are randomly initialized, we want to give the network time to knudge them in the correct direction.
But, when we have them set, we want to increase the lr a lot to our desired max.
Then once we have fixed the initial few values of our acc, we want to anneal our lr (slow down), so that the network 
can take smaller steps, to reach the desired local minima.

#### Grid Searching Parameters

The max-lr can be found using the range test.
But to find the min-lr which can be 1/5th or 1/10th of the max-lr, we need to perform a grid-search.

To find it start from 1/5th - 1/10th to the max-lr, and count for each case, which min-lr required the least number of iterations
to reach the max-lr. That becomes the min-lr.

#### Cyclic LR Training

The total number of iterations for an epoch is suppose 80.
For the first 30 epochs we go from the min-lr to the max-lr.
Then in the next 40 epochs we go from max-lr, back to min-lr.
The remaining 10 epochs we drop down to even further lower lrs. 


