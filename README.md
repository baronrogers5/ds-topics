## General Stuff

- What is a neural network ?
- Binary Classification
- Logistic Regression
- Linear Regression
- Logistic Regression Cost Function
- Derivatives
- Broadcasting
- Computing a nn's output
- train / dev /test splits

## Activation Functions

- RELU

  > The most commonly used activation function. It's high use is facilitated by it's simplicity.
  >
  >  A negative number has its MSB as 1. So, to perform RELU, only the MSB needs to be checked, which is super-fast. 
  >
  > ```python
  > relu = lambda x: x if x >= 0 else 0
  > 
  > # torch implementation
  > relu = lambda x: torch.clamp(x, min=0)
  > ```

- Sigmoid

- tanh

- Leaky RELU

## Backpropagation

- Internal Covariance Shift
- Why does it actually work ?
- RELU>BN>CONV or CONV>RELU>BN (The range of values left)

## Intialization 

- Kaiming He
- Xavier / Glorot
- Forward Propagation
- Bias/ Variance

## Regularization

- Dropout
- Data Augmentation
- Slightly due to BatchNormalization
  - Why does it work ?
- Higher learning rates
- weight decay
- Normalizing Inputs
- Vanishing / Exploding Gradients

## Optimizers

- Gradient Descent
- Gradient Descent with Momentum
- Gradient Descent with Nesterov
- Exponentially weighted moving averages
- Bias correction in exponentially weighed moving averages
- Adagrad
- RMSProp
- Adam
- *Saddle point, Gaussian Noise*

## HyperParameter Tuning

- Learning Rate Decay
- Appropriate scale for hyperparameters

## SuperConvergence

- One Cycle Policy
- Cyclic LR

## Loss Functions

- Softmax
- CrossEntropyLoss
- BCEWithLogits
- Categorical Cross Entropy

## Inference and Corrections

- GradCAM
- Heatmaps
- Cleaning up incorrectly labelled data
- Addressing data mismatch
- Transfer learning
- Multi-task learning (Learning with different heads )
- e2e deep learning
- Dense Layers

## Convolutional Neural Networks

- Padding
- Receptive Field
- Checkerboard Issue
- Feature Loss
- Channels
- Kernels / Filters
- Features
- Parameters / Weights

- **Types of Convolutions**
  - Strided Convolutions
  - Depth wise Separable convolutions
  - spatially separable convolutions
  - group convolutions
  - Atrous Convolutions / Dilated Convolutions
  - Deconvolutions  / transpose convolutions / fractionally strided
  - roots concept in convolutions
  - 1x1 Convolutions / Pointwise Convolutions
  - Why only 3x3 convolutions ?
- **Data Augmentation**
  - ImageDataGenerator
  - imgaug augmenters
  - cutout
- **Pooling** 
  - MaxPooling
  - AveragePooling
  - Global Average Pooling
  - Global Max Pooling
- **Architectures**
  - Renets (Skip Connection)
  - Inception
  - YOLO
  - DenseNets
  - UNets
  - Recurrent Neural Networks 
  - LSTM
- **Object Detection**
  - Object Localization
  - Landmark detection
  - Sliding Windows
  - BBox Predictions
  - IOU
  - Non-max Suppression
  - Anchor Boxes
  - Region Proposals
- **Face Recognition**
  - One Shot Learning
  - Siamese Network
  - Triplet Loss
  - Face Verification and Binary Classification
- **Neural Style Transfer**
  - What are deep convnets learning?
  - Cost Function
  - Content Cost Function
  - Style Cost function
  - 1D and 3D Generalizations
- **Generative Adversarial Networks**
  - UNets
  - Pre-training the encoder
  -  The middle convs
  - The cross connections
  - Passing of inputs as a feature to final prediction

