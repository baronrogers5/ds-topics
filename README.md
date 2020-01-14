## General Stuff

- What is a neural network ?

  ```python
  """
  A neural network is a universal function approximator. 
  It is a non-linear function that tries to map the inputs to the outputs.
  It works on the principle that given enough non-linearity, any input can be approximated to it's output.
  """
  
  y = f(x) # f -> neural network
  ```

- Binary Classification

- Logistic Regression

- Linear Regression

- Logistic Regression Cost Function

- Derivatives

- Broadcasting

  ```python
  """
  Broadcasting is the method by which any scalar, vector or matrix, tensor of a lower dimension, can be expanded to have the dimension of a higher rank tensor. 
  Broadcasting is carried on unit axes.
  """
  
  # Broadcasting with a scalar
  m = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
  
  a = torch.tensor([1., 2., 3.])
  a.expand_as(m)
  
  > tensor([[1., 2., 3.],
            [1., 2., 3.],
            [1., 2., 3.]])
  
  # Adding a new axis at the end
  a[:, None] * 2
  
  > tensor([[2.],
            [4.],
            [6.]])
  
  # Expand 2 to have the same shape as m
  torch.tensor(2).expand_as(m)
  
  > tensor([[2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]])
  ```
- Computing a nn's output

- train / dev /test splits

## Activation Functions

- RELU

  ```python
  """
  The most commonly used activation function. It's high use is facilitated by it's simplicity.
  A negative number has its MSB as 1. So, to perform RELU, only the MSB needs to be checked, which is super-fast. 
  """"
  
  relu = lambda x: x if x >= 0 else 0
  
  # torch implementation
  relu = lambda x: torch.clamp(x, min=0)
  ```

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
  
    ```python
    """
    MaxPool reduces the size of an image, by taking the maximum valued pixel in each convolution.
    MaxPooling or stride 2 convs can be used for reducing the size of an image.
    Generally we use a (2x2) kernel with a stride of 2.
    This reduces the size of an image by half, i.e. (nxn) -Maxpool-> (n//2 x n//2)
    If we want our networks to train faster, we go for stride 2 convs as the number of 
    times each pixel in the central area undergoes computation is less than MaxPooling.
    That being said, maxpool certainly provides better results.
    """
    
    # keras
    x = tf.keras.layers.MaxPooling2D()(x)
    # torch
    x = torch.nn.MaxPool2d((2, 2), stride=2)(x)
    ```
  
    
  
  - AveragePooling
  
    ```python
    """
    Similar to MAxPooling but instead of taking the maximum pixel value, an average of
    all pixels is taken.
    """
    
    # keras
    x = tf.keras.layers.AveragePooling2D()(x)
    # torch
    x = torch.nn.AvgPool2d((2, 2), stride=2)(x)
    ```
  
    
  
  - Global Average Pooling
  
    ```python
    """
    Consider a (batch, h: 7, w: 7, channels: 512) tensor. Applying GAP on the tensor, calcuates the mean of all (7x7) pixels along the channel axis.
    The resulting tensor has a shape (batch, channels: 512).
    This can be followed by dense layers, which reduce the nodes to the categories required for prediction.
    GAP is powerful as it helps ascertain the most important portions of the image per channel that is responsible for making the prediction.
    """
    
    # keras 
    tf.keras.layers.GlobalAveragePooling2D()(x)
    # torch
    x = torch.nn.AdaptiveAvgPool2d(1)(x)
    x = torch.nn.Flatten()(x)
    ```
  
    
  
  - Global Max Pooling
  
    ```python
    """
    Similar to GAP, only difference is that insted of mean, max value of feature maps are chosen.
    """
    
    # keras
    tf.keras.layers.GlobalMaxPooling2D()(x)
    # torch
    x = torch.nn.AdaptiveMaxPool2d(1)(x)
    x = torch.nn.Flatten()(x)
    
    ```
  
    
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
  
    ```python
    """
    The complete arch used is a UNet.
    We generally use resnet34 as the encoder. 
    GANs have a particularly hard time training if the generator and critic are not pretrained.
    
    Pre training involves that the generator knows the content of the images well.
    This is achieved by setting MSE as the loss func.
    """
    src = ImageImageList.from_folder(path_crappy).split_by_rand_pct(0.1, seed=42)
    
    def get_data(bs=bs, sz=sz):
        data = (src.label_from_func(lambda img_path: path_orig/img_path.name)
                .transform(get_transforms(max_zoom=0.2), size=sz, tfm_y=True)
                .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
        
        data.c = 3
        return data
    
    data_gen = get_data()
    
    wd = 1e-3
    y_range= (-3., 3.)
    loss_func = MSELossFlat()
    
    def create_gan_learner():
        return unet_learner(data_gen, arch=models.resnet34, blur=True, 		 	norm_type=NormType.Weight, y_range=y_range, self_attention=True, wd=wd, loss_func=loss_func)
    
    learn_gen.fit_one_cycle(3, slice(some_lr_range))
    ```
  
    
  
  - The middle convs
  
  - Pre-training the critic
  
  - The cross connections
  
  - Passing of inputs as a feature to final prediction

