### DataBunch API

- Create ItemList 
- Split by train and valid 
- Label it (to create LabelLists) 
- Transform it by calling `get_transforms` or passing a tuple of lists of tranforms to `labelList.transform(([training_tfms], []))`
- Call databunch after transforming
- normalize the data (load imagenet stats if loading a pre-trained model)

To grab a batch of data from a databunch, call  `one_batch()` on a databunch object

### Random Stuff

- DenseNets are super useful for small datasets and segmentation problems as the original image is retained till the end. They are obviously memory-intensive.

### Unets

- Instead of transpose or deconvolution. It's better to do nearest neighbour interpolation of pixels. As then the amount of wasted computation is low.
Also, it's better to replace the downsampling path of a unet with a resnet (jeremy uses a resnet34), the downside is that the original unet implemenation had 28x28x1024 as the final layer, whereas resnet34 gets it down to 7x7x2048 
