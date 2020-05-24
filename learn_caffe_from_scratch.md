[TOC]

### Caffe2 installation

Follow the steps in https://caffe2.ai/docs/tutorials/

### Caffe2 Operators

https://caffe2.ai/docs/operators-catalogue.html


### Sparse Operations
https://caffe2.ai/docs/sparse-operations.html



### Debugging 

1. db_. Cannot find db implementation of type lmdb (while trying to open /home/xiaocong/caffe2_notebooks/tutorial_data/mnist/mnist-train-nchw-lmdb) (Error from operator:

  Solution:  https://github.com/pytorch/pytorch/issues/10119

  ```
  By default LMDB is OFF in the setup file, which creates this error.

  Following steps can be done In order to remove this issue

  Clean the previous setup
  python setup.py clean

  Reinstall caffe2 using below command

  USE_LMDB=ON python setup.py install

  Worked for me ...
  ```


### Input layers

#### Layer: Data from LevelIDB or LMDB


```
layer {
  name: "cifar"  # name of this layer, user defined
  type: "Data"   # type of the layer, 'Data ' means the input data comes from LevelIDB or LMDB
  top: "data"   # top: output/ bottom: input
  top: "label"
  include {
    phase: TRAIN    # this layer is used in Training phase. 
  }
  transform_param {
    mean_file: "examples/cifar10/mean.binaryproto"   # data pre-processing
  }
  data_param {
    source: "examples/cifar10/cifar10_train_lmdb"
    batch_size: 100   # batch size
    backend: LMDB
  }
}

```


#### Layer: Data from memory


```
layer {
  top: "data"
  top: "label"
  name: "memory_data"
  type: "MemoryData"
  memory_data_param{
    batch_size: 2
    height: 100
    width: 100
    channels: 1
  }
  transform_param {
    scale: 0.0078125    # normalization
    mean_file: "mean.proto"
    mirror: false
  }
}

```

#### Layer： Data from HDF5

```
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "examples/hdf5_classification/data/train.txt"
    batch_size: 10
  }
}

```

#### Layer: Data from image

```
 layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    mirror: false   # mirror off
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
  image_data_param {
    source: "examples/_temp/file_list.txt"
    batch_size: 50
    new_height: 256   # resize the image
    new_width: 256
  }
}
```


### Vision layers


#### Layer: convolution


```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1   ### coefficient of learning rate of weight
  }
  param {
    lr_mult: 2   ### coefficient of learning rate of bias
  }
  convolution_param {
    num_output: 20     ## output channel
    kernel_size: 5
    stride: 1
    pad: 0
    weight_filler {
      type: "xavier"   # initialization, 0/xavier/gaussian
    }
    bias_filler {
      type: "constant"  ## bias initialization, constant means all 0
    }
  }
}
```


#### Layer: pooling

```
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX      ###  MAX/AVE/STOCHASTIC
    kernel_size: 3
    stride: 2
  }
}
```




### Activation Layers

#### Sigmoid

```
layer {
  name: "encode1neuron"
  bottom: "encode1"
  top: "encode1neuron"
  type: "Sigmoid"
}
```



#### ReLU

```
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "pool1"
  top: "pool1"
  negative_slope: 0,1  ### Leaky ReLU : f(x) = max(0, z) + negative_slope * min (0, z)
}
```


#### Softmax 


```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"   ## output is loss
  bottom: "ip1"
  bottom: "label"
  top: "loss"
}
```

```
layers {
  bottom: "cls3_fc"
  top: "prob"
  name: "prob"
  type: “Softmax"   ### output is likelyhood
}
```


