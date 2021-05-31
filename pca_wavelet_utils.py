
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def setupfilters3D(channels):
  filts = tf.constant([[1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0], [0.0, -1.0/4.0, 2.0/4.0, -1.0/4.0, 0.0], [0.0, 1.0/2.0, 0.0, -1.0/2.0, 0.0]], dtype=tf.float64)
  
  filts3D = []
  for k in range(channels):
    for i in range(3):
      for j in range(3):
        filt2D = tf.pad([tf.tensordot(filts[i], filts[j], axes=0)],[[k,channels-k-1],[0,0],[0,0]],mode="CONSTANT", constant_values=0)
        filts3D.append(filt2D)
  filters = tf.stack(filts3D)
  
  
  return filters

def filterImg3D(image, filts=None):
  if filts is None:
    filts = setupfilters3D(image.shape[2])
    filts = tf.transpose(filts,[2,3,1,0])
  
  img = tf.pad([image],[[0,0],[2,2],[2,2],[0,0]],"REFLECT")
  img = tf.nn.conv2d(img,filts,[1,2,2,1],'VALID',data_format='NHWC')
  return img[0]

def setupInverseFilters3D(channels):
  smooth = [0.0,      0.0,        1.0/16.0,   0.5,      14.0/16.0,  0.5,       1.0/16.0,   0.0,      0.0]
  even = [-1.0/128.0, -1.0/16.0, -10.0/64.0, -7.0/16.0, 85.0/64.0, -7.0/16.0, -10.0/64.0, -1.0/16.0, -1.0/128.0]
  odd = [1.0/256.0,  1.0/32.0,  15.0/128.0,17.0/32.0, 0.0,      -17.0/32.0,  -15.0/128.0, -1.0/32.0,  -1.0/256.0]
  
  filts = tf.constant([smooth, even, odd], dtype=tf.float64)
  
  filts3D = []

  for i in range(3):
    for j in range(3):
      filt2D = tf.tensordot(filts[i], filts[j], axes=0)
      filts3D.append(filt2D)
  filters = []
  for k in range(channels):
    filter = tf.pad(filts3D,[[k*9,(channels-k-1)*9],[0,0],[0,0]],mode="CONSTANT", constant_values=0)
    filters.append(filter)
  filters = tf.stack(filters)
  
  return filters

def addToPCA(ten, pca, mean):
  mat = tf.reshape(ten,[-1,ten.shape[2]])
  cov = tf.tensordot(mat,mat,[0,0])
  m = tf.ones(mat.shape[0], dtype=tf.float64)
  m = tf.linalg.matvec(mat,m, transpose_a=True)
  pca = pca+cov
  mean=mean+m
  
  return pca, mean

def completePCA(pca, mean):
  mouter = tf.tensordot(mean, mean, axes=0)
  pca -= mouter
  s, u, v = tf.linalg.svd(pca)
  
  return s,u

"""Filters for seperable implementation"""

def setupFilts1D():
   filts = tf.constant([[1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0], [0.0, -1.0/4.0, 2.0/4.0, -1.0/4.0, 0.0], [0.0, 1.0/2.0, 0.0, -1.0/2.0, 0.0]], dtype=tf.float64)
   smooth = [0.0,      0.0,        1.0/16.0,   0.5,      14.0/16.0,  0.5,       1.0/16.0,   0.0,      0.0]
   even = [-1.0/128.0, -1.0/16.0, -10.0/64.0, -7.0/16.0, 85.0/64.0, -7.0/16.0, -10.0/64.0, -1.0/16.0, -1.0/128.0]
   odd = [1.0/256.0,  1.0/32.0,  15.0/128.0,17.0/32.0, 0.0,      -17.0/32.0,  -15.0/128.0, -1.0/32.0,  -1.0/256.0]
   invfilts = tf.constant([smooth, even, odd], dtype=tf.float64)
   return filts, invfilts

import numpy as np
def borderMultiplier(shape, xAxis):
  mulval = np.ones(shape)
  if (xAxis):
    for i in range(3):
      for j in range(shape[0]):
        for k in range(int(shape[2]/3)):
          mulval[j,i,2+3*k]=-1
          mulval[j,shape[1]-i-1,2+3*k]=-1
  else:
    for i in range(3):
      for j in range(shape[1]):
        for k in range(int(shape[2]/3)):
          mulval[i,j,2+3*k]=-1
          mulval[shape[0]-i-1,j,2+3*k]=-1

  return mulval

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Conv2DTransposeSeparableLayer(Layer):

    def __init__(self, input_shape, **kwargs):
        self.filters, self.invfilters = setupFilts1D()
        self.input_shapeX = [input_shape[0],input_shape[1]+6,int(input_shape[2])]
        self.tfmultx = borderMultiplier(self.input_shapeX,True)
        self.input_shapeY = [input_shape[0]+6,input_shape[1]*2,int(input_shape[2]/3)]
        self.tfmulty = borderMultiplier(self.input_shapeY,False)
        
        super(Conv2DTransposeSeparableLayer, self).__init__(**kwargs)

    def get_config(self):
      return {'input_shapeX': self.input_shapeX,
              'tfmultx':self.tfmultx,
              'input_shapeY':self.input_shapeY,
              'tfmulty':self.tfmulty}

    def build(self, input_shape):
        super(Conv2DTransposeSeparableLayer, self).build(input_shape)

    def call(self, inputs):
        output = self.invFilterImgX(inputs, self.tfmultx)
        output = self.invFilterImgY(output, self.tfmulty)
        return output 

    def compute_output_shape(self, input_shape):
        output_shape=input_shape
        output_shape[0]*=2
        output_shape[1]*=2
        output_shape[2] = int(output_shape[2]/9)
        return (output_shape)

        import numpy as np

    def invFilterImgX(self,image, tfmulval):
      img = tf.pad(image,[[0,0],[0,0],[3,3],[0,0]],"SYMMETRIC")#"REFLECT")
      img = tf.math.multiply(img,tfmulval)
      filter, invfilter = setupFilts1D()
      invfilter = tf.transpose([[invfilter]],[0,3,1,2])
      outputs = []
      for i in range(int(img.shape[3]/3)):
        slice = tf.gather(img,[3*i+0,3*i+1,3*i+2],axis=3)
        im = tf.nn.conv2d_transpose(slice,invfilter,[img.shape[0],img.shape[1],img.shape[2]*2+7,1],[1,2],padding='VALID',data_format='NHWC')
        outputs.append(im)
        
      outimg = tf.stack(outputs)
      outimg = tf.transpose(outimg, perm=[1,2,3,0,4])[:,:,:,:,0]
      
      return outimg[:,:,10:outimg.shape[2]-9,:]

    def invFilterImgY(self, image, tfmulval):
      img = tf.pad(image,[[0,0],[3,3],[0,0],[0,0]],"SYMMETRIC")
      img = tf.math.multiply(img,tfmulval)
      filter, invfilter = setupFilts1D()
      
      invfilter = tf.transpose([[invfilter]],[3,0,1,2])
      outputs = []
      for i in range(int(img.shape[3]/3)):
        slice = tf.gather(img,[3*i+0,3*i+1,3*i+2],axis=3)
        im = tf.nn.conv2d_transpose(slice,invfilter,[img.shape[0],img.shape[1]*2+7,img.shape[2],1],[2,1],padding='VALID',data_format='NHWC')
        outputs.append(im)
        
      outimg = tf.stack(outputs)
      outimg = tf.transpose(outimg, perm=[1,2,3,0,4])[:,:,:,:,0]
      outimg = outimg[:,10:outimg.shape[1]-9,:,:]#8,-7
      
      return outimg

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class BiasLayer(Layer):

    def __init__(self, bias, data_format="channels_last", **kwargs):
        self.data_format = data_format
        self.bias = bias
        super(BiasLayer, self).__init__(**kwargs)

    def get_config(self):
      return {'data_format': self.data_format,
              'bias':self.bias}

    def build(self, input_shape):
        super(BiasLayer, self).build(input_shape)

    def call(self, inputs):
        if self.data_format is "channels_last":
            output = tf.nn.bias_add(inputs,self.bias,'NDHWC')
        elif self.data_format is "channels_first":
            output = tf.nn.bias_add(inputs,self.bias,'NCDHW')

        return output 

    def compute_output_shape(self, input_shape):
        return (input_shape)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class SymmetricUnPadding2D(Layer):

    def __init__(self, output_dim, padding=[1,1,1,1], 
                 data_format="channels_last", **kwargs):
        self.output_dim = output_dim
        self.data_format = data_format
        self.padding = padding
        super(SymmetricUnPadding2D, self).__init__(**kwargs)

    def get_config(self):
      return {'output_dim': self.output_dim,
              'data_format':self.data_format,
              'padding':self.padding}

    def build(self, input_shape):
        super(SymmetricUnPadding2D, self).build(input_shape)

    def call(self, inputs):
        if self.data_format is "channels_last":
            output = inputs[:,self.padding[0]:inputs.shape[1]-self.padding[1],self.padding[2]:inputs.shape[2]-self.padding[3],:]
            return (output)
        elif self.data_format is "channels_first":
            output = inputs[:,:,self.padding[0]:inputs.shape[1]-self.padding[1],self.padding[2]:inputs.shape[2]-self.padding[3]]
            return (output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class SymmetricPadding2D(Layer):

    def __init__(self, output_dim, padding=[1,1], 
                 data_format="channels_last", **kwargs):
        self.output_dim = output_dim
        self.data_format = data_format
        self.padding = padding
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def get_config(self):
      return {'output_dim': self.output_dim,
              'data_format': self.data_format,
              'padding': self.padding}

    def build(self, input_shape):
        super(SymmetricPadding2D, self).build(input_shape)

    def call(self, inputs):
        if self.data_format is "channels_last":
            pad = [[0,0]] + [[i,i] for i in self.padding] + [[0,0]]
        elif self.data_format is "channels_first":
            pad = [[0, 0], [0, 0]] + [[i,i] for i in self.padding]

        paddings = tf.constant(pad)
        out = tf.pad(inputs, paddings, "SYMMETRIC")
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

import numpy as np
def reverse(x,y):
  z = tf.reverse(x,[1])
  return z,y


"""Replace Lambda layers with a class to allow saving to file."""

from tensorflow.keras import backend as K
class MeanLayer(tf.keras.layers.Layer):
    def __init__(self, mean, **kwargs):
      super(MeanLayer, self).__init__()
      self.mean = tf.keras.backend.variable(tf.keras.backend.cast_to_floatx(mean), dtype='float64')
      print("self.mean.dtype",self.mean.dtype)
      super(MeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
      return inputs + self.mean
    
    def get_config(self):
      serial_mean = tf.keras.backend.eval(self.mean)
      print("Getting the config")
      config = super(MeanLayer,self).get_config()
      config.update({'mean': serial_mean})
      print("config = ", config)
      return config


"""Build method using 1D convolutions, to better handle borders, in particular the issue with asymmetric forward filters, need to pad the borders with asymmetric flipped values."""

import numpy as np

def build1D(dataset, channels = 3, count = 6, samplesize=-1, keep_percent = 0.2, flip=False, activity_regularizer=None, inverse_activity_regularizer=None, activation_before=False):

  keep_percent = 4.0/9.0*pow(keep_percent, 1/float(count))
  print("keep_percent",keep_percent, flush=True)
  head = tf.keras.Sequential()
  head.run_eagerly=True
  invhead = tf.keras.Sequential()
  invlist = []
  invbinit = tf.constant_initializer(np.zeros(3))
  
  subset = dataset
  
  
  if flip:
    flipped = subset.map(lambda x,y: reverse(x,y))
    
    subset = subset.concatenate(flipped)
    samplesize*=2
  it = iter(subset)
  meanimg = tf.cast(next(it)[0], tf.float64)
  sizex = meanimg.shape[1]
  IMAGE_SIZE_X = sizex
  sizey = meanimg.shape[0]
  IMAGE_SIZE_Y = sizey


  for i in range(1,samplesize):
    meanimg += tf.cast(next(it)[0], tf.float64)
  print("meanimg.dtype",meanimg.dtype, flush=True)
  meanimg /= float(samplesize)
  head.add(MeanLayer(-meanimg))
  invlist.append(MeanLayer(meanimg))

  for lev in range(count):
    print("Starting level",lev,flush=True)
    outchan = channels*9
    pca = tf.zeros([outchan,outchan], dtype=tf.float64)
    mean = tf.zeros(outchan, dtype=tf.float64)
    filts3D = setupfilters3D(channels)
    filts3D = tf.transpose(filts3D,[2,3,1,0])
    
    newsizex=sizex/2
    newsizey=sizey/2
    for image in subset:
      img = tf.cast(image[0], tf.float64)
      img = tf.transpose([img],[0,1,2,3])
      pred = head(img)[0]
      filtered = filterImg3D(pred, filts=filts3D)
      pca,mean = addToPCA(filtered, pca, mean)
    print("Completing",newsizex)
    pca = pca/float(newsizex*newsizey*samplesize)
    mean /= float(newsizex*newsizey*samplesize)
    print("pca shape",tf.shape(pca))
    s,u = completePCA(pca,mean)
    keep_channels = int(keep_percent*u.shape[1])#(4.0/9.0)
    var_explained=0
    var_total = tf.math.reduce_sum(s,0)
    s = s/var_total
    var_total_post = tf.math.reduce_sum(s,0)
    keep_max = channels*(IMAGE_SIZE_Y/filtered.shape[0])*(IMAGE_SIZE_X/filtered.shape[1])
    print("keep_channels",keep_channels, "keep_max", keep_max)
    compcount=0
    while (var_explained<1.0 and compcount<keep_max and compcount<keep_channels):
      var_explained+=s[compcount]
      compcount+=1

    keep_channels = compcount
    print("keep_channels",keep_channels)
    ufilts = tf.transpose([[[u[:,0:keep_channels]]]],[0,1,2,3,4])
    print("ufilts.shape",ufilts.shape)
    
  
    filts3D = tf.transpose([filts3D],[0,3,1,2,4])
    newfilt = tf.nn.conv3d(filts3D,ufilts,[1,1,1,1,1],'VALID',data_format='NDHWC')
    filtsOrig = tf.transpose(newfilt[0], [1,2,0,3])
    numpynewfilt = filtsOrig.numpy()
    init = tf.constant_initializer(numpynewfilt)
    bias = -tf.linalg.matvec(ufilts,mean, transpose_a=True)[0,0,0]
    binit = tf.constant_initializer(bias.numpy())
    if (activation_before):
      head.add(tf.keras.layers.Activation(activity_regularizer))
    
    head.add(SymmetricPadding2D(2, input_shape=(int(sizey),int(sizex),channels), padding=[2,2]))
    head.add(tf.keras.layers.Conv2D(keep_channels, (5, 5), strides=(2, 2), input_shape=(int(sizey)+4, int(sizex)+4, channels),
                                    kernel_initializer=init, bias_initializer=binit))
    if (not activation_before):
      head.add(tf.keras.layers.Activation(activity_regularizer))
    
    target_shape = [filtered.shape[0],filtered.shape[1],u.shape[1]]
    if (activation_before):
      invlist.append(tf.keras.layers.Activation(inverse_activity_regularizer))
    
    invlist.append(Conv2DTransposeSeparableLayer(target_shape))
    utfilts = tf.transpose([[u[:,0:keep_channels]]],[0,1,3,2])
    binit = tf.constant_initializer(mean.numpy())
    kinit = tf.constant_initializer(utfilts.numpy())

    invlist.append(tf.keras.layers.Conv2D(u.shape[0],[1,1],strides=1,padding='VALID', use_bias=True, kernel_initializer=kinit, bias_initializer=binit))
    if (not activation_before):
      invlist.append(tf.keras.layers.Activation(inverse_activity_regularizer))
    

    channels = keep_channels
    sizex=newsizex
    sizey=newsizey
    print("end loop", sizex)

  #invert invlist for reconstruction
  it = reversed(invlist)
  for e in it:
    invhead.add(e)

  return head, invhead
  
