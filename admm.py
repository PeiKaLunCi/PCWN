
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
else:
  print('Found GPU at: {}'.format(device_name))

"""Solver for L1-norm"""
import sys
sys.path.append('')
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import matplotlib
import matplotlib.pyplot as plt
import timeit
import pywt
import os

def vec(x):
    return x.ravel(order='F')


def sigmoid(x):
    return 1/(1+np.exp(-x))


def wavelet_transform(x):
    w_coeffs_rgb = []
    for i in range(x.shape[2]):
        w_coeffs_list = pywt.wavedec2(x[:,:,i], 'db4', level=None, mode='periodization')
        w_coeffs, coeff_slices = pywt.coeffs_to_array(w_coeffs_list)
        w_coeffs_rgb.append(w_coeffs)

    w_coeffs_rgb = np.array(w_coeffs_rgb)
    return w_coeffs_rgb, coeff_slices


def inverse_wavelet_transform(w_coeffs_rgb, coeff_slices, x_shape):
    x_hat = np.zeros(x_shape)
    for i in range(w_coeffs_rgb.shape[0]):
        w_coeffs_list = pywt.array_to_coeffs(w_coeffs_rgb[i,:,:], coeff_slices)
        x_hat[:,:,i] = pywt.waverecn(w_coeffs_list, wavelet='db4', mode='periodization')
    return x_hat


def soft_threshold(x, beta):
    y = np.maximum(0, x-beta) - np.maximum(0, -x-beta)
    return y



# A_fun, AT_fun takes a vector (d,1) or (d,) as input
def solve_l1(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, show_img_progress=False, alpha=0.2, max_iter=100, solver_tol=1e-6):
    """ See Wang, Yu, Wotao Yin, and Jinshan Zeng. "Global convergence of ADMM in nonconvex nonsmooth optimization."
    arXiv preprint arXiv:1511.06324 (2015).
    It provides convergence condition: basically with large enough alpha, the program will converge. """


    obj_lss = np.zeros(max_iter)
    x_zs = np.zeros(max_iter)
    u_norms = np.zeros(max_iter)
    times = np.zeros(max_iter)

    ATy = AT_fun(y)
    x_shape = ATy.shape
    d = np.prod(x_shape)

    def A_cgs_fun(x):
        x = np.reshape(x, x_shape, order='F')
        y = AT_fun(A_fun(x)) + alpha * x
        return vec(y)
    A_cgs = LinearOperator((d,d), matvec=A_cgs_fun, dtype='float')

    def compute_p_inv_A(b, z0):
        (z,info) = sp.sparse.linalg.cgs(A_cgs, vec(b), x0=vec(z0), tol=1e-3, maxiter=100)
        if info > 0:
            print('cgs convergence to tolerance not achieved')
        elif info <0:
            print('cgs gets illegal input or breakdown')
        z = np.reshape(z, x_shape, order='F')
        return z


    def A_cgs_fun_init(x):
        x = np.reshape(x, x_shape, order='F')
        y = AT_fun(A_fun(x))
        return vec(y)
    A_cgs_init = LinearOperator((d,d), matvec=A_cgs_fun_init, dtype='float')

    def compute_init(b, z0):
        (z,info) = sp.sparse.linalg.cgs(A_cgs_init, vec(b), x0=vec(z0), tol=1e-2)
        if info > 0:
            print('cgs convergence to tolerance not achieved')
        elif info <0:
            print('cgs gets illegal input or breakdown')
        z = np.reshape(z, x_shape, order='F')
        return z

    # initialize z and u
    z = compute_init(ATy, ATy)
    u = np.zeros(x_shape)


    plot_normalozer = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)


    start_time = timeit.default_timer()

    for iter in range(max_iter):

        # x-update
        net_input = z+u
        Wzu, wbook = wavelet_transform(net_input)
        q = soft_threshold(Wzu, lambda_l1/alpha)
        x = inverse_wavelet_transform(q, wbook, x_shape)
        x = np.reshape(x, x_shape)

        # z-update
        b = ATy + alpha * (x - u)
        z = compute_p_inv_A(b, z)

        # u-update
        u += z - x;

        if show_img_progress == True:

            fig = plt.figure('current_sol')
            plt.gcf().clear()
            fig.canvas.set_window_title('iter %d' % iter)
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(np.clip(x, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('x')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(np.clip(z, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('z')
            plt.subplot(1,3,3)
            plt.imshow(reshape_img_fun(np.clip(net_input, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('netin')
            plt.pause(0.00001)


        obj_ls = 0.5 * np.sum(np.square(y - A_fun(x)))
        x_z = np.sqrt(np.mean(np.square(x-z)))
        u_norm = np.sqrt(np.mean(np.square(u)))

        print('iter = %d: obj_ls = %.3e  |x-z| = %.3e  u_norm = %.3e' % (iter, obj_ls, x_z, u_norm))


        obj_lss[iter] = obj_ls
        x_zs[iter] = x_z
        u_norms[iter] = u_norm
        times[iter] = timeit.default_timer() - start_time

        if x_z < solver_tol:
            break

    infos = {'obj_lss': obj_lss, 'x_zs': x_zs, 'u_norms': u_norms,
             'times': times, 'alpha':alpha, 'lambda_l1':lambda_l1,
             'max_iter':max_iter, 'solver_tol':solver_tol}


    return (x, z, u, infos)



"""Setup for inpaint centre"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def setup_inpaint_centre(x_shape, box_size):


    mask = np.ones(x_shape)


    idx_row = np.round(float(x_shape[0]) / 2.0 - float(box_size) / 2.0).astype(int)
    idx_col = np.round(float(x_shape[1]) / 2.0 - float(box_size) / 2.0).astype(int)

    mask[idx_row:idx_row+box_size,idx_col:idx_col+box_size,:] = 0.


    def A_fun(x):
        y = np.multiply(x, mask);
        return y

    def AT_fun(y):
        x = np.multiply(y, mask);
        return x

    return (A_fun, AT_fun, mask)

"""Setup pixelwise inpaint"""

def setup_pixelwise_inpaint(x_shape, drop_prob = 0.5):

    mask = np.random.rand(*x_shape) > drop_prob;
    mask = mask.astype('double')

    def A_fun(x):
        y = np.multiply(x, mask);
        return y

    def AT_fun(y):
        x = np.multiply(y, mask);
        return x

    return (A_fun, AT_fun, mask)

"""Setup scattered inpaint"""

""" currently only support width (and height) * resize_ratio is an interger! """
def setup_scattered_inpaint(x_shape, box_size, total_box = 10):

    spare = 0.25 * box_size

    mask = np.ones(x_shape)

    for i in range(total_box):

        start_row = spare
        end_row = x_shape[0] - spare - box_size - 1
        start_col = spare
        end_col = x_shape[1] - spare - box_size - 1

        idx_row = int(np.random.rand(1) * (end_row - start_row) + start_row)
        idx_col = int(np.random.rand(1) * (end_col - start_col) + start_col)

        mask[idx_row:idx_row+box_size,idx_col:idx_col+box_size,:] = 0.


    def A_fun(x):
        y = np.multiply(x, mask);
        return y

    def AT_fun(y):
        x = np.multiply(y, mask);
        return x

    return (A_fun, AT_fun, mask)

"""Setup compressive sensing"""

def setup_cs(x_shape, compress_ratio=0.1):

    d = np.prod(x_shape).astype(int)
    m = np.round(compress_ratio * d).astype(int)

    A = tf.random.normal([m,d], dtype=tf.float64)/np.sqrt(m)
    print("A.shape", A.shape)

    def A_fun(x):
        xd = tf.reshape(x,[d])
        y = tf.linalg.matvec(A, xd)
        y = tf.reshape(y, [1,m])
        return y

    def AT_fun(y):
        y = tf.reshape(y, [m])
        x = tf.linalg.matvec(A, y, transpose_a=True)
        x = tf.reshape(x, x_shape)
        return x

    return (A_fun, AT_fun, A)

"""Setup super resolution"""

def setup_sr2(x_shape):
  filts = tf.constant([0.5,0.5], dtype=tf.float64)
  filts3D = []
  for k in range(x_shape[2]):
        filt2D = tf.pad([tf.tensordot(filts, filts, axes=0)],[[k,x_shape[2]-k-1],[0,0],[0,0]],mode="CONSTANT", constant_values=0)
        filts3D.append(filt2D)
  filters = tf.stack(filts3D)
  filters = tf.transpose(filters,[2,3,0,1])

  ifilts = tf.constant([1.0,1.0], dtype=tf.float64)
  ifilts3D = []

  for k in range(x_shape[2]):
        ifilt2D = tf.pad([tf.tensordot(ifilts, ifilts, axes=0)],[[k,x_shape[2]-k-1],[0,0],[0,0]],mode="CONSTANT", constant_values=0)
        ifilts3D.append(ifilt2D)
  ifilters = tf.stack(ifilts3D)
  ifilters = tf.transpose(ifilters,[2,3,0,1])
  out_shape = [1,x_shape[0],x_shape[1],x_shape[2]]

  def A_fun(x):
    y = tf.nn.conv2d([x],filters,strides=2,padding="VALID")
    return y[0]

  def AT_fun(y):
    x = tf.nn.conv2d_transpose([y],
    ifilters,
    out_shape,
    strides=2,
    padding='VALID',
    data_format='NHWC',
    dilations=None,
    name=None)
    return x[0]

  return (A_fun, AT_fun)

""" currently only support width (and height) * resize_ratio is an interger! """
def setup_sr(x_shape, resize_ratio=0.5):

    box_size = 1.0 / resize_ratio
    if np.mod(x_shape[1], box_size) != 0 or np.mod(x_shape[2], box_size) != 0:
        print("only support width (and height) * resize_ratio is an interger!")


    def A_fun(x):
        y = box_average(x, int(box_size))
        return y

    def AT_fun(y):
        x = box_repeat(y, int(box_size))
        return x

    return (A_fun, AT_fun)



def box_average(x, box_size):
    """ x: [1, row, col, channel] """
    im_row = x.shape[0]
    im_col = x.shape[1]
    channel = x.shape[2]
    out_row = np.floor(float(im_row) / float(box_size)).astype(int)
    out_col = np.floor(float(im_col) / float(box_size)).astype(int)
    y = np.zeros((out_row,out_col,channel))
    total_i = int(im_row / box_size)
    total_j = int(im_col / box_size)

    for c in range(channel):
        for i in range(total_i):
            for j in range(total_j):
                avg = np.average(x[i*int(box_size):(i+1)*int(box_size), j*int(box_size):(j+1)*int(box_size), c], axis=None)
                y[i,j,c] = avg

    return y


def box_repeat(x, box_size):
    """ x: [1, row, col, channel] """
    im_row = x.shape[0]
    im_col = x.shape[1]
    channel = x.shape[2]
    out_row = np.floor(float(im_row) * float(box_size)).astype(int)
    out_col = np.floor(float(im_col) * float(box_size)).astype(int)
    y = np.zeros((out_row,out_col,channel))
    total_i = im_row
    total_j = im_col

    for c in range(channel):
        for i in range(total_i):
            for j in range(total_j):
                y[i*int(box_size):(i+1)*int(box_size), j*int(box_size):(j+1)*int(box_size), c] = x[i,j,c]
    return y

"""add noise function"""

def add_noise(x, noise_mean = 0.0, noise_std = 0.1):
    noise = np.random.randn(*x.shape) * noise_std + noise_mean;
    y = x + noise
    return y, noise

def reshape_img(img):
        return img

IMAGE_SIZE = 64
def pre_process_image(image):
  print("pre_process image.shape", image.shape)
  image = tf.cast(image, tf.float64)
  image = image / 255.0
  print("pre_process image.shape resized", image.shape, image.dtype)
  image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) #resize changes type to float32!
  
  print("pre_process image.shape resized", image.shape, image.dtype)
  image = tf.cast(image, tf.float64)
  print("pre_process image.shape resized", image.shape, image.dtype)
  
  return image

def pre_process_entry(image, label):
  image = pre_process_image(image)  
  return image, label

"""Solve inpaint centre L1"""

def solve_inpaint_center(ori_img, reshape_img_fun, head, invhead, mean,
                        box_size=1, noise_mean=0, noise_std=0.,
                            alpha=0.3, lambda_l1=0.1, max_iter=100, solver_tol=1e-2, problem='inpaint_center', show_img_progress=False):
        #import inpaint_center as problem
        x_shape = ori_img.shape
        print("x_shape", x_shape)
        if (problem=='inpaint_center'):
          (A_fun, AT_fun, mask) = setup_inpaint_centre(x_shape, box_size=box_size)
        elif (problem=='inpaint_scattered'):
          (A_fun, AT_fun, mask) = setup_scattered_inpaint(x_shape, box_size=box_size)
        elif (problem=='inpaint_pixelwise'):
          (A_fun, AT_fun, mask) = setup_pixelwise_inpaint(x_shape)
        elif (problem == 'cs'):
          (A_fun, AT_fun, A) = setup_cs(x_shape)
        elif (problem == 'sr'):
          (A_fun, AT_fun) = setup_sr2(x_shape)
        y, noise = add_noise(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if True:#show_img_progress:
            fig = plt.figure(problem)
            plt.gcf().clear()
            fig.canvas.set_window_title(problem)
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            if (problem!='sr' and problem!='cs'):
              plt.subplot(1,3,3)
              plt.imshow(reshape_img_fun(mask), interpolation='nearest')
              plt.title('mask')
            plt.pause(0.00001)

        info = {'ori_img': ori_img, 'y': y, 'noise': noise,  'box_size': box_size, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_l1': lambda_l1}


        run_ours=True
        if run_ours:
            # ours
            (x, z, u, infos) = solve_pcaw(y, A_fun, AT_fun, lambda_l1,reshape_img_fun, head, invhead, mean,
                                          show_img_progress=show_img_progress, alpha=alpha,
                                          max_iter=max_iter, solver_tol=solver_tol)
        run_l1 = False
        if run_l1:
            # wavelet l1
            
            (x, z, u, infos) = solve_l1_alt(y, A_fun, AT_fun, lambda_l1, reshape_img_fun,
                                            show_img_progress=show_img_progress, alpha=alpha,
                                            max_iter=max_iter, solver_tol=solver_tol)


        z1 = reshape_img(np.clip(z, 0.0, 1.0))
        ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
        psnr_z = 10*np.log10( 1.0 /((np.linalg.norm(z1-ori_img1)**2)/np.prod(z1.shape)))
        print("psnr_z = ", psnr_z) 
        z1 = reshape_img(np.clip(x, 0.0, 1.0))
        psnr_x = 10*np.log10( 1.0 /((np.linalg.norm(z1-ori_img1)**2)/np.prod(z1.shape)))
        print("psnr_x = ", psnr_x)  

        if True:#show_img_progress:
            fig = plt.figure('current_sol')
            plt.gcf().clear()
            fig.canvas.set_window_title('final')
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(np.clip(x, 0.0, 1.0)), interpolation='nearest')
            plt.title('x')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(np.clip(z, 0.0, 1.0)), interpolation='nearest')
            plt.title('z')
            plt.subplot(1,3,3)
            plt.imshow(reshape_img_fun(np.clip(u, 0.0, 1.0)), interpolation='nearest')
            plt.title('netin')
            plt.pause(0.00001)

            fig = plt.figure('inpaint_center')
            plt.gcf().clear()
            fig.canvas.set_window_title('inpaint_center')
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            if (problem!='sr' and problem!='cs'):
              plt.subplot(1,3,3)
              plt.imshow(reshape_img_fun(mask), interpolation='nearest')
              plt.title('mask')
            plt.pause(0.00001)
        return psnr_z,psnr_x

class TrainingGenerator:
  def __init__(self):
    self.batch_size = 64
    self.data_dir = '/home/bpt/onenet/du-admm/diff_unrolled_admm_onenet/img_align_celeba'
    self.img_height = 64
    self.img_width = 64
    import os
    cwd = os.getcwd() 
    print("cwd",cwd)
    self.idg = tf.keras.preprocessing.image.ImageDataGenerator()
    self.iter = self.idg.flow_from_directory(self.data_dir,
            target_size=(self.img_width, self.img_height), 
            color_mode='rgb', classes=['train'], class_mode='input', batch_size=1, 
            shuffle=True, seed=None, save_to_dir=None, save_prefix='', 
            save_format='png', follow_links=False, subset=None, 
            interpolation='bilinear')
  

  def __iter__(self):
    return self

  def __next__(self):
    return self.iter.__next__()
    
"""Import some data to play with"""
def import_data():
    import tensorflow_datasets as tfds
    
    dataset, metadata = tfds.load('downsampled_imagenet/64x64:2.0.0',
                                  with_info=True, as_supervised=False)
    
    #dataset = import_celeba_local()
    return dataset

def import_celeba_local():
    batch_size = 64
    data_dir = 'du-admm/diff_unrolled_admm_onenet/img_align_celeba/train/'
    img_height = 64
    img_width = 64
    
    dataset = tf.data.Dataset.from_generator(TrainingGenerator, (tf.float32), (2, 1, 64, 64, 3))
    
    dataset = dataset.map(lambda x:x[0,0,:,:,:])
    return dataset
    
    """Import library for pca-wavelets"""
import pca_wavelet_utils

"""Set up activation functions"""

def scaledtanh(x):
  return tf.math.tanh(x*0.1)

def scaledatanh(x):
  return tf.math.atanh(x)*10.0

"""Build the model for pca-wavelet"""
def build_model(dataset):
    from pca_wavelet_utils import build1D
    tf.keras.backend.set_floatx('float64')
    trainset = dataset['train'].map(lambda x:[pre_process_image(x['image'])])
    testset = dataset['validation'].map(lambda x:[pre_process_image(x['image'])])
    
    head, invhead = build1D(trainset,count=4, samplesize=1281149, keep_percent=1.0, flip=False) #, activity_regularizer=scaledtanh, inverse_activity_regularizer=scaledatanh)
    return head, invhead, trainset, testset

"""Save the model"""
def save_model():
    sample = next(iter(testset.shuffle(100)))[0]
    sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
    head._set_inputs(sample)
    head.save('/content/drive/My Drive/Colab Notebooks/data/imagenet/lfw-head-full.h5')
    out = head(sample)
    print("out.shape",out.shape)
    sample = invhead(out)
    invhead.save('/content/drive/My Drive/Colab Notebooks/data/imagenet/lfw-invhead-full.h5')

"""Load the model"""
def load_model():
    head = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/data/imagenet/lfw-head-full.h5', custom_objects={'MeanLayer': MeanLayer, 'SymmetricPadding2D':SymmetricPadding2D})
    invhead = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/data/imagenet/lfw-invhead-full.h5')

"""Save the weights only"""
def save_weights(head, invhead, testset, file_name):
    sample = next(iter(testset.shuffle(100)))[0]
    sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
    out = head(sample)
    sample = sample*0.0
    lastLayerIndex = 12
    lastLayer = invhead.get_layer(index=lastLayerIndex)
    mean = lastLayer(sample)
    tf.io.write_file(file_name + '-mean.json', tf.io.serialize_tensor(mean))
    head.save_weights(file_name + '-head-weights.h5')
    out = head(sample)
    print("out.shape",out.shape)
    sample = invhead(out)
    invhead.save_weights(file_name + '-invhead-weights.h5')

"""Read the weights back in.  Need to reconstruct the architecture.  To do that I run a small set of images through the build method."""
def load_weights(file_name, keep_percent, trainset, testset):
    from pca_wavelet_utils import build1D
    head, invhead = build1D(trainset.take(100),count=4, samplesize=100, keep_percent=keep_percent,flip=False)
    sample = next(iter(testset.shuffle(100)))[0]
    print("sample.shape",sample.shape)
    sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
    print("after reshape: sample.shape",sample.shape)
    
    out = head(sample)
    head.load_weights(file_name + '-head-weights.h5')
    out = head(sample)
    print("out.shape",out.shape)
    sample = invhead(out)
    invhead.load_weights(file_name + '-invhead-weights.h5')
    mean = tf.io.parse_tensor(tf.io.read_file(file_name + '-mean.json'),out_type=tf.float64)
    lastLayerIndex = 12#8
    lastLayer = invhead.get_layer(index=lastLayerIndex)
    lastLayer.mean = mean
    firstLayer = head.get_layer(index=0)
    firstLayer.mean = -mean
    return head, invhead, mean

"""Check it has built OK"""
def check_build():
    plt.subplot(221)
    plt.title('Original')
    sample = next(iter(testset.shuffle(100)))[0]
    
    plt.imshow(sample)
    print("sample.shape",sample.shape)
    
    pred = head([sample])

    plt.subplot(222)
    plt.title('Slice')
    plt.imshow(pred[0,:,:,0]+0.5)
    plt.subplot(223)
    plt.title('Slice')
    plt.imshow(pred[0,:,:,1]+0.5)
    
    print("pred.shape",pred.shape)
    recon = invhead(pred)[0]
    print("recon.shape",recon.shape)
    plt.subplot(224)
    plt.title('Filtered')
    plt.imshow(recon)
    print("sample.dtype",sample.dtype)
    print("recon[0].dtype",recon.dtype)
    print("np.prod(sample.shape)",np.prod(sample.shape))
    psnr = 10*np.log10( 1.0 /((np.linalg.norm(recon-sample)**2)/np.prod(sample.shape)))
    ncc = np.corrcoef(tf.reshape(sample, [-1]), tf.reshape(recon, [-1]))
    print("psnr = ", psnr)
    print("ncc = ", ncc)
    print("sample[30:34,30:34,0]",sample[30:34,30:34,0])
    print("recon[30:34,30:34,0]",recon[30:34,30:34,0])

"""Solver for PCA wavelet"""

# A_fun, AT_fun takes a vector (d,1) or (d,) as input
def solve_pcaw(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, head, invhead, mean, show_img_progress=False, alpha=0.2, max_iter=100, solver_tol=1e-6):
    """ See Wang, Yu, Wotao Yin, and Jinshan Zeng. "Global convergence of ADMM in nonconvex nonsmooth optimization."
    arXiv preprint arXiv:1511.06324 (2015).
    It provides convergence condition: basically with large enough alpha, the program will converge. """


    obj_lss = np.zeros(max_iter)
    x_zs = np.zeros(max_iter)
    u_norms = np.zeros(max_iter)
    times = np.zeros(max_iter)

    ATy = AT_fun(y)
    x_shape = ATy.shape
    d = np.prod(x_shape)
    
    def vec(x):
      return tf.reshape(x, [-1])

    def A_cgs_fun(x):
        x = tf.reshape(x,x_shape)
        y = AT_fun(A_fun(x)) + alpha * x
        return vec(y)
    A_cgs = LinearOperator((d,d), matvec=A_cgs_fun, dtype='float')

    def compute_p_inv_A(b, z0):
        (z,info) = sp.sparse.linalg.cgs(A_cgs, vec(b), x0=vec(z0), tol=1e-3, maxiter=100)
        if info > 0:
            print('cgs convergence to tolerance not achieved')
        elif info <0:
            print('cgs gets illegal input or breakdown')
        z = tf.reshape(z, x_shape)
        return z


    def A_cgs_fun_init(x):
        x = tf.reshape(x, x_shape)
        y = AT_fun(A_fun(x))
        return vec(y)
    A_cgs_init = LinearOperator((d,d), matvec=A_cgs_fun_init, dtype='float')

    def compute_init(b, z0):
        (z,info) = sp.sparse.linalg.cgs(A_cgs_init, vec(b), x0=vec(z0), tol=1e-2)
        if info > 0:
            print('cgs convergence to tolerance not achieved')
        elif info <0:
            print('cgs gets illegal input or breakdown')
        z = tf.reshape(z,x_shape)
        return z

    # initialize z and u
    z = tf.reshape(mean,x_shape)
    u = np.zeros(x_shape)

    plot_normalozer = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)


    start_time = timeit.default_timer()

    for iter in range(max_iter):

        # x-update
        net_input = z+u
        
        Wzu = head([net_input])
        q = tfp.math.soft_threshold(Wzu, lambda_l1/alpha)
        x = invhead(q)[0]

        # z-update
        b = ATy + alpha * (x - u)
        z = compute_p_inv_A(b, z)

        # u-update
        u += z - x;

        if show_img_progress:

            fig = plt.figure('current_sol')
            plt.gcf().clear()
            fig.canvas.set_window_title('iter %d' % iter)
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(np.clip(x, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('x')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(np.clip(z, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('z')
            plt.subplot(1,3,3)
            plt.imshow(reshape_img_fun(np.clip(net_input, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('netin')
            plt.pause(0.00001)


        obj_ls = 0.5 * np.sum(np.square(y - A_fun(x)))
        x_z = np.sqrt(np.mean(np.square(x-z)))
        u_norm = np.sqrt(np.mean(np.square(u)))

        obj_lss[iter] = obj_ls
        x_zs[iter] = x_z
        u_norms[iter] = u_norm
        times[iter] = timeit.default_timer() - start_time

        if x_z < solver_tol:
            break

    infos = {'obj_lss': obj_lss, 'x_zs': x_zs, 'u_norms': u_norms,
             'times': times, 'alpha':alpha, 'lambda_l1':lambda_l1,
             'max_iter':max_iter, 'solver_tol':solver_tol}


    return (x, z, u, infos)

def extract_mean(invhead, testset):
    lastLayerIndex = 12
    sample = next(iter(testset.take(1)))[0]
    print("sample.shape",sample.shape)

    sample = sample*0.0
    lastLayer = invhead.get_layer(index=lastLayerIndex)
    mean = lastLayer([sample])[0]
    print("mean.shape",mean.shape)
    
    return mean

"""Run the solver"""
def run_solver_single():
    problem='inpaint_center'#'sr'
    print('problem', problem)
    ori_img = next(iter(testset.shuffle(1000)))[0]
    show_img_progress = True#False#
    # No noise alpha = 0.1, lambda=0.0005 seems to work well (in painting problems, at least)
    # Noise = 0.1, alpha = 0.3, lambda = 0.0015, or alpha = 0.6, lambda = 0.003 seem to work about the same
    # Super resolution, no noise settings seems OK
    # Compressive Sensing, 0.1 and 0.005 worked quite well
    alpha = 0.1
    max_iter = 100
    solver_tol = 1e-5
    alpha_update_ratio = 1.0
       
    alpha_l1 = 0.3
    lambda_l1 = 0.0000001
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4
       
    box_size = int(0.3 * ori_img.shape[1])#blockwise - 0.3*shape[1], scattere - 0.1*shape[1]
    noise_std = 0.0

    results = solve_inpaint_center(ori_img, reshape_img, box_size=box_size, noise_std=noise_std, alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol, problem=problem)#'inpaint_center')#

"""Run the solver on all images in testset and calculate results"""
def run_solver_all(head, invhead, mean, testset, problem):
    
    print("problem", problem)
    it = iter(testset.take(100))
    show_img_progress = False#True#
    alpha = 0.3
    max_iter = 100
    solver_tol = 1e-5
    alpha_update_ratio = 1.0
       
    alpha_l1 = 0.1
    lambda_l1 = 0.006
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4
       
    noise_std = 0.0
    mean_x = 0.0
    mean_z = 0.0
    sd_x = 0.0
    sd_z = 0.0
    count = 0.0
    print("alpha", alpha, "lambda_l1", lambda_l1)

    for x in it:
      ori_img = x[0]
      print("ori_img.shape", ori_img.shape, flush=True)
      box_size = int(0.1 * ori_img.shape[1])
      
      psnr_x,psnr_z = solve_inpaint_center(ori_img, reshape_img, head, invhead, mean, box_size=box_size, noise_std=noise_std, alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol, problem=problem, show_img_progress=False)
      mean_x += psnr_x
      sd_x += psnr_x*psnr_x
      mean_z += psnr_z
      sd_z += psnr_z*psnr_z
      count += 1
      print("count",count, "mean_x", mean_x, "sd_x", sd_x, "mean_z",mean_z,"sd_z",sd_z)

    mean_x /= count
    mean_z /= count
    sd_x -= count*mean_x*mean_x
    sd_z -= count*mean_z*mean_z
    sd_x /= (count-1.0)
    sd_z /= (count-1.0)

    print("mean_x",mean_x,"sd_x",sd_x)
    print("mean_z",mean_z,"sd_z",sd_z)

    
def main():
    print("python main function")
    print("importing data")
    dataset = import_data()
    #print("building model")
    #head, invhead, trainset, testset = build_model(dataset)
    #print("saving weights")
    #mean = extract_mean(invhead, testset)
    #save_weights(head, invhead, testset, 'imagenet') #celeba-190')#'imagenet-100k')#head, invhead, testset, file_name)
    #tf.keras.backend.set_floatx('float64')
    #dataset_resize = dataset.map(lambda x:[pre_process_image(x['image'])])
    
    #testset = dataset_resize.take(500)
    #trainset = dataset_resize.skip(500)
    tf.keras.backend.set_floatx('float64')
    trainset = dataset['train'].map(lambda x:[pre_process_image(x['image'])])
    testset = dataset['validation'].map(lambda x:[pre_process_image(x['image'])])
    print("loading model")
    head, invhead, mean = load_weights('imagenet', 1.0, trainset, testset)
    print("running solver")
    run_solver_all(head, invhead, mean, testset, 'scattered_inpaint')#'cs')#inpaint_pixelwise')#inpaint_center')
    
if __name__ == '__main__':
    main()
