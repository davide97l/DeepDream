from __future__ import print_function
import os
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

model_fn = 'models/tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)  # install itself as the default session on construction
# load the protobuf file from the disk and parse it to retrieve the un-serialized graph_def
with tf.gfile.FastGFile(model_fn, 'rb') as f:  # give access to the file containing the trained-model
    graph_def = tf.GraphDef()  # serialized version of graph
    graph_def.ParseFromString(f.read())  # read the protobuf file and generate a graph
t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)  # normalize the input image
tf.import_graph_def(graph_def, {'input': t_preprocessed})  # import a graph and eventually set some parameters


# convert np array to PIL image and return it
def get_img_from_array(img):
    a = np.uint8(np.clip(img, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def T(layer):
    """Helper for getting layer output tensor"""
    return graph.get_tensor_by_name("import/%s:0" % layer)


def tffunc(*argtypes):
    """Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below."""
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]

resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    """Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations."""
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def deepdream(t_obj, img, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # calculate the gradient

    # split the image into a number of octaves
    octaves = []
    for i in range(octave_n):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    generated_images = []  # generate one image for each octave

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            print('octave: {}, iteration: {}'.format(octave, i))

        pil_img = get_img_from_array(img / 255.0)
        generated_images.append(pil_img)

    return generated_images


# display net layers
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
print('Layers: {}'.format(layers))
print('Total number of feature channels:', sum(feature_nums))


objfuncs = {
            # layers
            'effect4-3': tf.square(T('mixed4c')),  # wolves
            'effect4-1': tf.square(T('mixed4a')),  # eyes
            'effect4-2': tf.square(T('mixed4b')),  # dogs
            'effect4-4': tf.square(T('mixed4d')),  # snakes
            'effect5-1': tf.square(T('mixed5a')),  # birds
            'effect5-2': tf.square(T('mixed4b')),  # dogs
            # filters
            'flowers': T('mixed4d_3x3_bottleneck_pre_relu')[:, :, :, 139],
            'wolves': T('mixed4d_3x3_bottleneck_pre_relu')[:, :, :, 138],
            'homes': T('mixed4d_3x3_bottleneck_pre_relu')[:, :, :, 65],
            }

# optional parameters
show = True
save = True
use_default_image = False  # random noise
# algorithm parameters
iter_n = [30]  # how many gradient ascent step per octave
octave_n = 4  # number of multiple scales of the image onto apply gradient ascent
octave_scale = 1.4  # how much resize the image with each octave
# input parameters
input_path = 'images'
output_path = 'deepdreams'
image_paths = ['tsinghua.jpg']
effect = ['effect4-1', 'effect4-2', 'effect4-3', 'effect4-4', 'flowers']

for image_path in image_paths:
    print(image_path)
    img_name = image_path.split(".")[0]
    img = 0
    if use_default_image:
        # start with a gray image with a little noise
        img = np.random.uniform(size=(499, 499, 3)) + 100.0
    else:
        img = PIL.Image.open(os.path.join(input_path, image_path))
        img = np.float32(img)

    generated_images = []
    for i, e in enumerate(effect):
        print("effect: {}".format(i))
        for n in iter_n:
            # apply deep dream algorithm
            images = deepdream(objfuncs[e], img, iter_n=n, octave_n=octave_n, octave_scale=octave_scale)
            generated_images = generated_images + images

    for i, img_ in enumerate(generated_images):
        if show:
            img_.show()
        if save:
            save_path = os.path.join(os.getcwd(), output_path, img_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img_.save(os.path.join(os.getcwd(), save_path,
                                   img_name + '_' + str(i)
                                   + '_' + str(iter_n[i % (len(iter_n))]) + '.jpg'))
