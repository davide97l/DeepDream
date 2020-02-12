# DeepDream

Convolutional Neural Network image generation techniques implemented with TensorFlow.

Project report:
- https://davideliu.com/2019/11/03/deep-dreaming/

References:
- https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

# How to easily generate your own images
Create two new directories and name them "images" and "deepdreams" and put your images in the "images" folder.

Run this command to resize that images that were too big, you can also ignore this step but it could significantly slow down the process.
```
"python resize_jpg.py -i images/
```
Run the following command to transform the images applying deep dreaming techinques.
```
python DeepDreamGenerator.py
```
Once the computation terminates you will find your processed images in the "deepdreams" folder, images will be grouped by sub-folders and each sub-folder will contain the processed images with different effects and sizes. You should look inside the file "DeepDreamGenerator.py" to modify internal parameters and get different results.

