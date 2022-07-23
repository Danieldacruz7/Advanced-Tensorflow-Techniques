# Advanced-Tensorflow-Techniques
A collection of projects focused on improving AI models using Tensorflow. 

The 4 main topics include computer vision, model layer customization, distributed training and generative adversarial networks. 

## 1. Advanced Computer Vision:

Computer vision is a branch of artificial intelligence that aims to process digital images, and derive meangingful information from those images and/or take action against those images. 

These tasks includes identifiying objects within an image, classify images, segment images or generate new images based on previous datasets. 

- ### Cat vs Dogs Saliency Maps

    Saliency maps provide an explanation for predictions in convolutional neural networks. After training the neural network on images of cats and dogs and differentiating between the two, saliency maps will highlight the areas within images that allowed the neural network to make an accurate prediction. 

<p align="center">
  <img src="./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Cat%20Saliency%20Map.png">
</p>

- ### Bounding Boxes for Bird Identification

    Bounding boxes are used to track and identify objects within an image. In this project, birds were identified within a series of images. The computer vision model is able to place a rectangular box around the bird that is identifies within an image.

    The evaluation metric used in bounding box creation is the Intersection Over Union (IOU) metric. This takes into account the groundtruth label, which is manually determined and comparing it to the predicted label after training. A perfection score is represented by 1, and complete misalignment is 0. 

    ![Bounding Boxes](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Bounding%20Boxes.png)

- ### Image Segmentation of Handwritten Digits

    Image segmentation is the process of breaking an image into multiple segments, or clusters of pixels that identify different objects within the image. In this example, the computer vision model is able to map over the handwritten digits in the image. 

    One measure of in image segmentation involves using the intersection over union (IOU). By taking the groundtruth label and the prediction label, it can be compared to return a score that indicate the amount of overlap between labels. The case where the IOU score is zero indicates a poor result and where a score of one equals a perfect result. 
    
    The second measure used in image segmenetation is the Dice score. The Dice score is very similar to the IOU score. It also ranges from 0 to 1, and takes into the account the same factors. However, the Dice score can be seen as the F1 score of image segmentation. It is important to use both methods for cross-validation for a more accurate view of model performance. 

    ![Image Segmentation](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Image%20Segmentation.png)

- ### Zombie Detector
    This project aims to place bounding boxes around zombies it has identified. Here transfer learning was used to speed up the training process and on only 5 images available for training. The Object Detection API and RetinaNet was used to build the model. 

    In building computer vision models, or AI models in general, one needs access to a lot of data in order to create a well-performing model. However, there are methods to mitigiating this problem. 

    Transfer learning is one way of resolving this issue. By taking a pre-existing model trained on similiar data, it is possible to improve model performance and reduce training time. Typically, the pre-trained model would have been trained on a larger dataset with more powerful compute resources. 
    The benefits of using the convolutions and features from another model can be transfered onto a new project.

    By doing so in this instance, we reduce the amount of data needed to create a model to detect zombies in an image. Here only 5 images were used. The original dataset included manually derived labels which is usually time consuming. The final result is an accurate model that can be retrained for other types of objects.  
    
    ![Zombie Detector](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Zombie%20detection.png)
    
## 2. Custom Models, Layers and Loss Functions:

These are extra technqiues that could be used to customize Tensorflow models. It can be be useful to create models from scratch for improved performance. On lower level, it is possible to customize the layers that can be inserted into a Tensorflow model.  
     
## 3. Custom and Distributed Training:

Training neural networks are computationally intensive. This means that without powerful GPU's, the time to train can be excessive. One way to mitigate against this problem is to use distributed training. It will speed up training to produce faster results. Distributed training involves distributing the workload across multiple GPU's, and the trained model is returned as a single entity.   

## 4. Generative Deep Learning with Tensorflow:

Generative adverserial networks (GANs) are a type of AI model that is able to reproduce images that it has been trained on. By using a discriminator and generator, the two models will compete with one another to create an image. In order to improve the quality of a GAN, large datasets and computational power is required. GANs are computationally intensive to produce and powerful GPU's are necessary to create high quality images. 

- ### AutoEncoder Model Loss and Accuracy

Autoencoders are a type of artificial neural network that is able to learn efficient coding of unlabelled data, and then validate that its results by reproducing the original data. The autoencoder is made up of an encoder layer and decoder layer.  

- ### Generative Adverserial Network for Hand Creation

  In this project, a typical GAN is built. Using a dataset that is includes relatively small images that make it easier for the GAN to reproduce. The dataset includes sign language images. The results can be seen below:   

    ![GAN Hands](./Generative%20Deep%20Learning%20with%20TensorFlow/Images/GAN%20Hands.png)

- ### Style Transfer GAN
    Style transfer is a technique used in to create a GAN that will take elements of one image and add it the another image. Below are two images. The first image is the image that will be altered using the overall style of the second image. 

    ![Dog](./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Dog.png)

    The results can be seen below. The image is clearly of the dog but it has been altered to include the style of the painting including the colours and textures. 

<p align="center">
  <img src="./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Style%20transfer%20dog.png">
</p>

- ### Variational Autoencoders on Anime Faces
    A Variational Autoencoders is an architecture that uses an encoder and decoder that is used to reduce the reconstruction error between the initial data and the created data. 

    In this example, a set of images using the [Anime Face dataset](https://github.com/bchao1/Anime-Face-Dataset). The initial dataset can be seen here:

<p align="center">
  <img src="./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Anime%20data.png">
</p>

The results can be seen here:

<p align="center">
  <img src="./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Anime%20Faces.png">
</p>
