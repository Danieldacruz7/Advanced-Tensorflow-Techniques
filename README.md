# Advanced-Tensorflow-Techniques
A collection of projects focused on improving AI models using Tensorflow. 

The 4 main topics include computer vision, model layer customization, distributed training and generative adversarial networks. 

## 1. Advanced Computer Vision:

- ### Cat vs Dogs Saliency Maps

    Saliency maps provide an explanation for predictions in convolutional neural networks. After training the neural network on images of cats and dogs and differentiating between the two, saliency maps will highlight the areas within images that allowed the neural network to make an accurate prediction. 

    ![Cat Saliency Map](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Cat%20Saliency%20Map.png)

- ### Bounding Boxes for Bird Identification

    Bounding boxes are used to track and identify objects within an image. In this project, birds were identified within a series of images. The computer vision model is able to place a rectangular box around the bird that is identifies within an image. 
    ![Bounding Boxes](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Bounding%20Boxes.png)

- ### Image Segmentation of Handwritten Digits

    Image segmentation is the process of breaking an image into multiple segments, or clusters of pixels that identify different objects within the image. In this example, the computer vision model is able to map over the handwritten digits in the image. 

    ![Image Segmentation](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Image%20Segmentation.png)

- ### Zombie Detector
    This project aims to place bounding boxes around zombies it has identified. Here transfer learning was used to speed up the training process and on only 5 images available for training. The Object Detection API and RetinaNet was used to build the model. 
    
    ![Zombie Detector](./Advanced%20Computer%20Vision%20with%20TensorFlow/Images/Zombie%20detection.png)
    
## 2. Custom Models, Layers and Loss Functions:

These are extra technqiues that could be used to customize Tensorflow models. It can be be useful to create models from scratch for improved performance. On lower level, it is possible to customize the layers that can be inserted into a Tensorflow model.  
     
## 3. Custom and Distributed Training:

Training neural networks are computationally intensive. This means that without powerful GPU's, the time to train can be excessive. One way to mitigate against this problem is to use distributed training. It will speed up training to produce faster results. Distributed training involves distributing the workload across multiple GPU's, and the trained model is returned as a single entity.   

## 4. Generative Deep Learning with Tensorflow:

Generative adverserial networks (GANs) are a type of AI models that are able to create or reproduce images that it has been trained on. By using a discriminator and generator, the two models will compete with one another to create an image. In order to improve the quality of a GAN, large datasets and computational power is required. GANs are computationally intensive to produce and powerful GPU's are necessary to create high quality images. 

- ### AutoEncoder Model Loss and Accuracy

- ### Generative Adverserial Network for Hand Creation

    ![GAN Hands](./Generative%20Deep%20Learning%20with%20TensorFlow/Images/GAN%20Hands.png)

- ### Style Transfer Dog
    ![Dog](./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Dog.png)

    ![Style transfer dog](./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Style%20transfer%20dog.png)

- ### Variational Autoencoders on Anime Faces

    ![Anime Faces](./Generative%20Deep%20Learning%20with%20TensorFlow/Images/Anime%20Faces.png)