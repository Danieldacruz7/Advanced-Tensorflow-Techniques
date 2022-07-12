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

## 3. Custom and Distributed Training:

## 4. Generative Deep Learning with Tensorflow:
- ### AutoEncoder Model Loss and Accuracy

- ### Generative Adverserial Network for Hand Creation

- ### Style Transfer Dog

- ### Variational Autoencoders on Anime Faces