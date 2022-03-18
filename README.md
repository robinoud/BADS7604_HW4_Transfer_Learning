# Image Data: CNN and Pretraining
## _Powered by The Deep Sleeping Crew (Group6)_

## 1.Introduction
Most Thai, who are Buddhists, tend to bond and pay respect to Buddha images in their daily life; however, only a few people can remember and recognize the details of Buddha images. Thus, the question has been raised 'Can you distinguish the outstanding features of the **`Five Floating Buddha Statues`** in the figure below?' If it is not, let our model do it! 

These Buddha images are one of the religious groups frequented by Thai to worship for good fortune; three are very similar. Therefore, this work aims to collect an image dataset of the five 'Floating Buddha Statues'. Then, to create an image classifier using **`CNN pre-trained on ImageNet dataset`**, transfer learning to perform **`multi-class classification`** and recognize classes of the images that were never trained before.

<p align="center">
  <img width="650" src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/d9161d1181fe12d2ba2763718c3d16c7a12a6d4c/5%20Floating%20Buddha%20Statues.jpeg">
</p>



According to the legend, there once were Five Buddha statues with miraculous power floating along five rivers. They were stranded and found by the local villagers, who enshrined each Buddha statue in a temple in the vicinity where they were found. 

The five Buddha images and temples are 1) **`Luang Pho Sothon (โสธร)`**, a Buddha image seated in the Dhyani pose, was found in the Bang Pakong River; 2) **`Luang Pho Toh (โต)`**, a Buddha image seated in the Bhumisparsa pose, was found at the Chao Phraya River; 3) **`Luang Pho Wat Ban Laem (วัดบ้านแหลม)`**, a Buddha image standing in the Pahng Um Baat pose, was found floating in the Mae Klong (แม่กลอง) River; 4) **`Luang Pho Wat Rai Khing (วัดไร่ขิง)`**, a Buddha image seated in the Bhumisparsa pose, was found in the Nakhon Chai Sri River; and 5) **`Luang Pho Thong Khao Ta-Khrao (ทองเขาตะเครา)`**, a Buddha image seated in the Bhumisparsa pose, was found at the Phetchaburi River.


## 2.Dataset
### Data source
Since the Five Floating Buddha Statues are mostly rare art objects belonging to personal or family property, the set of images cannot be collected by photographing itself. So various sources on the internet would be the solution now, especially the Thai amulet websites being like a gold mine, filled with Buddha images in good condition which clearly represent their details and patterns.


| Class Code No.| Thai Name | English Name |
| :------: | ------ | ------ | 
| 0 | หลวงพ่อโสธร | Sothon |
| 1 | หลวงพ่อโต | Toh | 
| 2 | หลวงพ่อวัดบ้านแหลม| Wat Ban Laem | 
| 3 | หลวงพ่อวัดไร่ขิง | Wat Rai Khing | 
| 4 | หลวงพ่อทอง| Thong | 

We decided that each class of Buddha statues would approximately have 100 images because reusing the lower layers of a  pre-trained model for transfer learning enables us to have a small dataset different from the ImageNet dataset.

Link to download the dataset: https://drive.google.com/drive/folders/1JzbkJWOOQNzhYEDNOGgOVJYv1KHgB7xU?usp=sharing

### Data pre-processing and splitting

In the process, all images are converted to a .png file and manually extracted into sub-folders for easy access in the next steps. Next, we resize the images by running **`tf.keras.preprocessing.image.load_img()`** function to load the images with different heights and widths into PIL format, sizing 224 x 224 pixels as CNN models expect such a target size. A PIL Image instance is then converted to a Numpy array using **`tf.keras.preprocessing.image.img_to_array()`** function, returning a 3D Numpy array (501, 224, 224, 3). Last, we also need to run the images through a preprocess input function of the model we have used, such as **`tf.keras.applications.efficientnet.preprocess_input()`** for preprocessing the NumPy array encoding a batch of images.

<p align="center">
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/786e5b558be0f610d95958e3cbe30c0b0b70fc31/preprocessed%20five%20Buddha%20images.jpg" style="width:800px;"/>
 </p>

Finally, we split each Buddha image into three sets: train, valid, and test. These classes are necessary for training our model. We decided to use an 53.6% train, 13.4% valid, and 33% test formula. 

## 3.Network Architecture
It would be impossible for us with no high computing power to train models from scratch with massive amounts of image data. However, there is now a transfer learning technique that empowers us to jump-start our CNNs with the big SOTA models and their pre-trained weights. 

<p align="center">
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/0236b467df162ed81b1eb582e0c5629e93364ea9/CNN%20for%20image%20classification.png" style="width:600px;"/></p>

We have used six ImageNet-pretrained models such as VGG16, ResNet50, EfficientNetB0, etc., in this experiment to reuse lower layers of the models (feature extractor) and train it with our custom dataset only on layers of classifier due to the small dataset differing from the ImageNet dataset. (including fine-tune model)

### The best model

EfficientNetB0, one of six ImageNet-pretrained models we experiment with, performs 93.37% accurately on the test set with transfer learning no fine-tuning. We freeze the pre-trained CNN parameters to be non-trainable — we can see that we have more than 4M non-trainable parameters in our new model. 

<p align="center">
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/6107d576979f9c328382ab49bbcad0adf78e2921/classifier%20of%20EfficientNetB0.png" style="width:600px;"/></p>
  
The model's classifier consists of one flatten layer, five dense layers, one dropout layer with 50%, and one output layer with softmax activation, totaling 32M trainable parameters as shown in the figure below. (This also results in a shorter training time per epoch when compared to the benchmark model.)

## 4.Training
Our custom models were compiled with Adam as the optimizer, sparse_categorical_crossentropy as the loss function, and ReLU as the activation function. A GPU used for training the model was Tesla P100-PCIE-16GB in Google Colab environment, providing access to decreasing the training time within xx seconds. We have trained the model for 100 epochs with a batch size of 100. Lastly, the trained model was exported in the HDF5 file as a multi-class classifier. 

<p align="center">
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/25a8ad01c49e30cb4039854c3704e2103585b198/asset/model%20acc%20&%20loss.jpeg" style="width:700px;"/></p>

### Using Pre-trained Layers for Fine-Tuning

## 5.Result

### Evaluation metric
We now have predictions for models we want to compare. Below is a function for visualizing class-wise predictions in a confusion matrix using the heatmap method. This tells us how many correct and incorrect classifications each model made by comparing the true class versus the predicted class. Naturally, the larger the values down the diagonal, the better the model did.

  <p align="center">
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/690cd86f7236386be1827258f930775a702b185a/asset/Evaluation%20metric.jpeg" style="width:700px;"/></p>

From the confusion matrix, the performance of the transfer learning model with no fine-tuning is closed to that with fine-tuning, evident from the stronger diagonal and lighter cells everywhere else. We can also see that this model most commonly misclassifies Thong as Sothon.  

### Comparing Models

<p align="center">
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/457f96aeee9d94bb14956d87a061425c89bdd828/asset/Results%20comparing%20the%206%20models%20tested.png" style="width:700px;"/></p>

Finally, we can compare the test metric between transfer learning with no fine-tuning and that with fine-tuning. The results show that the first approach with EfficientNetB0 architecture, training only layers of the classifier, captured the patterns in the data more effectively, increasing accuracy to 93.37% in the test set. This is probably thanks to the nature of the data where the model was initially trained and how it transfers to the character domain of the Buddha images. 


## 6.Discussion
 	
• Surprisingly, Transfer learning, training only a classifier for the new dataset, classifies data better than fine-tuning, replacing and retraining the classifier, and then fine-tuning the weights of the pre-trained network via backpropagation. However, recall that pre-trained on ImageNet dataset has been trained on millions of images, including xxx images. Its convolutional layers and trained weights can detect generic features such as edges, colors, etc.

• In this experiment, we find that using some higher model architectures requiring computational power does not guarantee to work best with every dataset.
On the other hand, EfficientNetB0 architecture with the least complexity outperforms the image dataset with lesser size.

## 7.Conclusion

 In this study, we solved an image classification problem with our custom dataset using transfer learning and fine-tuning. Transfer learning can be a great starting point for training a model when not possessing a large amount of data. It requires that a model has been pre-trained on a robust source task which can be easily adapted to solve a smaller target task like classifying the Buddha images.
Moreover, collecting our own set of images that cannot be classified with models pre-trained on ImageNet makes us think deeply about how ConvNet works with an image and how we handle the data before passing it through the model's layers.

## End Credit
 This study is a part of **`Deep Learning course`**  (BADS7604), Businuss Analytics and Data Science, National Institute of Development Admistration (**`NIDA`**)

### _The Deep Sleeping Crew (Group6) Contribution - Uniform_
**`16.67%`** 🍕 - **`6310422057`** Natdanai Thedwichienchai Collect data (Wat Ban Laem)+ train efficient net model

**`16.67%`** 🍕 - **`6310422061`** Wuthipoom Kunaborimas Collect data (Thong)+ train vgg16 model

**`16.67%`** 🍕 - **`6310422063`** Nuj Lael Collect data (Wat Rai Khing)+ train inception_v3

**`16.67%`** 🍕 - **`6310422064`** Krisna Pintong Collect data (Sothorn)+ train Mobile NetV2

**`16.67%`** 🍕 - **`6310422065`** Songpol Bunyang Collect data (Toh)+ train DenseNet121

**`16.67%`** 🍕 - **`6310422069`** Phawit Boonrat tranform data into same format data + train ResNet50




