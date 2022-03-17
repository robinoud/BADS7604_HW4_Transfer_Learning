# Image Data: CNN and Pretraining
## _Powered by The Deep Sleeping Crew (Group6)_

### Highlights
•	1
•	2
•	3
 
## 1.Introduction
Most Thais who are Buddhists tend to bond and pay homage to Buddha images in their daily lives. But few can remember and recognize them. Can you distinguish the outstanding features of the **`5 Floating Buddha Statues`** in the figure below? These Buddha images are one of the sacred groups frequented by Thais to worship for good fortune; three of them are very similar. Therefore, this work aims to collect an image dataset of the 5 Floating Buddha Statues and to build an image classifier by using a **`CNN pre-trained on ImageNet dataset`** and transfer learning to perform **`multi-class classification`** and recognize classes of the images it was never trained on.

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/d9161d1181fe12d2ba2763718c3d16c7a12a6d4c/5%20Floating%20Buddha%20Statues.jpeg" style="width:600px;"/>

According to the legend, there once were five Buddha statues with miraculous power floating along five rivers. They were stranded and found by the local villagers, who enshrined each Buddha statue in a temple in the vicinity where they were found. 

The five Buddha images and temples are 1) **`Luang Pho Sothon (โสธร)`**, a Buddha image seated in the Dhyani pose, was found in the Bang Pakong River; 2) **`Luang Pho Toh (โต)`**, a Buddha image seated in the Bhumisparsa pose, was found at the Chao Phraya River; 3) **`Luang Pho Wat Rai Khing (วัดไร่ขิง)`**, a Buddha image seated in the Bhumisparsa pose, was found in the Nakhon Chai Sri River; 4) **`Luang Pho Wat Ban Laem (วัดบ้านแหลม)`**, a Buddha image standing in the Pahng Um Baat pose, was found floating in the Mae Klong (แม่กลอง) River; and 5) **`Luang Pho Thong Khao Ta-Khrao (ทองเขาตะเครา)`**, a Buddha image seated in the Bhumisparsa pose, was found at the Phetchaburi River.


## 2.Dataset
### Data source
Since the 5 Floating Buddha Statues are mostly rare art objects belonging to personal or family property, the set of images cannot be collected by photographing itself. So various sources on the internet would be the solution now, especially the Thai amulet websites being like a gold mine, filled with Buddha images in good condition which clearly represent their details and patterns.

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

In the process, all images are converted to a .png file and manually extracted into sub-folders for easy access in the next steps. Next, we resize the images by running **`tf.keras.preprocessing.image.load_img()`** function to load the images with different heights and widths into PIL format, sizing 224 x 224 pixels as CNN models expect such a target size. A PIL Image instance is then converted to a Numpy array using **`tf.keras.preprocessing.image.img_to_array()`** function, returning a 3D Numpy array (501, 224, 224, 3). Last, we also need to run the images through a preprocess input function of the model we have used, such as **`tf.keras.applications.vgg16.preprocess_input()`** for preprocessing the NumPy array encoding a batch of images.

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/823814a4fdb8e79904e7bd55a1c93bfa84a8f834/preprocessed%20images.png" style="width:800px;"/>

## 3.Network Architecture
It would be impossible for us with no high computing power to train models from scratch with massive amounts of image data. However, there is now a transfer learning technique that empowers us to jump-start our CNNs with the big SOTA models and their pre-trained weights. 

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/0236b467df162ed81b1eb582e0c5629e93364ea9/CNN%20for%20image%20classification.png" style="width:600px;"/>

We have used six ImageNet-pretrained models such as VGG16, ResNet50, EfficientNetB0, etc., in this experiment to reuse lower layers of the models (feature extractor) and train it with our custom dataset only on layers of classifier due to the small dataset differing from the ImageNet dataset. (including fine-tune model)

### The best model

EfficientNetB0, one of six ImageNet-pretrained models we experiment with, performs 93.37% accurately on the test set with transfer learning no fine-tuning. We freeze the pre-trained CNN parameters to be non-trainable — we can see that we have more than 4M non-trainable parameters in our new model. The model's classifier consists of one flatten layer, five dense layers, one dropout layer with 50%, and one output layer with softmax activation, totaling 32M trainable parameters as shown in the figure below. (This also results in a shorter training time per epoch when compared to the benchmark model.)

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/6107d576979f9c328382ab49bbcad0adf78e2921/classifier%20of%20EfficientNetB0.png" style="width:600px;"/>

## 4.Training
Our custom models were compiled with Adam as the optimizer, sparse_categorical_crossentropy as the loss function, and ReLU as the activation function. A GPU used for training the model was Tesla P100-PCIE-16GB in Google Colab environment, providing access to decreasing the training time within xx seconds. We have trained the model for 100 epochs with a batch size of 100. Lastly, the trained model was exported in the HDF5 file as a multi-class classifier. 


