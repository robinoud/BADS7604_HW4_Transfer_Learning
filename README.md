# Image Data: CNN and Pretraining
## _Powered by The Deep Sleeping Crew (Group6)_

### Highlights
•	1
•	2
•	3
 
## 1.Introduction
Most Thais who are Buddhists tend to bond and pay homage to Buddha images in their daily lives. But few can remember and recognize them. Can you distinguish the outstanding features of the 5 Floating Buddha Statues in the figure below? These Buddha images are one of the sacred groups frequented by Thais to worship for good fortune; three of them are very similar. Therefore, this work aims to collect an image dataset of the 5 Floating Buddha Statues and to build an image classifier by using a CNN pre-trained on ImageNet dataset and transfer learning to perform multi-class classification and recognize classes of the images it was never trained on.

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/d9161d1181fe12d2ba2763718c3d16c7a12a6d4c/5%20Floating%20Buddha%20Statues.jpeg" style="width:600px;"/>

According to the legend, there once were five Buddha statues with miraculous power floating along five rivers. They were stranded and found by the local villagers, who enshrined each Buddha statue in a temple in the vicinity where they were found. 
The five Buddha images and temples are 1) Luang Pho Sothon (โสธร), a Buddha image seated in the Dhyani pose, was found in the Bang Pakong River; 2) Luang Pho Toh (โต), a Buddha image seated in the Bhumisparsa pose, was found at the Chao Phraya River; 3) Luang Pho Wat Rai Khing (วัดไร่ขิง), a Buddha image seated in the Bhumisparsa pose, was found in the Nakhon Chai Sri River; 4) Luang Pho Wat Ban Laem (วัดบ้านแหลม), a Buddha image standing in the Pahng Um Baat pose, was found floating in the Mae Klong (แม่กลอง) River; and 5) Luang Pho Thong Khao Ta-Khrao (ทองเขาตะเครา), a Buddha image seated in the Bhumisparsa pose, was found at the Phetchaburi River.


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

### Data pre-processing and splitting

In the process, all images are converted to a .png file and manually extracted into sub-folders for easy access in the next steps. Next, we resize the images by running tf.keras.preprocessing.image.load_img() function to load the images with different heights and widths into PIL format, sizing 224 x 224 pixels as CNN models expect such a target size. A PIL Image instance is then converted to a Numpy array using tf.keras.preprocessing.image.img_to_array() function, returning a 3D Numpy array (501, 224, 224, 3). Last, we also need to run the images through a preprocess input function of the model we have used, such as tf.keras.applications.vgg16.preprocess_input() for preprocessing the NumPy array encoding a batch of images.

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/823814a4fdb8e79904e7bd55a1c93bfa84a8f834/preprocessed%20images.png" style="width:800px;"/>

## 3.Network Architecture
It would be impossible for us with no high computing power to train models from scratch with massive amounts of image data. However, there is now a transfer learning technique that empowers us to jump-start our CNNs with the big SOTA models and their pre-trained weights. 

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/0236b467df162ed81b1eb582e0c5629e93364ea9/CNN%20for%20image%20classification.png" style="width:600px;"/>

We have used six ImageNet-pretrained models such as VGG16, ResNet50, 
EfficientNetB0, etc., in this experiment to reuse lower layers of the models (feature extractor) and train it with our custom dataset only on layers of classifier due to the small dataset differing from the ImageNet dataset. 

### The best model

