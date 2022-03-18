# Image Data: CNN and Pretraining
## _Powered by The Deep Sleeping Crew (Group6)_

### Highlights
•	1
•	2
•	3
 
## 1.Introduction
Most Thai who are Buddhists tend to bond and pay homage to Buddha images in their daily life. But few can remember and recognize them. Can you distinguish the outstanding features of the **`5 Floating Buddha Statues`** in the figure below? If not, let our model do it! These Buddha images are one of the sacred groups frequented by Thais to worship for good fortune; three of them are very similar. Therefore, this work aims to collect an image dataset of the 5 Floating Buddha Statues and to build an image classifier by using a **`CNN pre-trained on ImageNet dataset`** and transfer learning to perform **`multi-class classification`** and recognize classes of the images it was never trained on.

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/d9161d1181fe12d2ba2763718c3d16c7a12a6d4c/5%20Floating%20Buddha%20Statues.jpeg" style="width:600px;"/>

According to the legend, there once were five Buddha statues with miraculous power floating along five rivers. They were stranded and found by the local villagers, who enshrined each Buddha statue in a temple in the vicinity where they were found. 

The five Buddha images and temples are 1) **`Luang Pho Sothon (โสธร)`**, a Buddha image seated in the Dhyani pose, was found in the Bang Pakong River; 2) **`Luang Pho Toh (โต)`**, a Buddha image seated in the Bhumisparsa pose, was found at the Chao Phraya River; 3) **`Luang Pho Wat Ban Laem (วัดบ้านแหลม)`**, a Buddha image standing in the Pahng Um Baat pose, was found floating in the Mae Klong (แม่กลอง) River; 4) **`Luang Pho Wat Rai Khing (วัดไร่ขิง)`**, a Buddha image seated in the Bhumisparsa pose, was found in the Nakhon Chai Sri River; and 5) **`Luang Pho Thong Khao Ta-Khrao (ทองเขาตะเครา)`**, a Buddha image seated in the Bhumisparsa pose, was found at the Phetchaburi River.


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

In the process, all images are converted to a .png file and manually extracted into sub-folders for easy access in the next steps. Next, we resize the images by running **`tf.keras.preprocessing.image.load_img()`** function to load the images with different heights and widths into PIL format, sizing 224 x 224 pixels as CNN models expect such a target size. A PIL Image instance is then converted to a Numpy array using **`tf.keras.preprocessing.image.img_to_array()`** function, returning a 3D Numpy array (501, 224, 224, 3). Last, we also need to run the images through a preprocess input function of the model we have used, such as **`tf.keras.applications.efficientnet.preprocess_input()`** for preprocessing the NumPy array encoding a batch of images.

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/823814a4fdb8e79904e7bd55a1c93bfa84a8f834/preprocessed%20images.png" style="width:800px;"/>

Finally, I split each menu into three sets: train, valid, and test. These classes are necessary for training our model. I decided to use an 80% train, 10% valid, and 10% test formula. This meant that the train set has 240 images per one menu, valid has 30 images per one menu, and test has 30 images per one menu. That’s how I prepared my dataset.

## 3.Network Architecture
It would be impossible for us with no high computing power to train models from scratch with massive amounts of image data. However, there is now a transfer learning technique that empowers us to jump-start our CNNs with the big SOTA models and their pre-trained weights. 

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/0236b467df162ed81b1eb582e0c5629e93364ea9/CNN%20for%20image%20classification.png" style="width:600px;"/>

We have used six ImageNet-pretrained models such as VGG16, ResNet50, EfficientNetB0, etc., in this experiment to reuse lower layers of the models (feature extractor) and train it with our custom dataset only on layers of classifier due to the small dataset differing from the ImageNet dataset. (including fine-tune model)

### The best model

EfficientNetB0, one of six ImageNet-pretrained models we experiment with, performs 93.37% accurately on the test set with transfer learning no fine-tuning. We freeze the pre-trained CNN parameters to be non-trainable — we can see that we have more than 4M non-trainable parameters in our new model. 

<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/6107d576979f9c328382ab49bbcad0adf78e2921/classifier%20of%20EfficientNetB0.png" style="width:600px;"/>
The model's classifier consists of one flatten layer, five dense layers, one dropout layer with 50%, and one output layer with softmax activation, totaling 32M trainable parameters as shown in the figure below. (This also results in a shorter training time per epoch when compared to the benchmark model.)

## 4.Training
Our custom models were compiled with Adam as the optimizer, sparse_categorical_crossentropy as the loss function, and ReLU as the activation function. A GPU used for training the model was Tesla P100-PCIE-16GB in Google Colab environment, providing access to decreasing the training time within xx seconds. We have trained the model for 100 epochs with a batch size of 100. Lastly, the trained model was exported in the HDF5 file as a multi-class classifier. 


<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/51129b4ad4702386ebb0069459e5de5e1aa7c0b4/model%20accuracy.png" style="width:700px;"/>
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/51129b4ad4702386ebb0069459e5de5e1aa7c0b4/model%20loss.png" style="width:700px;"/>

### Using Pre-trained Layers for Fine-Tuning

## 5.Result
สรุปผลว่าเกิด underfit หรือ overfit หรือไม่
### Grad-Cam
### Evaluation metric
We now have predictions for models we want to compare. Below is a function for visualizing class-wise predictions in a confusion matrix using the heatmap method. This tells us how many correct and incorrect classifications each model made by comparing the true class versus the predicted class. Naturally, the larger the values down the diagonal, the better the model did.
------------figure of Evaluation metric---------------
The transfer learning model with no fine-tuning is closed to that with fine-tuning, evident from the stronger diagonal and lighter cells everywhere else. From the confusion matrix, we can also see that this model most commonly misclassifies Toh as Thong. 

### Comparing Models
<img src="https://github.com/robinoud/BADS7604_HW4_Transfer_Learning/blob/f11ed884eba456b864c9c6aa0cffdd4bda16c960/Results%20comparing%20the%206%20models%20tested.png" style="width:700px;"/>
Finally, we can compare the test metric between transfer learning (EfficientNetB0) with no fine-tuning and that with fine-tuning. The results show that the first approach, training only layers of the classifier, captured the patterns in the data more effectively, increasing accuracy to more than 93.37% in the test set. This is probably thanks to the nature of the data where the model was initially trained and how it transfers to the character domain of the Buddha images. 


## 6.Discussion
 	อภิปรายผลลัพธ์ที่ได้ว่ามีอะไรเป็นไปตามสมมติฐาน หรือมีอะไรผิดคาดไม่เป็นไปตามสมมติฐานบ้าง, วิเคราะห์เพิ่มเติมว่าสิ่งที่ผิดคาดหรือผิดปกตินั้นน่าจะเกิดจากอะไร, ในกรณีที่ dataset มีปัญหา วิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่
• pretrain ดีกว่าจูนเองทั้งหมดอย่างไร (++Different & small dataset: avoid overfitting by not fine-tuning the weights on a small dataset, and use extracted features from lower levels of the ConvNet which are more generalizable.)
•	โมเดลใหญ่ที่ parameter เยอะๆ ไม่ได้เหมาะกับทุกๆปัญหา

## 7.Conclusion
 สรุปผลของการบ้านนี้ โดยเน้นการตอบโจทย์ปัญหา (research question) หรือจุดประสงค์หลัก (objective) ของการบ้านแต่ละครั้ง
 • เป็นเรื่องง่าย ที่จะใช้ประโยชน์จาก pre-train models กับ custom dataset ด้วย TL
 • There are two main factors that will affect your choice of approach: Your dataset size
 && Similarity of your dataset to the pre-trained dataset (typically ImageNet)

In this article, we solved an image classification problem using a custom dataset using Transfer Learning. We saw that by employing various Transfer Learning strategies such as Fine-Tuning, we can generate a model that outperforms a custom-written CNN. Some key takeaways:

Transfer learning can be a great starting point for training a model when you do not possess a large amount of data.
Transfer learning requires that a model has been pre-trained on a robust source task which can be easily adapted to solve a smaller target task.
Transfer learning is easily accessible through the Keras API. You can find available pre-trained models here.
Fine-Tuning a portion of pre-trained layers can boost model performance significantly

### _The Deep Sleeping Crew (Group6) Contribution - Uniform_
**`16.67%`** 🍕 - **`6310422057`** Natdanai Thedwichienchai - **`Prepare dataset`** **`Experiment with MLP `**  **`Experiment with traditional ML`** 

**`16.67%`** 🍕 - **`6310422061`** Wuthipoom Kunaborimas - **`Prepare dataset`** **`Experiment with MLP `**  **`Experiment with traditional ML`** 

**`16.67%`** 🍕 - **`6310422063`** Nuj Lael - **`Experiment with MLP `**  **`Experiment with traditional ML`** **`Evaluate and conclude result`**

**`16.67%`** 🍕 - **`6310422064`** Krisna Pintong - **`Explore data`** **`Prepare dataset`**  **`Experiment with MLP `**

**`16.67%`** 🍕 - **`6310422065`** Songpol Bunyang - **`Explore data`** **`Experiment with MLP `** **`Evaluate and conclude result`**

**`16.67%`** 🍕 - **`6310422069`** Phawit Boonrat - **`Explore data`** **`Experiment with MLP `** **`Experiment with traditional ML`**



## End Credit
