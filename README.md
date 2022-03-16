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
Since the 5 Floating Buddha Statues are mostly rare art objects of personal or family property, the dataset cannot be collected by photographing itself. So the Internet is the best source of information, especially the Thai amulet websites as a gold mine, filled with vivid images as most of the audience need to see their details and patterns.

| Class Code No.| Thai Name | English Name |
| :------: | ------ | ------ | 
| 0 | หลวงพ่อโสธร | Sothon |
| 1 | หลวงพ่อโต | Toh | 
| 2 | หลวงพ่อวัดบ้านแหลม| Wat Ban Laem | 
| 3 | หลวงพ่อวัดไร่ขิง | Wat Rai Khing | 
| 4 | หลวงพ่อทอง| Thong | 

