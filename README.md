# TrafficSignDetection
Traffic Sign Detection using Transfer Learning to replicate the ResNet-34 model

**Objective **
To create a hybrid model that utilises transfer learning to copy the architecture of ResNet-34 to improve the accuracy of traffic signal recognition.

Road architecture and traffic sign identification are essential parts of the advanced driver assistance system. The world has evolved and become more sophisticated, and traffic sign recognition system algorithms have been put forward. This study aims to create a hybrid model that utilises transfer learning to copy the architecture of ResNet-34 to improve the accuracy of traffic signal recognition using the German Traffic Sign Recognition Benchmark (GTSRB) which contains 43 recognised classes, and 51,839 images of traffic signs are included in the dataset. This research developed a transfer learning-based ResNet-34 model that gradually reduces the amount of training data needed for traffic sign recognition and implementation. 

ResNet-34 is contrasted with alternative deep learning or machine learning models, based on the relatively lower computing costs in comparison to ResNet-50, VGG16, VGG19, YOLO, and the rest. Furthermore, a CNN has been proposed suggested, in which numerous convolutions and pooling processes were used to extract features layer-wise; these processes were then analysed and compared. Lastly, utilising feature engineering of the pre-trained model and fine-tuning parameters at various learning rates, the transfer learning-based model is repeatedly retrained. The results show that the ResNet-34 CNN algorithm can achieve a recognition rate of up to 100 percent for traffic sign recognition when it is used in conjunction with the learning rate of the optimizer scheduler. 

The report on categorisation and evaluation includes recall, accuracy, and F-1 score in addition to several other elements designed for examination and analysis. Roadside amenities and markings on the pavement are examples of the various types of traffic infrastructure that this research may help to identify. The code was run and deployed Google Collab based with major dependencies namely; Pytorch’s torch and torchvision.  


**Methodology**
In this investigation, a vast collection of German traffic signs was used. The dataset contains a broad array of traffic sign images that represent shapes, classes, colours, and environmental circumstances. 
The image dataset was cross-validated or split into 80% for training (sample of 4920), 10% for testing (sample of 615), and 10% for validation (sample of 615). The testing and validation datasets were set up for evaluating accuracy while training was used to train and learn the ResNet-34 Convolutional Neural Network (CNN) model.


**Pre-processing Methods**
Before training the models, the data was imported, and 43 classes were explored while also augmenting the set of data to account for differences that may occur in real-world circumstances (See the Input and Output Sizes of ResNet-34 Architecture in Appendix 1).

The applicable preprocessing techniques include: 
a.  Resizing of Image - The images are downsized to a preset input size that is appropriate for the model. Resizing guarantees that input dimensions are compatible and consistent throughout training and evaluation. 

b. Class imbalance – Exactly four most used signs were selected to be used 
![image](https://github.com/user-attachments/assets/81416a49-df36-4eed-851f-ee4507c7f43d)

Priority_road has class instances of 2100, give_way churned out 2160 while stop and no_entry churned out 780 and 1110 respectively. This represents a class imbalance issue, which necessitated an image augmentation technique to patch the size of the training dataset.

c. Data augmentation - These techniques include horizontal flips and rotations and random resizing. This artificially increases the size of the training dataset.

d. Normalisation - The images' pixel values are normalised to a specified range to ensure uniformity and convergence during training. This was done by using means and standard deviation (std_nums), in the form of a z-score;

**Pre-trained models**
A pre-trained model in deep learning and neural networks has been trained on a large benchmark dataset, like the GTSRB dataset, and will be fine-tuned for specific issues.
ResNet - By implementing bypasses that allow data to move straight between layers, ResNet addresses the disappearing gradient issue. This makes it possible to train extraordinarily deep networks—networks with more than a hundred layers—while preserving or even improving performance.

Transfer Learning - ResNet-34 CNN was implemented using the principle of transfer learning. The idea behind transfer learning is to apply a model that has already been trained for one job to another that is similar. Put another way, a model that has been trained for one task is applied to another that is similar. We used three pre-trained CNN architectures, ResNet-34, in this work. The GTSRB dataset (Canese et al. 2022; Youssof, 2022; Triki, Karray, and Ksantini, 2023) was used to train these architectures for the task of categorising 50,000+ different image classes (see Appendix 1). After that, these designs were adjusted and retrained to identify 43 different kinds of traffic signs. 32-piece batch sizes were used to separate the image. The learning rate scheduler was used to adjust the learning rate during training as well as three helper functions that use OpenCV and Torchvision to load and show images.

ImageNet - ResNet-34 was benchmarked using the ImageNet dataset, in other words for traffic sign classification.

Figure 1. Flowchart of the Approach
![image](https://github.com/user-attachments/assets/ef79fb58-5c6c-4e70-a28a-435f5df74747)

In sum, leveraging Torchvision, this study used transfer Learning to copy the architecture of ResNet-34 (See Appendix 1). In addition to using the learned weights of the model from training on the ImageNet dataset.

**Dataset**

The GTSRB dataset was compiled by the Institute of Neural Information Processing (INI) at the University of Ulm, Germany (Stallkamp et al. 2011). This dataset's techno-economic parameters might refer to some connected factors, including:

•	Size: The dataset includes 43 traffic sign classes (see Figure 10) (Stallkamp et al. 2011) with 51,839 images (34,799) for training, 12,640 for testing, and 4410 for validation, taken from real-world traffic scenes.
•	Image resolution: The GTSRB collection includes resolution of; 15 × 15 to 250 × 250 pixels;
•	Annotation: The dataset was annotated by human experts, allowing for the cautious labeling of the images;
•	Diversity: The dataset consists of several traffic sign classes including prohibition signs, warnings, and speed restrictions;
•	Collection method: The dataset was acquired using a camera mounted on vehicles in transit, which allowed for different situation captures (including lighting and blur);
•	Availability: The dataset is free and can be accessed for business and academic research;
•	Usage: The dataset can be used to study and assess traffic sign recognition using a variety of deep learning models;

Figure 2. The 43 Classes in the Dataset
![image](https://github.com/user-attachments/assets/b31c4195-4b1f-466b-9bf7-06c58fc5027c)




