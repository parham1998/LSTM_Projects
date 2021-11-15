# LSTM_Projects
Implementation of some fun projects with LSTM (long short-term memory) architecture by PyTorch library

## LSTM
![LSTM](https://user-images.githubusercontent.com/85555218/141677139-527a3919-d034-48e0-8a1c-833db0550cc4.png)

## Persian name generator
In this project I've tried to generate persian names with a character-level RNN (LSTM). I've used "دیتاست-اسامی-نام-های-فارسی.csv" as my names dataset, which contains 4055 names written in Persian. <br />
This network can generate up to ten characters in a row, which means the maximum name length is 10. <br />
I've trained this network once by embedding vector method, and once by one-hot vector method. At last, for generating new names, you have to specify the first few characters and K (K is used to specify the number of top predictions in each time step, and the model randomly selects one of these predictions). <br />

### You can see how to train and test the model in the figure below:
![Screenshot (487)](https://user-images.githubusercontent.com/85555218/141688507-61fa2422-9621-423f-a2e7-f8a9312b24ed.png)
![Screenshot (490)](https://user-images.githubusercontent.com/85555218/141688791-b64afb30-8f0f-4823-a89d-2cb23406351f.png)
![Screenshot (491)](https://user-images.githubusercontent.com/85555218/141694762-e1599a12-5646-4fc4-8282-d41b6d7b501b.png)

## Emojify
Emojify is something like emotion classification, but the difference is, we describe sentences with emojis or in better words with ❤️, ⚾, 😄, 😞 and 🍴. <br />
Such as the above project, I've used LSTM as my model and trained it once by embedding vector method and once by one-hot vector method. (using embedding vector method is essential here because vocabulary size is 400,000, and the one-hot vector method needs much more resources for training and, at last, the results for vocabulary that have not been used in training examples is worse). <br />
There are only 132 sentences to train models and 56 sentences to test models. <br />
"glove.6B.50d.txt" is the word embedding I've used in the project, and it is already trained on large datasets by Glove. It transforms every word index into a 50-dimensional embedding vector. You have to download this file from [here](https://www.kaggle.com/watts2/glove6b50dtxt?select=glove.6B.50d.txt) and put in "glove" folder.

### You can see the model architecture in the figure below:
![Screenshot (492)](https://user-images.githubusercontent.com/85555218/141694766-853ec5b1-6e79-4e9a-bbe3-c96cdf754862.png)
![Screenshot (494)](https://user-images.githubusercontent.com/85555218/141694770-50ec6e54-477b-4530-80c7-e60a33717e5b.png)

## Neural machine translation
In the last project, I've built a neural machine translation (NMT) model to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25"). <br />
I've implemented this NMT once by using an attention model and once by a simple sequence-to-sequence model. You can see the clear difference between the results of the models, and the attention weights show why this model performs much better. <br />
The model I've build here could be used to translate from one language to another, such as translating from English to Persian. (However, language translation requires massive datasets and usually takes days of training on GPUs). <br />
I generated 11000 data, 10000 for training and 1000 for testing models.

### You can see the models (attention & sequence-to-sequence) architecture in the figure below:
sequence-to-sequence
![photo_2021-11-15_12-57-40](https://user-images.githubusercontent.com/85555218/141756579-d0c66df5-c2b6-46e3-9f43-4c51dd751758.jpg)
![Screenshot (496)](https://user-images.githubusercontent.com/85555218/141756647-1d80feb8-66e1-41a3-ac63-1ec7dcea3367.png)
![Screenshot (497)](https://user-images.githubusercontent.com/85555218/141760439-45f88996-4fc3-4ade-88ee-688f3f2b07dd.png)
Attention
![Screenshot (500)](https://user-images.githubusercontent.com/85555218/141756754-ed20329b-ee24-4262-a09b-65b91d42fa0d.png)
![Screenshot (499)](https://user-images.githubusercontent.com/85555218/141760410-e6a7e125-06af-487f-b382-835bddeb29ec.png)

### attention weights for each character of the predicted output:
![attention1](https://user-images.githubusercontent.com/85555218/141697754-87f12f51-e665-480b-a817-e766047949d5.png)
![attention2](https://user-images.githubusercontent.com/85555218/141697757-371e402f-53c8-476e-a622-33454f2d4c17.png)

## Inspiration
Coursera course by Andrew Ng
