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
"glove.6B.50d.txt" is the word embedding I've used in the project, and it is already trained on large datasets by Glove. It transforms every word index into a 50-dimensional embedding vector. You have to download this file from [here](https://www.kaggle.com/watts2/glove6b50dtxt?select=glove.6B.50d.txt) and put in "glove" folder.

### You can see the model architecture in the figure below:
![Screenshot (492)](https://user-images.githubusercontent.com/85555218/141694766-853ec5b1-6e79-4e9a-bbe3-c96cdf754862.png)
![Screenshot (494)](https://user-images.githubusercontent.com/85555218/141694770-50ec6e54-477b-4530-80c7-e60a33717e5b.png)

## Inspiration
Coursera course by Andrew Ng
