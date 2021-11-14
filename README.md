# LSTM_Projects
Implementation of some fun projects with LSTM (long short-term memory) architecture by PyTorch library

## LSTM
![LSTM](https://user-images.githubusercontent.com/85555218/141677139-527a3919-d034-48e0-8a1c-833db0550cc4.png)

## Persian name generator
In this project I've tried to generate persian names with a character-level RNN (LSTM). I've used "دیتاست-اسامی-نام-های-فارسی.csv" as my names dataset, which contains 4055 names written in Persian. <br />
This network can generate up to ten characters in a row, which means the maximum name length is 10. <br />
I've trained this network once by embedding vector method, and once by one-hot vector method. At last, for generating new names, you have to specify the first few characters and K (K is used to specify the number of top predictions in each time step, and the model randomly selects one of these predictions). <br />

You can see how to train and test the model in the figure below:
![Screenshot (487)](https://user-images.githubusercontent.com/85555218/141688507-61fa2422-9621-423f-a2e7-f8a9312b24ed.png)
![Screenshot (490)](https://user-images.githubusercontent.com/85555218/141688791-b64afb30-8f0f-4823-a89d-2cb23406351f.png)
![Screenshot (489)](https://user-images.githubusercontent.com/85555218/141688512-f7362290-0923-4522-b9c3-b61e0d30e547.png)
