# User_Content_Text_Classification
 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


[Description](https://github.com/deenazati08/Covid19-Cases-Prediction#description) // [Result](https://github.com/deenazati08/Covid19-Cases-Prediction#result) // [Credits](https://github.com/deenazati08/Covid19-Cases-Prediction#credits)


## Description

Text documents are critical because they are one of the most valuable sources of data for businesses. Text documents frequently contain important information that can shape market trends or influence investment flows. As a result, companies frequently hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter, or newspaper articles. However, some businesses may prefer to limit their attention to articles about technology and politics. As a result, the articles must be sorted into different categories.

## Steps 

1. Data Loading:

        • Load the dataset to python by import os and pandas.

2. Data Inspection:
    
        • Check the data if there any special characters, duplicates and any other anomalities.

3. Data Cleaning:

        • RegEx are being used to remove all special characters.
        • Duplicates data were removed.
        
4. Data Preprocessing:
    
        • First, Tokenization, Padding and Truncating were applied to the features data.
        • Then, OneHotEncoder were used to transform the target data
        • The data were developed using Sequential API, Embedding, Bidirectional, LSTM, Dropout and Dense layers.
<p align="center">  
<img src="https://user-images.githubusercontent.com/120104404/208309245-96b0a044-1ca2-47a4-91ba-bbbd80d51ec3.png">
</p> 
        
        • Callbacks function also being used. (Earlystopping and TensorBoard)
        • Lastly, predict the data.

Last but not least, save all the model used.


## Result

Accuracy and Loss Graph from TensorBoard :
<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/208309499-bdca6cc2-ba16-45b2-9406-cb675df9b4f6.jpg">
<img src="https://user-images.githubusercontent.com/120104404/208309524-2c096111-1e70-417d-a24c-e607a9b93b6d.jpg">
</p>

## Discussion

As we can see from the results, this model is not particularly good. It can be improved by doing more data cleaning, such as removing stopwords and also performing stemming and lemmatization.

## Credits

- https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
