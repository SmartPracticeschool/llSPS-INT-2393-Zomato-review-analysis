import numpy as np
import pandas as pd

dataset = pd.read_csv("zomato.csv")

data_review = dataset['reviews_list']

# x is the review 
# y is the rating

x = []
y = []

rating_str = ""
review_str = ""

data_review.shape

for row_num in range(0,51717):
    lst = data_review[row_num].split("('")
    for i in lst:
        if len(i) > 5:
            if i.find("',") != -1:
                single_rev = i.split("',")
                if len(single_rev[0]) > 2:
                    x.append(single_rev[0])
                if len(single_rev[1]) > 2:    
                    y.append(single_rev[1])
                    
#x_series = pd.Series(x)
#y_series = pd.Series(y)        
#
#myDict = {"Ratings" : x_series, "Reviews" : y_series}   
#
#new_data = pd.DataFrame(myDict)         
#
#new_data.to_csv("newFinal.csv")

                    
import re 
import nltk
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

rating_final = []
review_final = []

for loop in range(0,10000):
    data_x = x[loop]
    data_x = re.sub('[a-zA-Z]', " ", data_x)
    data_x = data_x.split()
    data_x = ''.join(data_x)
    data_x = float(data_x)
    if data_x < 2.5:
        rating_final.append("poor") #poor
    elif data_x >= 2.5 and data_x <= 3.5 :    
        rating_final.append("average") # average
    elif data_x > 3.5:
        rating_final.append("good") #good
        
        
        
        


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
rating_final = le.fit_transform(rating_final)

rating_final = np.array(rating_final)
rating_final = np.expand_dims(rating_final, axis=1)
        
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
rates = one.fit_transform(rating_final).toarray()        
                
    
    

for loop in range(0,10000) : 
    data_y = y[loop]
    data_y = re.sub('[^a-zA-Z]', " ", data_y)
    data_y = data_y.lower()
    data_y = data_y.split()
    data_y = [ps.stem(word) for word in data_y if not word in set(stopwords.words('english'))]
    data_y = ' '.join(data_y)
    review_final.append(data_y)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20000)
x_final = cv.fit_transform(review_final).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final,rates, test_size = 0.2, random_state = 0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 7390, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 3, init = 'random_uniform', activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 128,epochs = 5)

y_pred = model.predict(x_test)

text =  "The nest is one of the best bar in Whitefield.The place is absolutely mind blowing and the ambiance here is also amazing.I would recommend this place for best cocktail and mocktails which really refreshing.The food was really good and the staffs are very kind and quick as well .I would specially thanks to miss Helen for making my evening very special. "
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)

y_p = model.predict(cv.transform([text]))












    
    
    


                    
                    


        

        





