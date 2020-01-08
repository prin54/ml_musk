import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

#load dataset
df = pd.read_csv("musk_csv.csv", index_col=0)

#preprocessing
#check for nan values
a=df.isnull().sum()
print(a)

#drop irrelevant columns
df.drop(['molecule_name', 'conformation_name'], axis=1, inplace=True)
X = df.drop(['class'], axis=1, inplace=False)
y = df['class']

#split into train and test data in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1, stratify=y)

#train mlp classifier
clf = MLPClassifier(hidden_layer_sizes=(500,500,500), random_state=1, max_iter=10, warm_start=True)


loss_test=[]
loss_train=[]
accuracy_train=[]
accuracy_test=[]


for epoch in range(10):
    print("epoch :",epoch+1)
    clf.partial_fit(X_train, y_train,  classes=np.unique(y))
    
    
    loss_test.append(log_loss(y_test,clf.predict(X_test)))
    loss_train.append(log_loss(y_train,clf.predict(X_train)))
    
    accuracy_train.append(accuracy_score(y_train, clf.predict(X_train)))
    accuracy_test.append(accuracy_score(y_test, clf.predict(X_test)))
    
    


#classification report
from sklearn.metrics import classification_report
target_names = ['Non-Musk', 'Musk']
print(classification_report(y_test, clf.predict(X_test), target_names=target_names))


#plot results
import matplotlib.pyplot as plt
epoch=[i+1 for i in range(10)]

plt.subplot(2,1,1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('model loss')
plt.plot(epoch, loss_train, label="train")
plt.plot(epoch, loss_test, label="test")
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.plot(epoch, accuracy_train, label="train")
plt.plot(epoch, accuracy_test, label="test")
plt.legend(loc='best')
plt.tight_layout()

# Saving our classifier
#import pickle
#with open('classifier.pickle','wb') as f:
    #pickle.dump(clf,f)
        
# Using our classifier
import pickle
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    
y_predicted = clf.predict(X_test)
from sklearn.metrics import classification_report
target_names = ['Non-Musk', 'Musk']
print(classification_report(y_test, y_predicted, target_names=target_names))
