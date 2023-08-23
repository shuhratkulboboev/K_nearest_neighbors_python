import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df=pd.read_csv("teleCust1000t.csv")

df["custcat"].value_counts()
# we can check information using graph
df.hist(column="income",bins=50)

x=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
y=df[["custcat"]].values

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=4)

x_train_norm=preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float))
x_test_norm=preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))

#KNN
#k=4
#neigh=KNeighborsClassifier(n_neighbors=k).fit(x_train_norm,y_train)

#prediction
#prediction=neigh.predict(x_test_norm)

#Accuaracy evaluation

#result_1=metrics.accuracy_score(y_train,neigh.predict(x_train))
#result_2=metrics.accuracy_score(y_test, prediction)
#In order to find the most suitable K ,we put all quantities of K which is high accuracy
Ks=10
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train_norm,y_train)
    prediction=neigh.predict(x_test_norm)
    mean_acc[n-1]=metrics.accuracy_score(y_test,prediction)
    std_acc[n-1]=np.std(prediction==y_test)/np.sqrt(prediction.shape[0])
#Ploting all results to find the most fixed one 
plt.plot(range(1,Ks),mean_acc,"g")
plt.fill_between(range(1,Ks),mean_acc-1*std_acc,mean_acc+1*std_acc,alpha=0.10,color="red")
plt.fill_between(range(1,Ks),mean_acc-3*std_acc,mean_acc+3*std_acc,alpha=0.10,color="green")
plt.legend(("Accuracy","+/-1xstd","+/-3xstd"))
plt.xlabel("Number of neighbors,K")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)







