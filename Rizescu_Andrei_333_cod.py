import numpy as np
import pandas as pd
import glob
from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.tests.test_seq_dataset import sample_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#am folosit libraria pandas, cu ajutorul careia am parsat fisierele .CSV. Primul pas a fost crearea unei liste ce contine
# cele 9000 de seturi de coordonate, salvate sub forma unei liste, care in cele din urma a fost salvata sub forma de vector.
# Avand in vedere ca toti clasificatorii aveau nevoie de un vector bi-dimensional pentru metoda fit, a trebuit ca cele 450 de
# coordonate sa fie salvate sub forma unui singur vector. Astfel, fiecare exemplu din cele 9000 avea o lista de feature-uri de
# 450 (am ales 405, pentru ca toate datele aveau minim 405 coordonate).

path = r'C:\Users\Andrei\PycharmProjects\untitled\Train' 
all_files = glob.glob(path + "/*.csv")

path = r'C:\Users\Andrei\PycharmProjects\untitled\Test'
all_test = glob.glob(path + "/*.csv")
li = []   # lista train
li2 = []
liTEST = [] # lista test

i = 0
for filename in all_files:
     # i+=1
     # if i <= 7500:

         df = pd.read_csv(filename, index_col=None, header=0)    #citirea fiecarui csv in parte si adaugarea coordonatelor sale in lista
         li.append(df)
      #else:
         # dfTEST = pd.read_csv(filename, index_col=None, header=0)
         # liTEST.append(dfTEST)

for filename in all_test:
     df2 = pd.read_csv(filename, index_col=None, header=0)     #citirea label-urilor
     li2.append(df2)



train_labels = pd.read_csv(r'C:\Users\Andrei\PycharmProjects\untitled\train_labels.csv')
test_labels = pd.read_csv(r'C:\Users\Andrei\PycharmProjects\untitled\sample_submission.csv')

arr = np.array(test_labels)
t = []

for idx in arr:
    t.append(idx[0])
t2 = np.array(t)  #vector pentru id-uri

n_samples = len(li)
v = np.array(train_labels)
l = []
lTEST=[]
i=0

for idx in v:
      #i+=1
      #if i <=7500:

         l.append(idx[1])
      #else:
          #lTEST.append((idx[1]))

v2 = np.array(l)

#Am folosit GridSearch pentru a gasi cei mai buni parametri pentru SVM-ul folosit. Mai jos evem params_grid, ce contine
# valori pentru C, gamma si tipul de kernel.
#Folosind acest grid, a rezultat faptul ca SVM-ul cel mai potrivit pentru cazul de fata este cel cu kernel= poly.
#Urmatorul pas a fost sa fac GridSearh pe degree, pentru a obtine cel mai bun rezultat.

params_grid = {'C': [0.001, 0.01, 1, 10, 100, 100],
           'gamma': [0.0001, 0.001, 0.01, 0.1],
           'kernel':['rbf','poly']}

#classifier = GridSearchCV(SVC(class_weight='balanced'), params_grid)



#Support vector machine

classifier = svm.SVC(kernel='poly', degree=21, C=100,class_weight='balanced', gamma=0.002)  #clasificatorul cu cea mai buna performanta

#Mai jos sunt alti clasificatori testati, care nu au obtinut rezultate satisfacatoare, nefiind potriviti pentru problema data
#classifier = RandomForestClassifier(n_estimators=350, max_depth=70)
#classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
#classifier = DecisionTreeClassifier(random_state=1234)
#classifier = LogisticRegression(C=100)

ll = []

for l in li:
    v3 = np.array(l)
    v3 = v3.flatten()
    aux = []
    o = 0
    for i in v3:
        if o == 405:
            break
        o+=1

        aux.append(i)
    ll.append(aux)

v4 = np.array(ll)  #vectorul de train

scores = cross_val_score(classifier, v4, v2, cv=3) #cross-validation pentru k=3 pe multimea de antrenare. Rezultatele sunt stocate in scores

print(scores.mean()) #media de acuratete pentru 3-fold cross-validation


#Am realizat un cross-validation manual, folosind metoda KFold, pentru a afisa la fiecare fold matricea de confuzie asociata

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(v4):

   X_train, X_test = v4[train_index], v4[test_index]
   y_train, y_test = v2[train_index], v2[test_index]

   classifier.fit(X_train, y_train)
   print (confusion_matrix(y_test, classifier.predict(X_test)))



classifier.fit(v4, v2)

lll =[]


for l in li2:   #li2 for submission, liTest for test
    v7 = np.array(l)
    v7 = v7.flatten()
    aux = []
    o = 0
    for i in v7:
        if o == 405:
            break
        o += 1
        aux.append(i)
    lll.append(aux)

v6 = np.array(lll)   #vectorul de test



predicted = classifier.predict(v6)


#Mai jos urmeaza salvarea datelor in format csv. kk este lista ce contine perechile de tip (id,class).
#Id-urile sunt selectate din vectorul t2 calculat mai sus, iar clasa provine din rezultatul predicitiei oferite de clasificator

kk = []
for i in range(5000):
    k=[]
    k.append(t2[i])
    k.append(predicted[i])
    kk.append(k)



import csv
with open('submission.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['id','class'])
    writer.writerows(kk)

csvFile.close()





#print(metrics.accuracy_score(predicted,lTEST))






