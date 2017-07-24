import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import normalize

#main dataframe 
datadf = pd.read_csv('medical_records.data',header=None)

dfnp = datadf.as_matrix()
dflabel = datadf[[1]].as_matrix()
#dfl = datadf[[1]].values

dfnp_n = normalize(dfnp[:,[2,3,4,5,13,14,15,22,23,24,25]],axis=0)


dfnp_mod = np.concatenate((dfnp[:,[6,7,8,9,10,11,12,16,17,18,19,20,21,26,27,28,29,30]],dfnp_n),axis=1)
#print(dfnp)
#print(dflabel.type)
#print(dfl)

# class labels converted to 0,1
t,datalabels = np.unique(dflabel,return_inverse=True)

#testdata = np.array_split(dfnp,3)

#print(testdata)

#x[1:4,:]

#training data as half of the entire dataset
trainingdata = dfnp_mod[1:258,:] 
#drop the diagnosis (class label) column
#trainingdata = np.delete(trainingdata,1,1)
#trainingdata_1 = np.delete(trainingdata,0,1)

testdata = dfnp_mod[259:387,:]


#print("normalize",trainingdata_mod.shape)
#drop the diagnosis (class label) column
#testdata = np.delete(testdata,1,1)
#testdata_i = np.delete(testdata,0,1)

crossvaliddata = dfnp[387:569,:]

traininglabel = datalabels[1:258]

#get the test labels for the same number of rows as testdata
testlabel = datalabels[259:387]

#Naive bayes classifier
gaussian = GaussianNB()
gaussian.fit(trainingdata,traininglabel)
y_pred = gaussian.predict(testdata)

print("testlabels",testlabel)

#print(y_pred)

acc = round(gaussian.score(trainingdata,traininglabel)*100,2)

print("nb model accuracy",acc)

#decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(trainingdata,traininglabel)

y_dtpred = decision_tree.predict(testdata)

print(y_dtpred)

matched = 0
notmatched = 0
#for x in range(len(testlabel)):
#	for y in range(len(y_dtpred)):
#		if x==y:
#			matched = matched + 1
#		else:
#			notmatched = notmatched + 1

# prediction accuracy for naive bayes
for x in zip(testlabel,y_pred):
	if x[0] == x[1]:
		matched = matched + 1
	else:
		notmatched = notmatched + 1

matched = 0
notmatched = 0

#prediction accuracy for deicsion tree
for x in zip(testlabel,y_dtpred):
	if x[0] == x[1]:
		matched = matched + 1
	else:
		notmatched = notmatched + 1

print("decision tree predicted",matched/len(testlabel))



print("nb predicted accuracy",matched/len(testlabel))
acc_dt = round(decision_tree.score(trainingdata,traininglabel)*100,2)
print("decision tree model accuracy",acc_dt)


knn = KNeighborsClassifier(n_neighbors=12,algorithm='ball_tree')
knn.fit(trainingdata,traininglabel)
yknn_pred = knn.predict(testdata)

print("knn predicted values",yknn_pred)
acc_knn = round(knn.score(trainingdata,traininglabel)*100,2)
print("knn model accuracy",acc_knn)

matched = 0
notmatched = 0
#prediction accuracy for knn
for x in zip(testlabel,yknn_pred):
	if x[0] == x[1]:
		matched = matched + 1
	else:
		notmatched = notmatched + 1

print("knn predicted accuracy",matched/len(testlabel))




linearsvc = LinearSVC()
linearsvc.fit(trainingdata,traininglabel)
ylinearsvc_pred = linearsvc.predict(testdata)

linearacc_svc = round(linearsvc.score(trainingdata,traininglabel)*100,2)
print("svm model accuracy",linearacc_svc)

matched = 0
notmatched = 0

#prediction accuracy for svm classifier
for x in zip(testlabel,ylinearsvc_pred):
	if x[0] == x[1]:
		matched = matched + 1
	else:
		notmatched  = notmatched + 1

print("svm prediction accuracy",matched/len(testlabel))

#for each in datadf:
	#print(each[[1]])
	#each[1] = each[1].map(labeldict)
	#each[1] = each[1].fillna(0)

