"""Copyright [2017] [Siddhant Mahapatra]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/Robosid/Artificial-Intelligence/blob/master/License.pdf
    https://github.com/Robosid/Artificial-Intelligence/blob/master/License.rtf

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_score
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ ('n9840371', 'Sid', 'Mahapatra'), ('n9160531', 'Alec', 'Gurman')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset():
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
     #main dataframe 
    datadf = pd.read_csv('medical_records.data',header=None) #using Pandas
    #print(datadf)
    dfnp = datadf.as_matrix() #convert into matrix
    #print(dfnp)
    dflabel = datadf[[1]].as_matrix() #2nd column of datadf, class labels
    #print(dflabel)
    dfnp_n = normalize(dfnp[:,[2,3,4,5,13,14,15,22,23,24,25]],axis=0) #normalize the non-normalized columns for SVM
    #print(dfnp_n)


    dfnp_mod = np.concatenate((dfnp[:,[6,7,8,9,10,11,12,16,17,18,19,20,21,26,27,28,29,30]],dfnp_n),axis=1) #concatenates with original

    #dfl = datadf[[1]].values

    #print(dfnp)
    #print(dflabel.type)
    #print(dfl)

    # class labels converted to 0,1
    t,datalabels = np.unique(dflabel,return_inverse=True)
    #print(datalabels)


    #testdata = np.array_split(dfnp,3)

    #print(testdata)

    #x[1:4,:]
    # making subset to be split into training and test subset
    dfnp_mod_sub = dfnp_mod[1:387, : ]   
    datalabels_sub = datalabels[1 : 387]

    Xtrain,Xtest,label_train,label_test = train_test_split(dfnp_mod_sub,datalabels_sub,test_size=0.4,random_state=0) ##split automatically
    #making cross-validation subsets
    Xcrossval = dfnp_mod[387:569, : ]
    Labelcrossval = datalabels[387:569]
    #training data as half of the entire dataset
    #trainingdata = dfnp_mod[1:258,:] 

    #drop the diagnosis (class label) column
    #print(trainingdata.shape)

    ##testdata = dfnp_mod[259:387,:]


    ##crossvaliddata = dfnp_mod[387:569,:]

    ##traininglabel = datalabels[1:258] # Y
    '''
    ones = 0
    zeroes = 0
    for i in range(len(traininglabel)):
        if(traininglabel[i] == 1):
            ones = ones + 1
        elif(traininglabel[i] == 0):
            zeroes = zeroes +1 
    label_data = [ones , zeroes]
    print(label_data)
    '''
    #get the test labels for the same number of rows as testdata
    ##testlabel = datalabels[259:387]

    return (Xtrain,label_train,Xtest,label_test,Xcrossval,Labelcrossval)
       
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(trainingdata, traininglabel, testdata, testlabel):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    #Naive bayes classifier
    gaussian = GaussianNB()
    gaussian.fit(trainingdata, traininglabel) #fits Gaussian Naive Bayes according to X, y of the training subset. 
    y_pred = gaussian.predict(testdata) # Gaussian’s predict  function is used to perform classification on an array of test vectors X. 

    #print("testlabels",testlabel)

    #print(y_pred)

    acc = round(gaussian.score(trainingdata,traininglabel)*100,2) #returns the mean accuracy on the given test data and labels

    print("nb model accuracy",acc)
   
    predict_accuracy(testlabel,y_pred,'NB')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(trainingdata, traininglabel, testdata, testlabel):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''

    #decision tree classifier
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(trainingdata,traininglabel) #fits  according to X, y of the training subset.

    y_dtpred = decision_tree.predict(testdata) #function is used to perform classification on an array of test vectors X.

    # print(y_dtpred)

    acc_dt = round(decision_tree.score(trainingdata,traininglabel)*100,2) #returns the mean accuracy on the given test data and labels
    print("decision tree model accuracy",acc_dt)
    predict_accuracy(testlabel,y_dtpred,'DT')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(trainingdata, traininglabel, testdata, testlabel,Xcrossval,Labelcrossval):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
        ######### KNN cross val classifier ##############

    mylist = list(range(1,30))
    neighbors = filter(lambda x: x%2!=0,mylist)
    cv_knnscores = []
    for k in neighbors: 
        clfknn = KNeighborsClassifier(n_neighbors=k,algorithm='ball_tree').fit(trainingdata,traininglabel)
        knnscore = cross_val_score(clfknn,Xcrossval,Labelcrossval,cv=10,scoring='accuracy') #cross validation
        cv_knnscores.append(knnscore.mean())
    k_value = cv_knnscores.index(max(cv_knnscores)) #gets optimal value of k

    ##print("knn cross val score accuracies",cv_knnscores)

    ############ end of knn cross val  ###########

    knn = KNeighborsClassifier(n_neighbors=k_value,algorithm='ball_tree') 
    knn.fit(trainingdata,traininglabel)  #fits  according to X, y of the training subset.
    yknn_pred = knn.predict(testdata)  #function is used to perform classification on an array of test vectors X.

    # print("knn predicted values",yknn_pred)
    acc_knn = round(knn.score(trainingdata,traininglabel)*100,2) #returns the mean accuracy on the given test data and labels
    print("knn model accuracy",acc_knn)

    predict_accuracy(testlabel,yknn_pred,'NN')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(trainingdata, traininglabel, testdata, testlabel, Xcrossval,Labelcrossval):


    #SVM classifier
    svc =  LinearSVC()
    svc.fit(trainingdata,traininglabel)  #fits  according to X, y of the training subset.
    ysvc_pred = svc.predict(testdata) #function is used to perform classification on an array of test vectors X.

 
    acc_svc = round(svc.score(trainingdata,traininglabel)*100,2) #returns the mean accuracy on the given test data and labels
    print("svm model accuracy",acc_svc) 

    predict_accuracy(testlabel,ysvc_pred,'SVM')

    ############# just SVM cross val classifier #############
    clf = SVC(kernel='linear',C=1).fit(trainingdata,traininglabel)
    #print("svm score",clf.score(testdata,testlabel))
    score = cross_val_score(clf,Xcrossval,Labelcrossval,cv=5) #cross validation
    #print("cross val score",score)
    print("cross val accuracy : %0.2f"%round(score.mean()*100,2))



    ########## end of SVM testing ############

    #for each in datadf:
        #print(each[[1]])
        #each[1] = each[1].map(labeldict)
        #each[1] = each[1].fillna(0)

def predict_accuracy (testlabel,y_prediction,name):

    matched = 0
    notmatched = 0
    #for x in range(len(testlabel)):
    #   for y in range(len(y_dtpred)):
    #       if x==y:
    #           matched = matched + 1
    #       else:
    #           notmatched = notmatched + 1

    # prediction accuracy for naive bayes
    for x in zip(testlabel, y_prediction):
        if x[0] == x[1]:
            matched = matched + 1
        else:
            notmatched = notmatched + 1
    print("predicted error percentage of",name,":",round(notmatched/len(testlabel)*100,2)) #predicted error percentage


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    print(my_team())
    (X_training,Y_training, x_test, y_test, Xcrossval,Labelcrossval) = prepare_dataset()
    while(True):
        choice = input(" Enter your choice of classifier: NB / NN / DT / SVM or press '1' to exit: ")
        if (choice == 'NB'):
            build_NB_classifier(X_training,Y_training, x_test, y_test)
        elif (choice =='NN'):
            build_NN_classifier(X_training,Y_training, x_test, y_test, Xcrossval,Labelcrossval)
        elif (choice == 'DT'):
            build_DT_classifier(X_training,Y_training, x_test, y_test)
        elif (choice == 'SVM'):
            build_SVM_classifier(X_training,Y_training, x_test, y_test, Xcrossval,Labelcrossval)
        elif (choice == '1'):
            exit()
        else:
            pass
