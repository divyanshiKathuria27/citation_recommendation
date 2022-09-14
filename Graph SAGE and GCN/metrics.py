import codecs,string,re,os
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
import pickle
from nltk.stem import WordNetLemmatizer
from collections import Counter
import inflect
import sys
import operator
import matplotlib.pyplot as plt
import statistics



pos_edges=pickle.load(open("pos_edges.p","rb"))
neg_edges=pickle.load(open("neg_edges.p","rb"))
test_metrics_transformed=pickle.load(open("test_metrics_transformed.p","rb"))
G_test=pickle.load(open("G_test.p","rb"))

title_dict=pickle.load(open("title_dict.pickle","rb"))
abstract_dict=pickle.load(open("abstract_dict.pickle","rb"))

ground_truth_references = {}


'''
l=0
for one_list in G_test:
	print(l)
	l+=1
	if ground_truth_references.get(one_list[0]) is not None:
		val_list= ground_truth_references[one_list[0]]
		val_list.append(one_list[1])
		ground_truth_references[one_list[0]] = val_list
	else:
		newl=[]
		newl.append(one_list[1])
		ground_truth_references[one_list[0]] = newl
		
#69 ---> 617
outfile = open("ground_truth_references.p",'wb')
pickle.dump(ground_truth_references,outfile)
outfile.close() 
'''
ground_truth_references=pickle.load(open("ground_truth_references.pickle","rb"))

#3515
'''
for key in ground_truth_references.keys():
	if len(ground_truth_references[key]) > 81:
		print(key,end = " ")
		print(len(ground_truth_references[key]))
'''		

tuple_truth=[]
for val in ground_truth_references[3515]:
	tuple1=(3515,val)
	tuple_truth.append(tuple1)
outfile = open("tuple_truth.p",'wb')
pickle.dump(tuple_truth,outfile)
outfile.close() 

true_postive=0
true_negative=0
false_postive=0
false_negative=0

recommended_papers=[]
for edge in pos_edges:
	if edge in tuple_truth:
		recommended_papers.append(edge[1])
		true_postive+=1

for edge in neg_edges:
	if edge in tuple_truth:
		true_negative+=1
pred_labels=[]
for e in tuple_truth:
	if e in pos_edges:
		pred_labels.append(1)
	if e in neg_edges:
		pred_labels.append(0)

y_actual=[1]*len(tuple_truth)


print("\n")

print(" Input Paper title : ", end = " ")
print(title_dict["3515"])
print("\n")
print("Citations recommended by the Model are : ")
for p in recommended_papers:
	p = str(p)
	print("->",end= " ")
	print(title_dict[p]) 
print(pred_labels)
print(y_actual)


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
actual = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

predicted = [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]
results = confusion_matrix(actual, predicted) 
  

print(results) 
import pylab as plt
from sklearn.metrics import confusion_matrix
labels = [ 'Papers marked as citation','Papers not marked citations']
labels1=[ 'Correct citations','Incorrect citations']
cm = confusion_matrix(y_actual, pred_labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels1)
ax.set_yticklabels([''] + labels)

plt.show()
