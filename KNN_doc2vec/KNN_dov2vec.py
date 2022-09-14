import codecs,string,re,os
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy  

'''file="H:\\MTECH2\\IR\\IR_Project\\Graph\\DBLPOnlyCitationOct19.txt"
fp =codecs.open(file,"r",encoding='utf-8',errors='ignore')  
text=fp.read()
text=text.split("\n\n")
#print(text)


unique_nodes_list=[]
G_test=pickle.load(open("H:\\MTECH2\\IR\\IR_Project\\Graph\\G_test.p","rb"))
for single_tuple in G_test:

    unique_nodes_list.append(int(single_tuple[0]))
    unique_nodes_list.append(int(single_tuple[1]))
unique_nodes_list=list(set(unique_nodes_list))

print(len(unique_nodes_list))
title_dict={}
abstract_dict={}
ground_truth_references={}



counter=0
for i in text:
    #print(i)
    counter+=1
    print(counter)
    j = i.split("\n")
    #print(j)
    #print(1/0)
    abstract=""
    references=[]
    for k in j:
        #print(k)
        if (k.startswith("#index")):
            index = k[6:]
            print(index)
        if (k.startswith("#*")):
            title = k[2:]
        if (k.startswith("#!")):
            abstract = k[2:]
        if (k.startswith("#%")):
            references.append(k[2:])
    
    #print(unique_nodes_list)
   
        
    title_dict[index]=title
    if abstract != "":
        abstract_dict[index] = abstract
    ground_truth_references[index] = references
    
document_frequency_pickle1 = open("title_dict.pickle","wb")
pickle.dump(title_dict, document_frequency_pickle1)

document_frequency_pickle2 = open("abstract_dict.pickle","wb")
pickle.dump(abstract_dict, document_frequency_pickle2)

document_frequency_pickle3 = open("ground_truth_references.pickle","wb")
pickle.dump(ground_truth_references, document_frequency_pickle3)

1/0'''



references=pickle.load(open("ground_truth_references.pickle","rb"))  

print(references['1517610'])

abstracts=pickle.load(open("abstract_dict.pickle","rb"))  
titles=pickle.load(open("title_dict.pickle","rb"))  
#print(len(references))
#print(len(titles))
#print(len(abstracts))


all_titles_list=[]
for i in references.keys():
    #print(i)
    for j in references[i]:
        #print(j)
        all_titles_list.append(j)
        #print(all_titles_list)
        
#print((set(all_titles_list)))




for i in references.keys():
    if len(references[i]) !=0:
        all_titles_list.append(i)



all_titles_list=list(set(all_titles_list))

#print(titles.keys())


'''new_references={}
new_abstracts={}
new_titles={}

cnt=0
for i in all_titles_list:
    
    cnt+=1
    #print(references[i])
    if i in references.keys():
        new_references[i]=references[i]
    if i in abstracts.keys():
        new_abstracts[i]=abstracts[i]
    if i in titles.keys():
        print(cnt)
        new_titles[i]=titles[i]

print(len(new_abstracts))
print(len(new_titles))
print(len(new_references))

document_frequency_pickle1 = open("title_modified.pickle","wb")
pickle.dump(new_titles, document_frequency_pickle1)

document_frequency_pickle2 = open("abstract_modified.pickle","wb")
pickle.dump(new_abstracts, document_frequency_pickle2)

document_frequency_pickle2 = open("references_modified.pickle","wb")
pickle.dump(new_references, document_frequency_pickle2)

print("done")'''

abstracts_modified=pickle.load(open("abstract_modified.pickle","rb"))  
titles_modified=pickle.load(open("title_modified.pickle","rb"))
references_modified=pickle.load(open("references_modified.pickle","rb"))





'''sentence_list=[]   
for i in all_titles_list:
    abstract=""
    title=""
    if i in titles_modified.keys():
        title=titles_modified[i]
    if i in abstracts_modified.keys():
        abstract=abstracts_modified[i]
    sentence_list.append(title+" "+abstract)

print("set len:",len(sentence_list))

document_frequency_pickle1 = open("sentence_list_modified1.pickle","wb")
pickle.dump(sentence_list, document_frequency_pickle1) 

print("done")'''


sentence_list=pickle.load(open("sentence_list_modified1.pickle","rb")) 
print("len:",len(sentence_list)) 


'''tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentence_list)]
list_tag=[]
for i in tagged_data:
    list_tag.append(i[1])
#print(list_tag)


print("pickle loaded")
max_epochs = 10
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)

print("model saved")
  
model.build_vocab(tagged_data)
print("vocab build")

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v_modified.model")
print("Model Saved")'''


        
# khelna with model


from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v_modified.model")
print(type(model.docvecs))
print(model.docvecs)
samples=[]
index_to_sample_map=[]
#print(model.docvecs['4990'])
'''for i in titles_modified.keys():
    #print(i)
    #print(type(i))
    if i in all_titles_list:
        if model.docvecs[i] is not None:
            samples.append(model.docvecs[i])
            index_to_sample_map.append(i)'''
for vector in model.docvecs.vectors_docs:
    samples.append(vector)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
L1=['283631','283678','614290','1179195','1517610']
for i in L1:
    #print(i)
    i=str(i)
    test_data=titles_modified[i]+""+abstracts_modified[i]
    test_data = word_tokenize(test_data.lower())
    v1 = model.infer_vector(test_data)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(samples)
    neighbour=(neigh.kneighbors([v1],return_distance=False))

    #print("neighbour",neighbour)

    final_result=[]
    for n in neighbour[0]:
        final_result.append(n)
        #print(type(i))
        n=str(n)
        #print(type(i))
        if n in references.keys():
            #print(i)
            #print(references[i])
            for j in references[i]:
                final_result.append(j)
                #a=1
    #print(final_result)
    final_results=[]
    for f in final_result:
        final_results.append(str(f)) 
    #print(final_results)       
    cnt=0
    gn=references[i]
    #print("gn",gn)
    final_results=list(set(final_results))
    for k in final_results:
        #print(type(k))
        if k in gn:
            cnt+=1
    print("paper id:",i)
    print("original references:",len(gn))
    print("matched_references:",cnt)