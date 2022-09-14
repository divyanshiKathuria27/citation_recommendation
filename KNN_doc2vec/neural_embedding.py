import codecs,string,re,os
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import random
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15



def find_similar(name, weights, index_name = 'paper', n = 20, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    print(index_paper[22822])
    print(map_index_key[22822])
    print("YAA")
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between paper and all others
        dists = np.dot(weights, weights[22822])
    except KeyError:
        print(f'{name} Not Found.')
        print("NOOOO")
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
        print(f'{index_name.capitalize()}s furthest from {name}.\n')
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]
        #print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    
    print(sorted_dists)
    # Print the most similar and distances
    for c in reversed(closest):
        if index_paper[c] !="":
            print(index_paper[c])
            print(map_index_key[c])
            print("Similarity:",dists[c])
            



def paper_embedding_model(embedding_size = 50, classification = False):
    
    # Both inputs are 1-dimensional
    paper  = Input(name = 'paper', shape = [1])
    reference  = Input(name = 'reference', shape = [1])
    
    # Embedding the paper (shape will be (None, 1, 50))
    paper_embedding = Embedding(name = 'paper_embedding',
                               input_dim = len(index_paper),
                               output_dim = embedding_size)(paper)
    
    # Embedding the reference (shape will be (None, 1, 50))
    reference_embedding = Embedding(name = 'reference_embedding',
                               input_dim = len(index_reference),
                               output_dim = embedding_size)(reference)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([paper_embedding, reference_embedding])

     # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [paper, reference], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [paper, reference], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model


def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):

    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    # This creates a generator
    pairs_set=set(pairs)
    while True:
        # randomly choose positive examples

        for idx, (paper_id, reference_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (paper_id, reference_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection

            random_paper = random.randrange(len(map_index_key))
            random_reference = random.randrange(len(map_ref_key))
            
            
            if (random_paper, random_reference) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_paper, random_reference, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'paper': batch[:, 0], 'reference': batch[:, 1]}, batch[:, 2]



references=pickle.load(open("references_modified.pickle","rb"))  
abstracts=pickle.load(open("abstract_modified.pickle","rb"))  
titles=pickle.load(open("title_modified.pickle","rb"))  
 
#sprint(sentence_list)

#print(len(sentence_list))
all_titles_list=[]
for i in references.keys():
    #all_titles_list.append(i)
    for j in references[i]:
        #print(j)
        all_titles_list.append(j)
        #print(all_titles_list)
        
#print(all_titles_list)

for i in references.keys():
    if len(references[i]) !=0:
        all_titles_list.append(i)

all_titles_list=list(set(all_titles_list))
print(len(all_titles_list))



counter=0
map_index_key={}
index_paper={}
val_abs_list=[]
for k in all_titles_list:
    k=str(k)
    abstract=""
    title=""
    if k in titles.keys():
        title=titles[k]
    if k in abstracts.keys():
        abstract=abstracts[k]

    val=title+""+abstract
    #print(val)
    val_abs_list.append(val)

    map_index_key[counter]=k
    index_paper[counter]=val
    counter+=1

'''
dict_val_abs=Counter(val_abs_list)
#print(dict_val_abs)
for i in dict_val_abs:
    if dict_val_abs[i]>1:
        print("\n",i)
        print(dict_val_abs[i])
print(len(val_abs_list))
print(len(set(val_abs_list)))

'''

map_ref_key={}
number_ref=[]
tempor=[]
for k in references.keys():
    ref_list=references[k]
    for each_citation in ref_list:
        each_citation=str(each_citation)
        abstract=""
        title=""
        if each_citation in titles.keys():
            title=titles[each_citation]
            number_ref.append(each_citation)
        if each_citation in abstracts.keys():
            abstract=abstracts[each_citation]

        val=title+""+abstract
        tempor.append((val,each_citation))


number_ref=list(set(number_ref))  
tempor=list(set(tempor))  
index_reference={}
rev_map_ref={}
counter1=0
for t in tempor:
    index_reference[counter1]=t[0]
    map_ref_key[counter1]= t[1]
    rev_map_ref[t[1]] = counter1
    counter1=counter1+1

print("len of index_reference:",len(index_reference))
'''
outfile = open("index_reference.p",'wb')
pickle.dump(index_reference,outfile)
outfile.close() 
outfile = open("index_paper.p",'wb')
pickle.dump(index_paper,outfile)
outfile.close() 

klkl=1
pairs=[]
for key in index_paper.keys():
    print(klkl)
    klkl+=1
    paper_corr_id = map_index_key[key]
    reflist=references[paper_corr_id]
    for corr_id in reflist:
        key11 = rev_map_ref[corr_id]
        t1=(key,key11)
        pairs.append(t1)
        
outfile = open("pairs.p",'wb')
pickle.dump(pairs,outfile)
outfile.close() 
'''

index_reference=pickle.load(open("index_reference.p","rb"))
index_paper=pickle.load(open("index_paper.p","rb"))
pairs=pickle.load(open("pairs.p","rb"))

# Instantiate model and show parameters
model = paper_embedding_model()
model.summary()
n_positive = 500
gen = generate_batch(pairs, n_positive, negative_ratio = 2)

gh=1000

# Train
h = model.fit_generator(gen,
 epochs = 15, 
    steps_per_epoch = gh)
model.save('first_attempt.h5')


# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
model=load_model('first_attempt.h5')

paper_layer = model.get_layer('paper_embedding')

paper_weights = paper_layer.get_weights()[0]

paper_weights = paper_weights / np.linalg.norm(paper_weights, axis = 1).reshape((-1, 1))
print(len(paper_weights))
print(type(paper_weights))
print(np.sum(np.square(paper_weights[0])))
print(find_similar('Spatial Data Structures. An overview is presented of the use of spatial data structures in spatial databases. The focus is on hierarchical data structures, including a number of variants of quadtrees, which sort the data with respect to the space occupied by it. Such techniques are known as spatial indexing methods. Hierarchical data structures are based on the principle of recursive decomposition. They are attractive because they are compact and depending on the nature of the data they save space as well as time and also facilitate operations such as search. Examples are given of the use of these data structures in the representation of different data types such as regions, points, rectangles, lines, and volumes.', paper_weights))

    

