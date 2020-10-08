from os import listdir
from os.path import isfile, join
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from os import walk
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from collections import Counter
rowsx = []
yx = []

with open("E:\\twitterTweetsExtraction\\train.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)-9):

            for j in row[i].split("\n"):
                # s+=j+" "

                    # print(j)
                    rows1.append(j)


        yx.append(row[0])
        del (row[0])
    # print(rows1)

        rowsx.append(rows1)

print("done")
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(rows)
embeddings = Word2Vec(size=200, min_count=3)
embeddings.build_vocab([sentence for sentence in rowsx])
embeddings.train([sentence for sentence in rowsx],
                 total_examples=embeddings.corpus_count,
                 epochs=embeddings.epochs)

# print(list(embeddings.wv.vocab))

gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx])
tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map))
# print(tfidf_map)




def encode_sentence(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings.wv[word].reshape((1, emb_size)) * tfidf_map[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector


x_train = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx)]))

print(x_train.shape)

modelknn = KNeighborsClassifier(n_neighbors=1)
modelknn.fit(x_train,yx)
print("done")







rowsx1 = []
with open("E:\\twitterTweetsExtraction\\test_csv\\" + "62.csv", 'r', encoding='latin1') as csv1:

    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []

        # print(row)
        for i in range(1, len(row)-9):

            for j in row[i].split("\n"):
                rows1.append(j)

        del (row[0])
        # print(rows1)

        rowsx1.append(rows1)
x_test = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx1)]))

# Test = vectorizer.transform(rows1)
predicted_labels_knn = modelknn.predict(x_test)
# print(predicted_labels_knn)


counts = Counter(predicted_labels_knn)
# print(counts)
# print(connections)

dict = {}
for i in counts:
    dict[i] = []

for i in range(0,len(predicted_labels_knn)):

    dict[predicted_labels_knn[i]].extend(rowsx[i])
# print(dict)
rows1 =[]
fig, ax1 = plt.subplots()
for i in counts:
    vectorizer = TfidfVectorizer(stop_words='english')
    vect = vectorizer.fit_transform(dict[i])
    count_vect_df = pd.DataFrame(vect.todense(), columns=vectorizer.get_feature_names())
    # print(i)
    f = 0
    for i1 in count_vect_df.shape:
        if i1==1:
            f=1
            break
    if f==1:
        continue
    # print(count_vect_df)

    pca = PCA(n_components=2).fit(count_vect_df)

    data2D = pca.transform(count_vect_df)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(counts)))
    #print(colors)
    #cs = [colors[int(i/len(Test.todense()))] for i in range(len(vectorizer.transform(predicted_labels_knn).todense())*len(Test.todense()))]
    #print(cs)

        #print(vectorizer.transform(i))
        #data = pca1.transform(vectorizer.transform(list(i)).todense())
    x=""
    x+=i
    # k =x.replace("_","")
    ax1.scatter(data2D[:,0],data2D[:,1],label=x)
ax1.legend()
plt.show()