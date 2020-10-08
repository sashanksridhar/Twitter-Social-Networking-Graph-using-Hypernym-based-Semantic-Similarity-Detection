from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from os import walk
import nltk
import csv
import pandas as pd
stop_words = set(stopwords.words('english'))


def check_for_hypernim(token):
    hypernims = []
    for i in range(15):
        try:
            hypernims1 = []
            for i, j in enumerate(wn.synsets(token)):
                # hypernims1=list(chain(*[l.lemma_names() for l in j.hypernyms()]))
                for l in j.hypernyms():
                    hypernims1.append(l.lemma_names()[0])
                # print token
                # print(hypernims1)
                # print(hypernims1[0])
            token = hypernims1[0]
            hypernims.append(hypernims1)
        except IndexError:
            hypernims.append([token])
            continue
    return hypernims

test = []
for (dirpath, dirnames, filenames) in walk("E:\\twitterTweetsExtraction\\test"):
    test.extend(filenames)
    break
for i in range(0,len(filenames)):
    test[i] = test[i].split(".")[0]
dir  = []
for (dirpath, dirnames, filenames) in walk("E:\\twitterTweetsExtraction\\tweets"):
    dir = dirnames
    break
files = []
count = 0
for i1 in dir:
    f0 = []
    for (dirpath, dirnames, filenames) in walk("E:\\twitterTweetsExtraction\\tweets\\"+str(i1)+"\\"):
        f0.extend(filenames)
        break

    for f in f0:
        print(count)
        count+=1

        filename = f
        df = pd.read_csv("E:\\twitterTweetsExtraction\\tweets\\"+str(i1)+"\\"+filename,encoding='latin1',header=None)
        df.columns = ["id","name","date","tweet"]
        # file1 = open("E:\\twitterTweetsExtraction\\tweets\\"+str(i)+"\\"+filename,encoding='latin1')
        uid = filename.split(".")[0]
        if uid in files:
            continue
        elif uid in test:
            continue
        else:
            files.append(uid)
        print(uid)
        tweets = df["tweet"].values.tolist()
        for line in tweets:

            words = line.split()
            words[0] = words[0][2:]
            with open("E:\\twitterTweetsExtraction\\tweets\\train.csv", 'a',encoding='latin1') as csv_file:
                filewriter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                filtered_sentence = []

                x = []
                for r in words:

                    if not r in stop_words:
                        # print(r)

                        filtered_sentence.append(r)
                tagged = nltk.pos_tag([word for word in filtered_sentence if word])
                for i in tagged:
                    if len(i[0]) != 0 or len(i[0]) != 1:


                        if i[1] == 'IN' or i[1] == 'DT' or i[1] == 'CD' or i[1] == 'CC' or i[1] == 'EX' or i[1] == 'MD' or   i[1] == 'WDT' or i[1] == 'WP' or i[1] == 'UH' or i[1] == 'TO' or i[1] == 'RP' or i[1] == 'PDT' or i[1] == 'PRP' or i[1] == 'PRP$' or i[0] == 'co':
                            # print(i[0])
                            continue
                        else:

                            x.append(i[0].strip(".,?!"))
                # print(x)
                for i in x:
                    # print(i)
                    l = []
                    l.append(uid)
                    l.append(i)
                    hype = check_for_hypernim(i)
                    # print("hype")
                    # print(i)
                    # print(hype)
                    if len(hype) == 0:
                        # print("hi")
                        continue
                        # hype.append(i)  # 1
                        # hype.append(i)  # 2
                        # hype.append(i)  # 3
                        # hype.append(i)  # 4
                        # hype.append(i)  # 5
                        # hype.append(i)  # 6
                        # hype.append(i)  # 7
                        # hype.append(i)  # 8
                        # hype.append(i)  # 9
                        # hype.append(i)  # 10
                        # hype.append(i)  # 11
                        # hype.append(i)  # 12
                        # hype.append(i)  # 13
                        # hype.append(i)  # 14
                        # hype.append(i)  # 15
                    for hyper in hype:
                        l.append(hyper[0])
                    # print(l)
                    if l[2] == l[3]:
                        continue
                    filewriter.writerow(l)
                csv_file.close()
print(len(files))




