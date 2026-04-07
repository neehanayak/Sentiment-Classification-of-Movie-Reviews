'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
from xml.sax.handler import feature_external_ges
import nltk
import re
from nltk.corpus import stopwords
import sentiment_read_subjectivity
import sentiment_read_LIWC_pos_neg_words
import crossval
from nltk.metrics import ConfusionMatrix
from nltk.collocations import *
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from wordcloud import WordCloud











#import sklearn
#from nltk.classify.scikitlearn import SklearnClassifier
#from sklearn.ensemble import RandomForestClassifier




nltkstopwords = nltk.corpus.stopwords.words('english')
words_stop = [
    'could', 'would', 'might', 'must', 'need', 'sha', 'wo', 'y', "'s", "'d", "'ll",
    "'t", "'m", "'re", "'ve", "n't", "'i", 'not', 'no', 'can', 'don', 'nt',
    'actually', 'also', 'always', 'even', 'ever', 'just', 'really', 'still', 
    'yet', 'however', 'nevertheless', 'furthermore', 'therefore', 'otherwise', 
    'meanwhile', 'though', 'although', 'thus', 'hence', 'indeed', 'perhaps', 
    'especially', 'specifically', 'usually', 'often', 'sometimes', 'certainly', 
    'sometimes', 'typically', 'mostly', 'generally', 'about', 'above', 'across', 
    'after', 'against', 'among', 'around', 'at', 'before', 'behind', 'below', 
    'beneath', 'beside', 'between', 'beyond', 'during', 'inside', 'onto', 'outside', 
    'through', 'under', 'upon', 'within', 'without'
]
stopwords = nltkstopwords + words_stop


# initialize the positive, neutral and negative word lists
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('C:/Users/kulve/OneDrive/Documents/FinalProjectData (5)/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')


# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

dpath = 'C:/Users/kulve/OneDrive/Documents/FinalProjectData (5)/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'
SL = sentiment_read_subjectivity.readSubjectivity(dpath)

#Defining preprocessing function
def preprocessing(line):
  #converting to lower
  a = re.split(r'\s+', line.lower())
  #removing punctuations
  p = re.compile(r'[!#$%&()*+,"-./:;<=>?@[\]^_`{|}~]')
  w = [p.sub("",i) for i in a]
  #removing stop words
  y = []
  for x in w:
    if x in stopwords:
      continue   
    else:
      y.append(x)
  l = " ".join(y)
  return l

def ft(t):
  a=[]
  for n in t[0]:
    if len(n)>2:
      a.append(n)
  return (a,t[1])




# Different Functions for feature sets :

def bw(a,i):
  a = nltk.FreqDist(a)
  wf = [w for (w,c) in a.most_common(i)]
  return wf   

def uf(d,wf):
  df= set(d)
  f = {}
  for word in wf:
    f['V_%s'% word] = (word in df)
  return f


def bigram_bow(wordlist,n):
  bigram_measure = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(wordlist)
  finder.apply_freq_filter(2)
  b_features = finder.nbest(bigram_measure.chi_sq,4000)
  return b_features[:n]


def bf(doc,word_features,bigram_feature):
  dw = set(doc)
  db = nltk.bigrams(doc)
  features = {}

  for word in word_features:
    features['V_{}'.format(word)] = (word in dw)
  
  for b in bigram_feature:
    features['B_{}_{}'.format(b[0],b[1])] = (b in db)

  return features


def pf(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features




def slf(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg) 

    if 'positivecount' not in features:
      features['positivecount'] = 0
    if 'negativecount' not in features:
      features['negativecount'] = 0

    return features


def liwc(doc,word_features,poslist,neglist):
  doc_words = set(doc)
  features= {}

  for word in word_features:
    features['contains({})'.format(word)] = (word in doc_words)
  
  pos = 0
  neg = 0
  for word in doc_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      pos+=1
    elif sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      neg+=1
    features ['positivecount'] = pos
    features ['negativecount'] = neg


  if 'positivecount' not in features:
    features['positivecount'] = 0
  if 'negativecount' not in features:
    features['negativecount'] = 0

  return features


def combo(doc,word_features,SL,poslist,neglist):
  doc_words = set(doc)
  features={}

  for word in word_features:
    features['contains({})'.format(word)] = (word in doc_words )
  
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in doc_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      strongPos +=1
    elif sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      strongNeg +=1
    elif word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
    features['positivecount'] = weakPos + (2 * strongPos)
    features['negativecount'] = weakNeg + (2 * strongNeg)

  if 'positivecount' not in features:
    features['positivecount'] = 0
  if 'negativecount' not in features:
    features['negativecount'] = 0

  return features    



# Saving feature sets for for other classifier training
def save(features, path):
    f = open(path, 'w')
    featurenames = features[0][0].keys()
    fnameline = ''
    for fname in featurenames:
        fname = fname.replace(',','COM')
        fname = fname.replace("'","SQ")
        fname = fname.replace('"','DQ')
        fnameline += fname + ','
    fnameline += 'Level'
    f.write(fnameline)
    f.write('\n')
    for fset in features:
        featureline = ''
        for key in featurenames:
            # Check if the key exists in the feature set
            if key in fset[0]:
                featureline += str(fset[0][key]) + ','
            else:
                featureline += 'NA,'  # If the key does not exist, write 'NA' instead
        if fset[1] == 0:
          featureline += str("Less Negitive")
        elif fset[1] == 1:
          featureline += str("Strong negitive")
        elif fset[1] == 2:
          featureline += str("Neutral")
        elif fset[1] == 3:
          featureline += str("Strongly positive")
        elif fset[1] == 4:
          featureline += str("Less positive")
        f.write(featureline)
        f.write('\n')
    f.close()



def naivebayesaccuracy(features):
  train_set,test_set = features[int(0.1*len(features)):], features[:int(0.1*len(features))]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("\nAccuracy : ")
  print(nltk.classify.accuracy(classifier,test_set),"\n")
  l1 = []
  tl=[]
  for (features,label) in test_set:
    l1.append(label)
    tl.append(classifier.classify(features))
  print(ConfusionMatrix(l1,tl))




def dt(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_dt = SklearnClassifier(DecisionTreeClassifier())
    classifier_dt.train(train_set)
    print("Classifier-DecisionTree \n")
    print("Accuracy : ", nltk.classify.accuracy(classifier_dt, test_set))







def svm(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_svm = SklearnClassifier(SVC())
    classifier_svm.train(train_set)
    print("Classifier-SVM \n")
    print("Accuracy : ", nltk.classify.accuracy(classifier_svm, test_set))
    
    

def rf(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_rf = SklearnClassifier(RandomForestClassifier())
    classifier_rf.train(train_set)
    print("Classifier - Random Forest \n")
    print("Accuracy:", nltk.classify.accuracy(classifier_rf, test_set))










def plot_sentiment_distribution(phrasedata):
    sentiments = [int(phrase[1]) for phrase in phrasedata]
    plt.hist(sentiments, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Sentiment Label')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentiment Labels')
    plt.xticks(range(0, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_word_frequency_distribution(tokens):
    freq_dist = nltk.FreqDist(tokens)
    plt.figure(figsize=(10, 6))
    freq_dist.plot(30, cumulative=False)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Distribution')
    plt.grid(True)
    plt.show()


def plot_top_words(tokens, n=20):
    freq_dist = nltk.FreqDist(tokens)
    top_n = freq_dist.most_common(n)
    words, frequencies = zip(*top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title(f'Top {n} Words Frequency Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()




def generate_wordcloud(tokens):
    text = ' '.join(tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

# Function to plot word frequency distribution





def plot_histogram(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.show()


def plot_word_length_distribution(tokens):
    word_lengths = [len(word) for word in tokens]
    plt.figure(figsize=(10, 6))
    plt.hist(word_lengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Word Length Distribution')
    plt.grid(True)
    plt.show()

def plot_word_length_boxplot(tokens):
    word_lengths = [len(word) for word in tokens]
    plt.figure(figsize=(8, 6))
    plt.boxplot(word_lengths, vert=False)
    plt.xlabel('Word Length')
    plt.title('Box Plot of Word Lengths')
    plt.grid(True)
    plt.show()








# define a feature definition function here

# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label


# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('C:/Users/kulve/OneDrive/Documents/FinalProjectData (5)/FinalProjectData/kagglemoviereviews/corpus/train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:

    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore th
      # e phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])

  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  for phrase in phraselist[:10]:
    print (phrase)
  
  
  withpreprocessing = []
  withoutpreprocessing= []
 
  for p in phraselist:

   
    tokens = nltk.word_tokenize(p[0])
    withoutpreprocessing.append((tokens, int(p[1])))

    
    p[0] = preprocessing(p[0])
    tokens = nltk.word_tokenize(p[0])
    withpreprocessing.append((tokens, int(p[1])))
  
   
  withpreprocessing_filter=[]
  
  for p in withpreprocessing:
    withpreprocessing_filter.append(ft(p))

  filtered_tokens =[]
  unfiltered_tokens = []
  for (d,s) in  withpreprocessing_filter:
    for i in d:
      filtered_tokens.append(i)

  for (d,s) in withoutpreprocessing:
    for i in d:
      unfiltered_tokens.append(i)

  
  plot_sentiment_distribution(phrasedata)
  generate_wordcloud(filtered_tokens)
  plot_histogram(filtered_tokens)
  plot_top_words(filtered_tokens)
  plot_word_length_distribution(filtered_tokens)
  plot_word_length_boxplot(filtered_tokens)
  
 
  
  


  # continue as usual to get all words and create word features
  
  # feature sets from a feature definition function

  filtered_bow_features = bw(filtered_tokens,350)
  unfiltered_bow_features = bw(unfiltered_tokens,350)

  filtered_unigram_features = [(uf(d,filtered_tokens),s) for (d,s) in withpreprocessing_filter]
  unfiltered_unigram_features = [(uf(d,unfiltered_tokens),s) for (d,s) in withoutpreprocessing]

  filtered_bigram_features = [(bf(d,filtered_bow_features,bigram_bow(filtered_tokens,350)),s) for (d,s) in withpreprocessing_filter]
  unfiltered_bigram_features = [(bf(d,unfiltered_bow_features,bigram_bow(unfiltered_tokens,350)),s) for (d,s) in withoutpreprocessing]

  filtered_pos_features = [(pf(d,filtered_bow_features),s) for (d,s) in withpreprocessing_filter]
  unfiltered_pos_features = [(pf(d,unfiltered_bow_features),s) for (d,s) in withoutpreprocessing]

  

  filtered_sl_features = [(slf(d, filtered_bow_features, SL), c) for (d, c) in withpreprocessing_filter]
  unfiltered_sl_features = [(slf(d, unfiltered_bow_features, SL), c) for (d, c) in withoutpreprocessing]


  filtered_liwc_features = [(liwc(d, filtered_bow_features, poslist,neglist), c) for (d, c) in withpreprocessing_filter]
  unfiltered_liwc_features = [(liwc(d, unfiltered_bow_features, poslist,neglist), c) for (d, c) in withoutpreprocessing]

  filtered_combo_features =  [(combo(d, filtered_bow_features,SL, poslist,neglist), c) for (d, c) in withpreprocessing_filter]
  unfiltered_combo_features = [(combo(d, unfiltered_bow_features,SL, poslist,neglist), c) for (d, c) in withoutpreprocessing]



  #Saving features
  #savingfeatures(filtered_bow_features,'filtered_bow.csv')
  #savingfeatures(unfiltered_bow_features,'unfiltered_bow.csv')
  

  save(filtered_unigram_features,'filtered_unigram.csv')
  save(unfiltered_unigram_features,'unfiltered_unigram.csv')

  save(filtered_bigram_features,'filtered_bigram.csv')
  save(unfiltered_bigram_features,'unfiltered_bigram.csv')

  save(filtered_pos_features,'filtered_pos.csv')
  save(unfiltered_pos_features,'unfiltered_pos.csv')

 

  save(filtered_sl_features,'filtered_sl.csv')
  save(unfiltered_sl_features,'unfiltered_sl.csv')

  save(filtered_liwc_features,'filtered_liwc.csv')
  save(unfiltered_liwc_features,'unfiltered_liwc.csv')

  save(filtered_combo_features,'filtered_combo.csv')
  save(unfiltered_combo_features,'unfiltered_combo.csv')
  


  # train classifier and show performance in cross-validation

  labels = [0,1,2,3,4]
  print("Cross Validation for all features(unfiltered) : \n ")

  print("\n Unigram Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_unigram_features,labels)
  print("\n Bigram Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_bigram_features,labels)
  print("\n Pos Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_pos_features,labels)
  print("\n SL Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_sl_features,labels)
  print("\n LIWC Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_liwc_features,labels)
  print("\n Combined SL LIWC Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_combo_features,labels)

  print("\n Unigram filtered : ")
  crossval.cross_validation_PRF(5,filtered_unigram_features,labels)
  print("\n Bigram filtered : ")
  crossval.cross_validation_PRF(5,filtered_bigram_features,labels)
  print("\n Pos filtered : ")
  crossval.cross_validation_PRF(5,filtered_pos_features,labels)
  print("\n SL filtered : ")
  crossval.cross_validation_PRF(5,filtered_sl_features,labels)
  print("\n LIWC filtered : ")
  crossval.cross_validation_PRF(5,filtered_liwc_features,labels)
  print("\n Combined SL LIWC filtered: ")
  crossval.cross_validation_PRF(5,filtered_combo_features,labels)



  print("\n Unigram Unfiltered : ")
  naivebayesaccuracy(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  naivebayesaccuracy(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  naivebayesaccuracy(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  naivebayesaccuracy(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  naivebayesaccuracy(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  naivebayesaccuracy(unfiltered_combo_features)


  print("\n Unigram filtered : ")
  naivebayesaccuracy(filtered_unigram_features)
  print("\n Bigram filtered : ")
  naivebayesaccuracy(filtered_bigram_features)
  print("\n Pos filtered : ")
  naivebayesaccuracy(filtered_pos_features)
  print("\n SL filtered : ")
  naivebayesaccuracy(filtered_sl_features)
  print("\n LIWC filtered : ")
  naivebayesaccuracy(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  naivebayesaccuracy(filtered_combo_features)

  print("--------------------------------------------------For desicion tree -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  dt(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  dt(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  dt(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  dt(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  dt(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  dt(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  dt(filtered_unigram_features)
  print("\n Bigram filtered : ")
  dt(filtered_bigram_features)
  print("\n Pos filtered : ")
  dt(filtered_pos_features)
  print("\n SL filtered : ")
  dt(filtered_sl_features)
  print("\n LIWC filtered : ")
  dt(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  dt(filtered_combo_features)


  



  print("--------------------------------------------------For svm -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  svm(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  svm(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  svm(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  svm(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  svm(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  svm(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  svm(filtered_unigram_features)
  print("\n Bigram filtered : ")
  svm(filtered_bigram_features)
  print("\n Pos filtered : ")
  svm(filtered_pos_features)
  print("\n SL filtered : ")
  svm(filtered_sl_features)
  print("\n LIWC filtered : ")
  svm(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  svm(filtered_combo_features)



  
  
 

  print("--------------------------------------------------For random forest-----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  rf(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  rf(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  rf(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  rf(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  rf(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  rf(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  rf(filtered_unigram_features)
  print("\n Bigram filtered : ")
  rf(filtered_bigram_features)
  print("\n Pos filtered : ")
  rf(filtered_pos_features)
  print("\n SL filtered : ")
  rf(filtered_sl_features)
  print("\n LIWC filtered : ")
  rf(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  rf(filtered_combo_features)


 

  

  

  
  


  
  
  



"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])






