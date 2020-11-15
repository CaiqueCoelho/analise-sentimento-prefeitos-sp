# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from operator import itemgetter
from optparse import OptionParser
from time import time

import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import sys

import nltk
import nltk.stem
import csv

import numpy as np

stop_words = {"de","a","o","que","e","do","da", "das", "em","um","para","é","com","não","uma","os","no","se","na","por","mais","as","dos",
"como","mas","foi","ao","ele","das","tem","à","seu","sua","ou","ser","quando","muito","há","nos","já","está","eu","também","só","pelo",
"pela","até","isso","ela","entre","era","depois","sem","mesmo","aos","ter","seus","quem","nas","me","esse","eles","estão","você","tinha",
"foram","essa","num","nem","suas","meu","às","minha","têm","numa","pelos","elas","havia","seja","qual","será","nós","tenho","lhe","deles",
"essas","esses","pelas","este","fosse","dele","tu","te","vocês","vos","lhes","meus","minhas","teu","tua","teus","tuas","nosso","nossa",
"nossos","nossas","dela","delas","esta","estes","estas","aquele","aquela","aqueles","aquelas","isto","aquilo","estou","está","estamos",
"estão","estive","esteve","estivemos","estiveram","estava","estávamos","estavam","estivera","estivéramos","esteja","estejamos","estejam",
"estivesse","estivéssemos","estivessem","estiver","estivermos","estiverem","hei","há","havemos","hão","houve","houvemos","houveram",
"houvera","houvéramos","haja","hajamos","hajam","houvesse","houvéssemos", "haveria", "houvessem","houver","houvermos","houverem","houverei","houverá",
"houveremos","houverão","houveria","houveríamos","houveriam","sou","somos","são","era","éramos","eram","fui","foi","fomos","foram","fora",
"fôramos","seja","sejamos","sejam","fosse","fôssemos","fossem","for","formos","forem","serei","será","seremos","serão","seria","seríamos",
"seriam","tenho","tem","temos","tém","tinha","tínhamos","tinham","tive","teve","tivemos","tiveram","tivera","tivéramos","tenha","tenhamos",
"tenham","tivesse","tivéssemos","tivessem","tiver","tivermos","tiverem","terei","terá","teremos","terão","teria","teríamos","teriam", "vai",
"vou", "tão", "alguma", "interesse", "ter", "caso", "abaixo", "animais", "ainda", "outras", "etc", "em", "a", "b", "c", 
"d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z", ".", ",", "-", "#", 
"black", "pra", "preço", "friday", "comprar", "tá", "tudo", "promoção", "agora", "metade"
"blackfraude", "desconto", "hoje", "lojas", "preços", "dia", "brasil", "nada", "nessa", "ano", 
"comprei", "aqui", "reais", "tava", "gente", "ver", "produtos", "todo", "produto", "antes", "loja", "pq", "aí", "fazer",
"ta", "semana", "vc", "coisa", "brasileiro", "povo", "to", "queria", "ontem", "compra", "tô", "vi",
"sempre", "sendo", "quero", "bem", "coisas", "dias", "bom", "nunca", "vcs", "pessoas", "mês", "pode", "3", "pro", "faz", "olha",
"nao", "menos", "friday", "lá", "assim", "real", "melhor", "sabe", "realmente", "parabéns", "friday", "!", "fazendo", "vamos", "hj", "dessa", 
"boa", "ai", "fica", "dar", "quer", "porque", "sei", "vendo", "alguém", "dá", "sobre", "onde", "verdade", "única", "começou", "dizer", 
"apenas", "ficar", "uns", "vão", "acho", "kkkk", "vez", "desde", "kkk", "algo", "normal", "dois", "atrás", "mundo", "chega", 
"pois", "desse", "sim", "...", "galera", "nesse", "duas", "né", "alguns", "algumas", "ia", "fazem"}
 

dictionary_stemmer = {}

def distribuicao_dos_dados_nos_clusters():
  print("\nDISTRIBUIÇAO DOS DADOS NOS CLUSTERS")

  total = 0

  for cluster in range(0, n_clusters):
    for index in range(0, tam):
      if(cluster_labels[index] == cluster):
        total+=1

    porcentagem = 100.0 * total/tam
    print("Total de tarefas no cluster " +str(cluster)+ ": " +str(total) + " tarefas, em porcentagem temos: {0:.2f}".format(porcentagem)  + "%")

    total = 0

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--use-normalizer",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--use-stemmer-snowball",
              action="store_true", default=False,
              help="Stemming all words with stemmer Snowball")
op.add_option("--use-stemmer-rslps",
              action="store_true", default=False,
              help="Stemming all words with stemmer RSLPS")
op.add_option("--n-features", type=int, default=None,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--print-distribuicao-clusters",
              action="store_true", default=False,
              help="Print distribution of tasks in each cluster.")
op.add_option("--terms-per-cluster",
              action="store_true", default=False,
              help="Print top terms per cluster.")

print(__doc__)
op.print_help()

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


portuguese_stemmer = nltk.stem.RSLPStemmer()

if opts.use_stemmer_snowball:
  portuguese_stemmer = nltk.stem.SnowballStemmer('portuguese')


def stemmed_reverse(doc):
  analyzer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stop_words,
                                 use_idf=opts.use_idf).build_analyzer()

  for w in analyzer(doc):

    if w not in stop_words:

      word_stemmed = portuguese_stemmer.stem(w)

      existe = False
      if(word_stemmed in dictionary_stemmer):
        existe = True      

      if(not existe):
        dictionary_stemmer[word_stemmed] = [[w,1]]

      else:
        lists_word_stemmed_from_dictionary = dictionary_stemmer[word_stemmed]
        achou = False
        for i in range(len(lists_word_stemmed_from_dictionary)):
          list_word = lists_word_stemmed_from_dictionary[i]

          if(list_word[0] == w):
            list_word[1] = list_word[1] + 1
            achou = True
            lists_word_stemmed_from_dictionary[i] == list_word
            dictionary_stemmer[word_stemmed] = lists_word_stemmed_from_dictionary

        if(not achou):
          lists_word_stemmed_from_dictionary.append([w, 1])
          dictionary_stemmer[word_stemmed] = lists_word_stemmed_from_dictionary
  '''  
  if w not in stop_words:
    return (portuguese_stemmer.stem(w) for w in analyzer(doc))
  else:
    return 'XXX'
  '''
  return (portuguese_stemmer.stem(w) for w in analyzer(doc))


'''
class StemmedTfidfVectorizer(TfidfVectorizer):
  def build_analyzer(self):
    analyzer = super(TfidfVectorizer, self).build_analyzer()

    return lambda doc: stemmed_reverse(doc)
'''


class StemmedTfidfVectorizer(TfidfVectorizer):
  def build_analyzer(self):
    analyzer = super(TfidfVectorizer, self).build_analyzer()

    return lambda doc: ([portuguese_stemmer.stem(w) for w in analyzer(doc)])



# Load some categories from the training set
categories = ['1']

candidate = 'russomanno'
dataset = load_files(container_path = "/home/caiquecoelho/Documents/projetos/Extracao-de-Emocao-no-Twitter/separated_tweets/"+candidate, 
 categories = categories, load_content = True, shuffle = True, encoding = None, random_state = 0)

list_words_stemmers = []
list_words_stemmer = []
list_words_to_stemmer = []

print("\n%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

tam = len(dataset.target)

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=10000,
                                   stop_words=stop_words, alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=10000,
                                       stop_words=stop_words,
                                       alternate_sign=False, norm='l2',
                                       binary=False)
elif opts.use_stemmer_rslps:
    vectorizer = StemmedTfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stop_words,
                                 use_idf=opts.use_idf)
    

    
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stop_words,
                                 use_idf=opts.use_idf,
                                 analyzer = stemmed_reverse)
    

    print("USE STEMER RSLPS")

    if opts.use_normalizer:
      normalizer = Normalizer(copy=False)
      vectorizer_normalized = make_pipeline(vectorizer, normalizer)


elif opts.use_stemmer_snowball:
    vectorizer = StemmedTfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stop_words,
                                 use_idf=opts.use_idf)
    

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stop_words,
                                 use_idf=opts.use_idf,
                                 analyzer = stemmed_reverse)

    print("USE STEMER Snowball")

    if opts.use_normalizer:
      normalizer = Normalizer(copy=False)
      vectorizer_normalized = make_pipeline(vectorizer, normalizer)


else:
   vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stop_words,
                                 use_idf=opts.use_idf)

   if opts.use_normalizer:
      normalizer = Normalizer(copy=False)
      vectorizer_normalized = make_pipeline(vectorizer, normalizer)

if opts.use_normalizer:
  X = vectorizer_normalized.fit_transform(dataset.data)

else:
  X = vectorizer.fit_transform(dataset.data)

print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("\nPerforming dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    #print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    #print("Explained variance of the SVD step: {}%".format(
    #    int(explained_variance * 100)))

else:
  X = X.toarray()


# Do the actual clustering

range_n_clusters = [5]
for n_clusters in range_n_clusters:
    
    type = ''
    if opts.minibatch:
      clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                           init_size=1000, batch_size=1000, verbose=opts.verbose, random_state =10)
      type = 'MiniBatchKmeans'
    else:
      clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state =10, 
                  verbose=opts.verbose)
      type = 'Kmeans'

    t0 = time()
    cluster_labels = clusterer.fit_predict(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    #print("Adjusted Rand-Index: %.3f"
    #      % metrics.adjusted_rand_score(labels, km.labels_))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    #print()

    if not opts.use_hashing and opts.terms_per_cluster:

      print("\nTop terms per cluster with " +str(n_clusters) + " clusters:")

      if opts.n_components:
          original_space_centroids = svd.inverse_transform(clusterer.cluster_centers_)
          order_centroids = original_space_centroids.argsort()[:, ::-1]
      else:
          order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]

      terms = vectorizer.get_feature_names()


      dicionario_top_terms = {}
      for i in range(n_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :5]:
          if opts.use_stemmer_rslps or opts.use_stemmer_snowball:
            if(terms[ind] in dictionary_stemmer):
              lists_word_stemmed_from_dictionary = dictionary_stemmer[terms[ind]]
              word_maior = ''
              maior = 0
                  
              for list_word in lists_word_stemmed_from_dictionary:
                word = list_word[0]
                vezes = list_word[1]
                if(vezes > maior):
                  word_maior = word
                  maior = vezes


              existe0 = False
              if(word_maior in dicionario_top_terms):
                existe0 = True
                lista_word = dicionario_top_terms[word_maior]
                if(i not in lista_word):
                  lista_word.append(i)  

              if(not existe0):
                dicionario_top_terms[word_maior] = [i]

              print(' %s' % word_maior, end='')
            else:
              print('\n\nSteemed not in dict!!!')
              print(terms[ind])
              sys.exit(0)
          else:
            print(' %s' % terms[ind], end='')
        print()
      print()

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
     # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

    
    if(opts.print_distribuicao_clusters):
      distribuicao_dos_dados_nos_clusters()
      print("\n")


    file_name = "tweets_"+candidate+"_" + str(n_clusters) + "_clusters.csv"
    csvFile = open(file_name, 'w', encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    for cluster_selected in range(0, n_clusters):  
      index = 0
      for cluster_label in cluster_labels:
        if(cluster_label == cluster_selected):
          tweet = dataset.data[index].decode("utf-8")
          if('Nov' in tweet):
            print("index")
            print(index)
          else:
            csvWriter.writerow([tweet, cluster_label])
        index += 1

    '''
    file_name = str(n_clusters) +"_clusters.txt"
    file = open(file_name, "w")
    for cluster_selected in range(0, n_clusters):  
      index = 0
      number_cluster = "\n\nTWEETS IN CLUSTER: " + str(cluster_selected) + "\n"
      #print(number_cluster)
      file.write(number_cluster)

      for cluster_label in cluster_labels:
        if(cluster_label == cluster_selected):
          rejection = dataset.data[index].decode("utf-8") + "\n\n"
          file.write(rejection) 
        index += 1
      #print("\n\n")

    file.close()
    '''

    
    #Salva o output da clusterização com 10 clusters
    '''
    if(n_clusters == 20):
      for index in range(0, len(dataset.target)):
        #print(cluster_labels)
        nome_arquivo = "cluster_" +str(cluster_labels[index]) +"_estimativa_"+str(dataset.target_names[dataset.target[index]]) +"_tarefa_"+str(index)
        file = open(nome_arquivo, "w")
        string = dataset.data[index]
        string_decode = string.decode("utf-8")
        dado = str(string_decode)
        file.write(dado) 
        file.close()
    '''
