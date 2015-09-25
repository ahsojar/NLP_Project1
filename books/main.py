#######################
# ALISHA SOJAR AND SHREYA SITARAMAN
# Place main.py in books directory
# cd to books directory
# run python main.py 
#######################

import re
import os
import random
import nltk
import collections
import sys
import math

reload(sys)
sys.setdefaultencoding('UTF8')

from collections import Counter
from nltk.tokenize import word_tokenize

####################################


def unigram(tokens):

  unigram_counts = {} 
  unigram_prob = {} 

  for word in tokens:
    #Create unigram counts
    if word in unigram_counts:
      unigram_counts[word] += 1
    else:
      unigram_counts[word] = 1

  #Normalize unigram probabilities from 1 to 0
  for k in unigram_counts:
    word_count = len(tokens)
    unigram_prob[k] = unigram_counts[k]/float(word_count)

  return [unigram_counts, unigram_prob]

####################################

'''
Computes the bigram model given a set of tokens.

Params:
tokens: array of tokens in text
unigram_counts: dictionary of wordcounts of these tokens

Returns:
result[0] as the bigram counts
result[1] as the bigram counts two dimesnional
result[2] as the bigram probabilities
'''
def bigram(tokens, unigram_counts):
  bigram_counts = {} #{'word1 word2': count, 'word2 word3': count}
  bigrams_2d = {} #2D bigram counts {word1: {word2: count, word3: count}, word2: {word4: count}}
  bigram_prob = {} #probability of each bigram, 2D

  prev_word = 'START'
  for word in tokens:
  #create bigram dictionary with 'word1 word2' as key
    bigram = prev_word + ' ' + word
    if bigram in bigram_counts:
        bigram_counts[bigram] += 1
    else:
        bigram_counts[bigram] = 1

    #create 2 dimensional bigrams
    if prev_word in bigrams_2d:
      if word in bigrams_2d[prev_word]:
        bigrams_2d[prev_word][word] += 1
      else:
        bigrams_2d[prev_word][word] = 1
    else:
      bigrams_2d[prev_word] = {}
      bigrams_2d[prev_word][word] = 1

    prev_word = word

  #normalize the 2D bigram counts (divide by probability of first word)
  for first_word in bigrams_2d:
    second_words = bigrams_2d[first_word]
    normalized_words = {}
    for w in second_words:
      if w != 'START':
        normalized_words[w] = second_words[w]/ float(unigram_counts[first_word])
        bigram_prob[first_word] = normalized_words

  return [bigram_counts, bigrams_2d, bigram_prob]

####################################

'''
Generates a random sentence given a probability model of words.

Params:
model: pass in string 'unigram' or 'bigram' to indicate type of model
probabilties: dictionary of probabilties

Returns:
a string random sentence
'''
def randomSentence(model, probabilities):
  cum_prob = cumulativeProb(model, probabilities)
  
  words = 1
  added = False
  prev_word = 'been'
  sentence = 'START Professor Claire Cardie is'
  if model == 'unigram':
    word_prob = cum_prob
  elif model == 'bigram':
    word_prob = cum_prob[prev_word]

  while (not 'END' in sentence) and (words < 30):
    if model == 'bigram':
      word_prob = cum_prob[prev_word]
    random_p = random.uniform(0,1)
    added = False
    for key,value in word_prob.items():
        if value > random_p - 0.001 and value < random_p + 0.001 and not added and key != 'START':
            words += 1
            sentence += ' ' + key
            added = True
            prev_word = key
  return sentence


####################################

'''
Recomputes ngram probabilities using Good Turing smoothing.

Params:
ngram counts: dictionary of ngram counts

Returns:
dictionary of Good Turing ngram probabilities
'''
def goodTuringSmoothing(model, ngram_counts, ngram_prob):
  freq_counts= Counter(ngram_counts.values()).most_common()
  freq_counts = dict(freq_counts)
  unk_prob = 0.0
  goodTuring = ngram_counts
  length = sum(ngram_counts.values())
  x = 1

  for key, value in ngram_counts.items():
      unk_prob = float(float(freq_counts[1])/float(length))
      if value >= 1 and value <= 5:
        if value+1 in freq_counts:
          num = float(value + 1) * (float(freq_counts[value+1])/float(freq_counts[value]))
          new_prob = float(num/float(length))
          goodTuring[key] = new_prob
        else:
          num = float(value + 1) * (0)/float(freq_counts[value])
          new_prob = float(num/float(length))
          goodTuring[key] = new_prob

      if value > 5:
        if model == 'unigram':
          new_prob = ngram_prob[key]
          goodTuring[key] = new_prob
        elif model == 'bigram':
          new_prob ==  ngram_prob[key.split()[0]][key.split()[1]]
  goodTuring['<UNK>']= unk_prob

  return goodTuring

####################################

'''
Computes perpelxity for a test set against a training set unigram probability model.

Params:
unigram_prob: dictionary of unigram probibilities
tokens: tokens in test set
goodTuring_uni: 

Returns:
float, perplexity value
'''
def perplexityUnigrams(unigram_prob, tokens, goodTuring_uni):
  total = 0
  word_count = len(tokens)

  for w in tokens:
    if w in unigram_prob:
      x = -math.log(unigram_prob[w])
      total = total + x
    else:
      x = -math.log(goodTuring_uni['<UNK>'])
      total = total + x
  
  perplexity = total/float(word_count)
  return perplexity

####################################

'''
Computes perplexity for a test set against a training set bigram probability model.

Params:
bigram_prob: dictionary of bigram probibilities
tokens: tokens in test set
goodTuring_bi:

Returns:
float, perplexity value
'''
def perplexityBigrams(bigram_prob, tokens, goodTuring_bi):
  total = 0
  word_count = len(tokens)
  prev_word = tokens[0]
  not_first_word = False

  for word in tokens:
    if not_first_word:
      if word in bigram_prob:
        x = -math.log(bigram_prob[prev_word][word])
        total = total + x
        
      else:
        x = -math.log(goodTuring_bi['<UNK>'])
        total = total + x
    not_first_word = True
  
  perplexity = total/float(word_count)
  return perplexity

####################################

'''
Computes the unigram model given a set of tokens
Uses add-one smoothing, so also handles case of unknown words.

Params:
unigram counts: dictionary of unigram counts

Returns:
dictionary of add-one smoothing unigram probabilities
'''
def addOneSmoothingUnigram(unigram_counts, tokens):
  len_corpus = len(tokens)
  len_vocab = len(unigram_counts)
  cum_prob = 0

  add_one_smooth_uni = {}
  for key,value in unigram_counts.items():
      add_one_smooth_uni[key] = float(value + 1.0) / (len_vocab+len_corpus)  
      cum_prob += add_one_smooth_uni[key] 
  add_one_smooth_uni['<UNK>'] = 1-cum_prob
  return add_one_smooth_uni

####################################

'''
Computes the bigram model with add-one smoothing, which can now handle unseen bigrams.

Params:
unigram counts: dictionary of unigram counts for these tokens

Returns:
dictionary of Good Turing unigram probabilities
'''
def addOneSmoothingBigram(unigram_counts, bigrams):
  vocab_length = len(unigram_counts)
  add_one_smooth_bi = bigrams
  for first_word in add_one_smooth_bi:
      cum_prob = 0
      for w in add_one_smooth_bi[first_word]:
        add_one_smooth_bi[first_word][w] = float(add_one_smooth_bi[first_word][w]+1)/float(unigram_counts[first_word] + vocab_length)   
        cum_prob += add_one_smooth_bi[first_word][w]
      add_one_smooth_bi[first_word]['<UNK>'] = 1-cum_prob
  return add_one_smooth_bi

####################################

'''
Given a genre, this function will find all test books of this genre, 
and run them through the classifier to see if they are accurately identified

Params: 'children', 'history', or 'crime'
Returns: classifications, rate
'''
def genreClassification(true_genre):
  genre_models ={}
  for genre in ['children', 'history', 'crime']:
    genre_models[genre] = trainModel(genre)
  files = os.listdir(os.getcwd()+ '/test_books/' + true_genre)

  for f in files:
    test_tokens = tokenizedText([f], os.getcwd()+'/test_books/'+ true_genre)
    print genreClassifier(test_tokens, genre_models)


####################################

'''
Given a list of tokens, will classify which genre these tokens will most likely appear in.

Params: 
test_tokens, or list of tokens from test file
genre_models, a list of probability models from each genre we are looking at
Returns: classification in form of string
'''

def genreClassifier(test_tokens, genre_models):
  tokens = test_tokens
  most_common = dict(Counter(test_tokens).most_common())

  models = {
  'children': genre_models['children']['addone_uni'], 
  'history': genre_models['history']['addone_uni'], 
  'crime': genre_models['crime']['addone_uni']
  }

  probs = {'children':1, 'history': 1, 'crime': 1}
  for word, count in most_common.items():
    for genre in probs:
      probs[genre] *= genre_models[genre][word]
  return probs

###################################
# HELPER FUNCTIONS 
###################################

def trainModel(genre):
  files = os.listdir(os.getcwd()+ '/train_books/' + genre)
  x = tokenizedText(files, os.getcwd()+'/train_books/'+genre)
  #x = ["START", "this", "is", 'this', 'is', "my", "sample", "text", "END"]
  unigrams = unigram(x)
  unigram_counts = unigrams[0]
  unigram_prob = unigrams[1]
  bigrams = bigram(x, unigram_counts)
  bigram_counts = bigrams[0]
  bigrams_2d = bigrams[1]
  bigram_prob = bigrams[2]
  add_one = addOneSmoothingUnigram(unigram_counts, x)
  add_one_bi = addOneSmoothingBigram(unigram_counts,bigrams_2d)
  
  goodTuring_uni = goodTuringSmoothing('unigram', unigram_counts, unigram_prob)
  goodTuring_bi = goodTuringSmoothing('bigram', bigram_counts, bigrams_2d)
  
  #print goodTuring_bi

  #print perplexityUnigrams(unigram_prob, ['START', 'Alisha', 'is'], goodTuring_uni)
  print perplexityBigrams(bigram_prob, ['START', 'Alisha', 'is'], goodTuring_bi)
  # print "UNIGRAM"
  # for i in range(1,10):
  #   print randomSentence('unigram', unigram_prob)

  # print "BIGRAM"
  # for i in range(1,10):
  #   print randomSentence('bigram', bigram_prob)
  
  return {"unigram": unigram_prob, "bigram": bigram_prob, "addone_uni": add_one, "addone_bi": add_one_bi}

####################################

'''
Reads a list of textfiles and returns all tokens from the text.

Params: 
files: array of textfile names to go through
directory: string indicating where the files live relative to main.py

Returns:
Array of tokens (non-unique)
'''
def tokenizedText(files, directory):
  tokens =[]
  for filename in files:
    if '.txt' in filename:
      print "tokenizing ", filename
      lines = open(directory + '/'+ filename, 'r').read()
      sentences = re.compile(r'(?<=[.!?;])\s*').split(lines)
      sentences_with_tag = '';
      for sentence in sentences:
        sentences_with_tag += ' START ' + sentence + ' END '
      try:
        tokens += word_tokenize(sentences_with_tag.decode('utf8'))    
      except:
        print filename, " did not tokenize"
  return tokens

####################################

'''
Given a model type and probabilities, will calculate the cumulative probabilities. 
Needed for randomly sampling the model for random sentence generation
'''
def cumulativeProb(model, probabilities):
  cum_prob ={}
  if model == 'unigram':
    total = 0
    for k in probabilities:
      total += probabilities[k]
      cum_prob[k]=total
  
  if model =='bigram':
    #find cumulative probabilities for the second word, given first word
    for first_word in probabilities:
      total = 0
      cum_second_words = {}
      for second_word in probabilities[first_word]:
        total += probabilities[first_word][second_word]
        cum_second_words[second_word] = total
      cum_prob[first_word] = cum_second_words
  return cum_prob

def main():
  #genre = raw_input("Enter genre you would like to train model on (children, crime, or history): ") 
  #genreClassification ('history')
 
  # unigrams = unigram(t)
  # unigram_counts = unigrams[0]
  # unigram_prob = unigrams[1]
  # bigrams = bigram(t, unigram_counts)
  # bigram_counts = bigrams[0]
  # bigrams_2d = bigrams[1]
  # bigram_prob = bigrams[2]
 
  #perplexityUnigrams(unigram_counts, tokens)
  # print bigram_counts
  # print bigrams_2d
  # print bigram_prob
  
  trainModel('history')
  #genreClassifier(t, {})


main()