#######################
# ALISHA SOJAR AND SHREYA SITARAMAN
# Place file in training_books directory
# cd to training_books directory
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

# Takes in an array of filenames
# Return a list of tokens from this file
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

def randomSentence(model, probabilities):
  cum_prob = cumulativeProb(model, probabilities)
  
  words = 1
  added = False
  prev_word = 'START'
  sentence = 'START'
  if model == 'unigram':
    word_prob = cum_prob
  elif model == 'bigram':
    word_prob = cum_prob[prev_word]

  while (not 'END' in sentence) and (words < 50):
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


def goodTuringSmoothingUnigram(unigram_counts):
  goodTuring = dict()
  sample["<UNK>"] = 0

  N = sample.values()

  for k in sample:
   word_count = sample[k]  
   if word_count == 1:
    print word_count

  return goodTuring


def perplexityUnigrams(unigram_prob, tokens):
  total = 0
  word_count = len(tokens)

  for w in tokens:
    if w in unigram_prob:
      x = -math.log(unigram_prob[w])
      total = total + x
    else:
      x = -math.log(unigram_prob['<UNK>'])
      total = total + x
  
  perplexity = total/float(word_count)
  return perplexity

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

def genreClassification(testfile):
    tokens = tokenizedText(testfile)

def trainModel(genre):
  files = os.listdir(os.getcwd()+ '/train_books/' + genre)
  x = tokenizedText(files, os.getcwd()+'/train_books/'+genre)
  #x = ["START", "this", "is", "my", "sample", "text", "END"]
  unigrams = unigram(x)
  unigram_counts = unigrams[0]
  unigram_prob = unigrams[1]
  bigrams = bigram(x, unigram_counts)
  bigram_counts = bigrams[0]
  bigrams_2d = bigrams[1]
  bigram_prob = bigrams[2]
  add_one = addOneSmoothingUnigram(unigram_counts, x)
  add_one_bi = addOneSmoothingBigram(unigram_counts,bigrams_2d)

  print perplexityUnigrams(add_one, ['START', 'Alisha', 'is'] )
  print randomSentence('bigram', bigram_prob)
  return unigram_counts

def main():
  genre = raw_input("Enter genre you would like to train model on (children, crime, or history): ") 
  trainModel(genre)

main()