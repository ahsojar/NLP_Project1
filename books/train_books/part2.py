#######################
# ALISHA SOJAR AND SHREYA SITARAMAN
# Place file in training_books directory
# cd to training_books directory
# run python part2.py 
#######################

import re
import os
import random
import nltk
import collections

from collections import Counter
from nltk.tokenize import word_tokenize

unigram_counts = {} #{unigram: count, unigram2: count}
unigram_prob = {} #{unigram: probability, unigram2: probability}
cum_unigram_prob = {} #cumulative unigram probability

bigram_counts = {} #{'word1 word2': count, 'word2 word3': count}
bigrams = {} #2D bigram counts {word1: {word2: count, word3: count}, word2: {word4: count}}
bigram_prob = {} #probability of each bigram, 2D
cum_bigram_prob = {} #culmulative bigram probability, 2D

add_one_uni = {} # dictionary of unigram probabilities with add-one smoothing
add_one_bigram= {} #dictionary of bigram probabilities with add-one smoothing


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

# def getCounts(tokens):
#   return dict(Counter(tokens))

def unigram(tokens):
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

  total = 0
  for k in unigram_counts:
    total += unigram_prob[k]
    cum_unigram_prob[k]=total

  return cum_unigram_prob

def bigram(tokens):
  prev_word = 'START'
  if unigram_counts == {}:
    unigram(tokens)

  for word in tokens:
  #create bigram dictionary with 'word1 word2' as key
    bigram = prev_word + ' ' + word
    if bigram in bigram_counts:
        bigram_counts[bigram] += 1
    else:
        bigram_counts[bigram] = 1

    #create 2 dimensional bigrams
    if prev_word in bigrams:
      if word in bigrams[prev_word]:
        bigrams[prev_word][word] += 1
      else:
        bigrams[prev_word][word] = 1
    else:
      bigrams[prev_word] = {}
      bigrams[prev_word][word] = 1
  
    prev_word = word

  #normalize the 2D bigram counts (divide by probability of first word)
  for first_word in bigrams:
    second_words = bigrams[first_word]
    normalized_words = {}
    for w in second_words:
      if w != 'START':
        normalized_words[w] = second_words[w]/ float(unigram_counts[first_word])
        bigram_prob[first_word] = normalized_words

  #find cumulative probabilities for the second word, given first word
  for first_word in bigram_prob:
    total = 0
    cum_second_words = {}
    for second_word in bigram_prob[first_word]:
      total += bigram_prob[first_word][second_word]
      cum_second_words[second_word] = total
    cum_bigram_prob[first_word] = cum_second_words

  return cum_bigram_prob

def randomSentence(model):
  words = 1
  added = False
  prev_word = 'START'
  sentence = 'START'
  if model == 'unigram':
    word_prob = cum_unigram_prob
  elif model == 'bigram':
    word_prob = cum_bigram_prob[prev_word]

  while (not 'END' in sentence) and (words < 50):
    if model == 'bigram':
      word_prob = cum_bigram_prob[prev_word]
    random_p = random.uniform(0,1)
    added = False
    for key,value in word_prob.items():
        if value > random_p - 0.001 and value < random_p + 0.001 and not added and key != 'START':
            words += 1
            sentence += ' ' + key
            added = True
            prev_word = key
  return sentence


def addOneSmoothingUnigram(unigram_counts, tokens):
    len_corpus = len(tokens)
    len_vocab = len(unigram_counts)
    cum_prob = 0

    add_one_smooth_uni = {}
    for key,value in unigram_counts.items():
        add_one_smooth_uni[key] = float(value + 1.0) / (len_vocab+len_corpus)  
        cum_prob += add_one_smooth_uni[key] 
    add_one_smooth_uni['UNKNOWN'] = 1-cum_prob
    return add_one_smooth_uni

def addOneSmoothingBigram(unigram_counts, bigram_counts):
    vocab_length = len(unigram_counts)
    cum_prob = 0

    add_one_smooth_bi = {}
    for key,value in bigram_counts.items():
        #Add one in the numerator and add vocab_length in the denominator
        add_one_smooth_bi[key] = float(value + 1.0) / (float(unigram_counts[key.split()[0]]) + vocab_length)
        cum_prob += add_one_smooth_bi[key] 
    add_one_smooth_bi['UNKNOWN'] = 1-cum_prob
    return add_one_smooth_bi

def main():
  #genre = raw_input("Enter genre you would like to train model on (children, crime, or history): ") 
  #files = os.listdir(os.getcwd()+ '/' + genre)
  #x = tokenizedText(files, os.getcwd()+'/'+genre)
  x = ["START", "this", "is", "my", "sample", "text", "END"]
  print unigram(x)
  print addOneSmoothingUnigram(unigram_counts, x)
  bigram(x)
  print bigram_prob
  print addOneSmoothingBigram(unigram_counts,bigram_counts)

  # print "Unigram sentence: "
  # for i in range(1,15):
  #   print randomSentence('unigram')

  # print "Bigram Sentences:"
  # for i in range(1,15):
  #   print randomSentence('bigram')

main()