#######################
# ALISHA SOJAR AND SHREYA SITARAMAN
# TO RUN: place main.py in 'train_books' directory
# in command line, run 'python main.py <genre>'
# <genre> can be 'children', 'crime', or 'history'
#######################

import re
import os
import sys
import random
import math
import nltk

from nltk.tokenize import word_tokenize

unigrams = {} #{unigram: count, unigram2: count}
unigram_prob = {} #{unigram: probability, unigram2: probability}
cum_unigram_prob = {} #cumulative unigram probability

bigrams = {} #bigram counts {bigram: count, bigram:count }
bigram_prob = {} #probability of each bigram  {bigram: probability, bigram2: probability}
cum_bigram_prob = {} #culmulative bigram probability




def unigram(genre):
  word_count = 0
  # Open & read file
  for filename in os.listdir(os.getcwd()+ '/' + genre):
    if '.txt' in filename:
      lines = open(genre + '/' + filename, 'r').read()
      sentences = re.compile(r'(?<=[.!?;])\s*').split(lines)
      sentences_with_tag = '';
      
      for sentence in sentences:
        sentences_with_tag += 'AAASTARTAAA '+sentence+' AAAENDAAA '
      tokens = nltk.word_tokenize(sentences_with_tag.decode('utf8'))
      word_count += len(tokens)
      for word in tokens:
        #Create unigram counts
        if word in unigrams:
          unigrams[word] += 1
        else:
          unigrams[word] = 1

    #Normalize unigram probabilities from 1 to 0
    for k in unigrams:
      unigram_prob[k] = unigrams[k]/float(word_count)

    total = 0
    for k in unigrams:
      total += unigram_prob[k]
      cum_unigram_prob[k]=total

  return cum_unigram_prob



def bigram(genre):
  # Open & read file
  prev_word = "AAASTARTAAA"

 # Open & read file
  for filename in os.listdir(os.getcwd()+ '/' + genre):
    if '.txt' in filename:
      lines = open(genre + '/' + filename, 'r').read()
      sentences = re.compile(r'(?<=[.!?;])\s*').split(lines)
      sentences_with_tag = '';
      
      for sentence in sentences:
        sentences_with_tag += 'AAASTARTAAA '+sentence+' AAAENDAAA '
      tokens = nltk.word_tokenize(sentences_with_tag.decode('utf8'))
       
      for word in tokens:
        #Create bigram counts
        if prev_word in bigrams:
          if word in bigrams[prev_word]:
            bigrams[prev_word][word] += 1
          else:
            bigrams[prev_word][word] = 1
        else:
          bigrams[prev_word] = {}
          bigrams[prev_word][word] = 1
        prev_word = word

  #normalize the bigram counts (divide by probability of first word)
  for first_word in bigrams:
    second_words = bigrams[first_word]
    normalized_words = {}
    for w in second_words:
      if w != '<s>':
        normalized_words[w] = second_words[w]/ float(unigrams[first_word])
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


def goodTuring(bigram):
  



def randomSentence(model):
  sentence = 'AAASTARTAAA'
  words = 1
  added = False

  if model == 'unigram':
    while (not 'AAAENDAAA' in sentence) and (words < 30):
              random_p = random.uniform(0,1)
              added = False
              for key,value in cum_unigram_prob.items():
                  if value > random_p - 0.001 and value < random_p + 0.001 and not added and key != 'AAASTARTAAA':
                      words += 1
                      sentence += ' ' + key
                      added = True
    return sentence

  if model == 'bigram':
    prev_word = 'AAASTARTAAA'
    while (not 'AAAENDAAA' in sentence) and (words < 30):
              random_p = random.uniform(0,1)
              added = False
              for key,value in cum_bigram_prob[prev_word].items():
                  if value > random_p - 0.001 and value < random_p + 0.001 and not added and key != 'AAASTARTAAA':
                      words += 1
                      sentence += ' ' + key
                      added = True
                      prev_word = key
    return sentence


#Pass in genre argument when running python
genre = sys.argv[1]
unigram(genre)
print bigram(genre)
# bigram(genre)

# print "Unigram Random Sentence:"
# for i in range(0,5):
#   print randomSentence('unigram')

print "Bigram Random Sentence:"
for i in range(0,5):
  print randomSentence('bigram')