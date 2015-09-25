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
  """
  Computes the unigram model given a set of tokens.

  Params:
  tokens: array of tokens in text

  Returns:
  result[0] is dicionary of unigram word counts
  result[2] is dictionary of the unigram probabilities
  """
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

def bigram(tokens, unigram_counts):
  """
  Computes the bigram model given a set of tokens.

  Params:
  tokens: array of tokens in text
  unigram_counts: dictionary of wordcounts of these tokens

  Returns:
  result[0] as the bigram counts
  result[1] as the bigram counts two dimesnional
  result[2] as the bigram probabilities
  """
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

def randomSentence(model, probabilities):
  """
  Generates a random sentence given a probability model of words.

  Params:
  model: pass in string 'unigram' or 'bigram' to indicate type of model
  probabilties: dictionary of probabilties

  Returns:
  a string, the random sentence
  """
  cum_prob = cumulativeProb(model, probabilities)
  
  words = 1
  added = False
  prev_word = 'START'
  sentence = 'START'
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

def goodTuringSmoothing(model, ngram_counts, ngram_prob):
  """
  Recomputes ngram probabilities using Good Turing smoothing.

  Params:
  model: pass in string 'unigram' or 'bigram' to indicate type of model
  ngram_counts: dictionary of ngram counts
  ngram_prob: dictionary of ngram probabilities

  Returns:
  dictionary of Good Turing ngram probabilities
  """
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

def perplexityUnigrams(unigram_prob, tokens):
  """
  Computes perpelxity for a test set against a training set unigram probability model.

  Params:
  unigram_prob: dictionary of unigram probibilities
  tokens: tokens in test set

  Returns:
  float, perplexity value
  """
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

####################################

def perplexityBigrams(bigram_prob, tokens):
  """
  Computes perplexity for a test set against a training set bigram probability model.

  Params:
  bigram_prob: dictionary of bigram probibilities
  tokens: tokens in test set

  Returns:
  float, perplexity value
  """
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
        x = -math.log(bigram_prob['<UNK>'])
        total = total + x
    not_first_word = True
  
  perplexity = total/float(word_count)
  return perplexity

####################################

def addOneSmoothingUnigram(unigram_counts, tokens):
  """
  Computes the unigram model given a set of tokens
  Uses add-one smoothing, so also handles case of unknown words.

  Params:
  unigram counts: dictionary of unigram counts
  tokens: list of tokens in training set

  Returns:
  dictionary of add-one smoothing unigram probabilities
  """
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

def addOneSmoothingBigram(unigram_counts, bigrams):
  """
  Computes the bigram model with add-one smoothing, which can now handle unseen bigrams.

  Params:
  unigram counts: dictionary of unigram counts for these tokens
  bigrams: dictionary of bigram counts in training set

  Returns:
  dictionary of add-one smoothing bigram probabilities
  """
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

def genreClassifier(test_tokens, genre_models):
  
  """
  Given a list of tokens, will classify which genre these tokens will most likely appear in.

  Params: 
  test_tokens, or list of tokens from test file
  genre_models, a list of probability models from each genre we are looking at

  Returns: classification in form of string
  """
  tokens = test_tokens
  most_common = Counter(test_tokens).most_common()
  top100 = [x[0] for x in most_common]
  top100 = top100[:100]

  models = {
  'children': genre_models['children']['good_turing_uni'], 
  'history': genre_models['history']['good_turing_uni'], 
  'crime': genre_models['crime']['good_turing_uni']
  }

  probs = {'children':1, 'history': 1, 'crime': 1}
  for word in top100:
    for genre in probs:
      if word in models[genre]:
        probs[genre] *= models[genre][word]
  print probs
  return max(probs, key=probs.get)

###################################
# HELPER FUNCTIONS 
###################################
  

def runSentenceGenerator(genre):
  """
  Runs the random sentence generator on the corpus of the genre given.

  Params: 
  genre: string; either 'history', 'children', or 'crime'
  
  """
  model = trainModel(genre)

  print "UNIGRAM sentences"
  for i in range(1,10):
    print randomSentence('unigram', model['unigram'])

  print "BIGRAM sentences"
  for i in range(1,10):
    print randomSentence('bigram', model['bigram'])

####################################

def runPerplexity(test_genre):
  """
  Runs the perplexity demo (both unigram and bigram) on all test files of genre given. 
  Uses good-turing smoothing.

  Params: 
  test_genre: string; either 'history', 'children', or 'crime'
  """
  genre_models ={}
  genres = ['children', 'history', 'crime']
  
  for genre in genres:
    genre_models[genre] = trainModel(genre)
  
  #get test files for this genre
  files = os.listdir(os.getcwd()+ '/test_books/' + test_genre)
  for f in files:
    if ".txt" in f:
      for g in genres:
        test_tokens = tokenizedText([f], os.getcwd()+'/test_books/'+ test_genre)
        print "Perplexity for " + f + " against " + g +" good turing unigram model:"
        print perplexityUnigrams(genre_models[g]['good_turing_uni'], test_tokens)

        print "Perplexity for " + f + " against " + g +" good turing bigram model:"
        print perplexityBigrams(genre_models[g]['good_turing_bi'], test_tokens)


####################################

def runGenreClassification():
  """
  Given a genre, this function will find all test books of this genre, 
  and run them through the classifier to see if they are accurately identified

  Params: none

  Returns: classifications, probabilities
  """
  genres = ['children', 'history', 'crime']

  genre_models ={}
  for genre in genres:
    genre_models[genre] = trainModel(genre)
  
  for true_genre in genres:
    files = os.listdir(os.getcwd()+ '/test_books/' + true_genre)
    for f in files:
      if '.txt' in f:
        print "Genre classification for " + f + ":"
        test_tokens = tokenizedText([f], os.getcwd()+'/test_books/'+ true_genre)
        print "Classification is: " + genreClassifier(test_tokens, genre_models)
    
####################################

def trainModel(genre):
  """
  Given a genre, this will look at all the training files in that specific corpus.
  It will then create both unigram and bigram models, and also run the different smoothing techniques.

  Params: 
  genre: string, 'children', 'history', or 'crime'. Defines the corpus for this model
  
  Return:
  A dictionary of the different types of probability models:
    "unigram": unigram model, unsmoothed
    "bigram": bigram model, unsmoothed
    "addone_uni": unigram model, add-one smoothing
    "addone_bi": bigram model, add-one smoothing
    "good_turing_uni":  unigram model, good turing smoothing
    "good_turing_bi": bigram model, good turing smoothing 
  """

  print "Training on " + genre + " corpus..."
  files = os.listdir(os.getcwd()+ '/train_books/' + genre)
  x = tokenizedText(files, os.getcwd()+'/train_books/'+genre)
  unigrams = unigram(x)
  unigram_counts = unigrams[0]
  unigram_prob = unigrams[1]
  bigrams = bigram(x, unigram_counts)
  bigram_counts = bigrams[0]
  bigrams_2d = bigrams[1]
  bigram_prob = bigrams[2]
  add_one_uni = addOneSmoothingUnigram(unigram_counts, x)
  add_one_bi = addOneSmoothingBigram(unigram_counts,bigrams_2d)
  good_turing_uni = goodTuringSmoothing('unigram', unigram_counts, unigram_prob)
  good_turing_bi = goodTuringSmoothing('bigram', bigram_counts, bigrams_2d)
  
  return {
    "unigram": unigram_prob, 
    "bigram": bigram_prob, 
    "addone_uni": add_one_uni, 
    "addone_bi": add_one_bi,
    "good_turing_uni":good_turing_uni,
    "good_turing_bi": good_turing_bi
  }

####################################

def tokenizedText(files, directory):
  """
  Reads a list of textfiles and returns all tokens from the text.

  Params: 
  files: array of textfile names to go through
  directory: string indicating where the files live relative to main.py

  Returns:
  Array of tokens (non-unique)
  """
  tokens =[]
  for filename in files:
    if '.txt' in filename:
      lines = open(directory + '/'+ filename, 'r').read()
      sentences = re.compile(r'(?<=[.!?;])\s*').split(lines)
      sentences_with_tag = '';
      for sentence in sentences:
        sentences_with_tag += ' START ' + sentence + ' END '
      try:
        tokens += word_tokenize(sentences_with_tag.decode('utf8'))    
      except:
        pass
  return tokens

####################################

def cumulativeProb(model, probabilities):
  """
  Given a model type and probabilities, will calculate the cumulative probabilities. 
  Needed for randomly sampling the model for random sentence generation
  """
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

####################################

def main():
  """
  Main method to run demos
  """
  demo = -1
  while int(demo) != 0:
    print('Choose one of the Following Demos: ')
    print('1 - Random Sentence Generation')
    print('2 - Measure Perplexity')
    print('3 - Genre Classification')
    print('0 - Exit')
    print('')

    demo = input('Enter input here: ')
    if int(demo) == 1:
      genre = raw_input('Choose a genre to train on: history, children or crime: ')
      runSentenceGenerator(genre)
        
    elif int(demo) == 2:
      test_genre = raw_input('Choose the genre of the test_book files we want to compute perplexity for: (history, children or crime): ')
      runPerplexity(test_genre)

    elif int(demo) == 3:
      runGenreClassification()
          
main()