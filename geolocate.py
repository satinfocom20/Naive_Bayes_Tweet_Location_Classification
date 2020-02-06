#!/usr/bin/env python
'''
We implemented Naive Bayes model to calculate probability of train data to classify the locations
in test data. Below are the steps in our design.
1. Load both train and test data and pre-process it by removing special characters, numbers, non-
ascii codes et.,

2. Build the model by pre-calculating the probabilities using training data for the below formula
Probability to be calculated for each tweet text from test data is:
P(L=l | w1, w2, w3) = P(w1 | l) * P(w2 | l) * P(w3 | l) * P(l)
P(w1 | l) = Count of word w1 in location l / Count of all words in location l
P(l) = Count of records with location l / Count of all records
In this step in building model, we have stored count of all words in each location, count of all words
in all records(all locations), count of records for each location, count of all records, count of all
unique records(to be used for smoothing).


3. Read pre-processed each test record and calculate the probability using the formula given in
above step. Used laplace smoothing to avoid 0 probabilities as absence of some words in training data
may bring down the entire probability to 0.

4. Store the classify the tweet with location found with highest probability and compare it with location
in test data to calculate accuracy.


Results:
We have tried below clean ups in the data to ahceive the accuracy of close to 67%. We may need more
training data and identify more stop words to increase the accuracy.
(a) Not converting upper cases to lower cases and removing special characters yielded 61.4% accuracy
(b  Converting all upper cases to lower cases and removing special characters yielded 61.8% accuracy
(c) Removing all blanks/spaces/white spaces, special characters and convert all upper to lower cases yielded 63.8% accuracy
(d) Removing non ASCII characters and all others mentioned above yielded 64.45% accuracy
(e) Removing stop words and less frequent words yielded 65.20% accuracy
'''
import sys
from collections import defaultdict
import operator
import re
import  time
#from nltk.corpus import stopwords



class geolocate:
    # Initialize data structure
    def __init__(self):
        self.vocabulary = set()
        self.ptopwinl = {}
        self.train_dict = {}
        self.locfreq_dict = {}
        self.train_len = 0
        self.test_len = 0
        self.prob_loc = {}
        self.fileholder = []
        self.classified_loc = []
        self.winlvalcnt = {}
        self.count_all_words = {}

    # Load file and clean up
    def load_file(self, filename):
        self.fileholder = []
        file = open(filename, 'r')
        lines = file.readlines()
        lineconcat = ''
        lineprev = []
        for line in lines:
            if line and line != ' ' and line!= '\n':
                linesplit = line.strip().split()
                try:
                    if linesplit[0].find(',_') != -1:
                        lineprev.append(linesplit[0])
                        if len(lineprev) == 1:
                            lineconcat += line
                        else:
                            self.fileholder.append(lineconcat)
                            lineconcat = line
                    else:
                        lineconcat += line
                except IndexError:
                    continue
            else:
                continue

        self.fileholder.append(lineconcat)
        clean = self.clean_data(self.fileholder)
        return clean


    #Clean data by removing special characters and stop words
    def clean_data(self, rawdata):
        cleaned_data = []
        removed_splchar = []
        lessfreqwords = []
        stopwords = ['job','jobs','hiring','i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                     'your', 'yours', 'yourself','yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself','a', 'an', 'the','they','them','their','what','which','who',
                     'whom','this','that','these','those','am','is','are','was','were','be','been','being','have',
                     'has','had','having','do','does','did','doing','and','but','if','or','at','of','as','by','to',
                     'from','in','into','how','why','when','where','here','there','just','now', 'for', 'with',
                     'careerarc', 'on', 'with', 'st', 'amp', 'latest', 'click', 'day', 'opening', 'out', 'up',
                     'report','see', 'great', 'all', 'work', 'can', 'night', 'so', 'want', 'bw', 'nursing', 'healthcare',
                     'apply', 'center', 'anyone', 'any', 'sales', 'fit']
        ta = time.time()
        count = 0 # remove
        for line in rawdata:
            word = ''
            if line:
                count += 1
                print('1st processing: ', count)
                words_in_line = line.strip().split(' ')
                for w in words_in_line:
                    if w.find(',_') != -1:
                        word += w + ' '
                        continue
                    else:
                        # remove special characters, numbers and convert upper case to lower case
                        wordcleaned = ((re.sub('[^A-Za-z]+', '', w)).lower())
                        if wordcleaned in stopwords:
                            continue
                        else:
                            word +=  wordcleaned + ' '

                        # Adds all words to with it's frequency to find less frequent words and remove it
                        if wordcleaned not in self.count_all_words:
                            self.count_all_words[wordcleaned] = 1
                        else:
                            self.count_all_words[wordcleaned] += 1

                removed_splchar.append(word)
            else:
                continue
        print('time taken for 1st train scan: ', time.time() - ta)
        tb = time.time()
        #Less frequent words
        lessfreqwords = [k for k, v in self.count_all_words.items() if v == 1]
        #print('Less frequent words: ', len(lessfreqwords))
        count = 0
        for line in removed_splchar:
            count = count + 1
            #print('processing ', count)
            word = ''
            if line:
                words_in_line = line.strip().split(' ')
                for w in words_in_line:
                    if w in lessfreqwords:
                        continue
                    else:
                        word += w + ' '
                cleaned_data.append(word)
        print('time take for 2nd train scan: ', time.time() - tb)
        return cleaned_data


    #Build the model by pre calculating probabilities of training data
    def model_train(self, pre_procsd_train):
        #Count of all records in training dataset
        self.train_len = len(pre_procsd_train)

        #Count of records in each location stored in dictionary
        for line in pre_procsd_train:
            linesplit = line.strip().split(' ')
            if linesplit[0].find(',_') != -1:
                if linesplit[0] in self.locfreq_dict:
                    self.locfreq_dict[str(linesplit[0])] += 1
                else:
                    self.locfreq_dict[str(linesplit[0])] = 1

            for w in linesplit[1:]:
                if linesplit[0]+w in self.train_dict:
                    self.train_dict[linesplit[0]+w] += 1
                else:
                    self.train_dict[linesplit[0]+w] = 1

                #Vocabulary contains all unique words in training dataset
                self.vocabulary.add(w)

        # sum of all words for each location
        for location, val in self.locfreq_dict.items():
            self.winlvalcnt[location] = sum ( [v for k, v in self.train_dict.items() if k.startswith(location)] )


    #Classsify each tweet in test dataset using probability from train dataset
    def test_run(self, pre_procsd_test):
        self.test_len = len(pre_procsd_test)
        test_file = pre_procsd_test
        for line in test_file:
            linesplit = line.strip().split(' ')
            self.classified_loc.append([self.prob_calc(line), linesplit[0]])

        #print (self.classified_loc)
        #Print top 5 words in each location in the output
        for loc, val in self.locfreq_dict.items():
            print ('Top 5 words in ', loc)
            t = [[k, v] for k, v in self.ptopwinl.items() if k.startswith(loc)]
            p = sorted(t, key=lambda item : item[1], reverse=True)[:6]
            for w in p:
                print(w[0].split(' ')[1])
            print (' ')

        return self.classified_loc


    #Find the probability for each word to classify the tweet with location
    def prob_calc(self, testline):
       
        for location, v in self.locfreq_dict.items():
            p_w_in_l = 1.0
            #Calculate probability of each location with Laplace smoothing; Ex:P(Chicago,_IL), P(Washington,_DC) etc.,.
            p_l = (self.locfreq_dict[location] + 1.0) / float (((self.train_len) + len(self.vocabulary)))

            # calculate P(each word | each location) with Laplace smoothing;
            # Ex:P('game'|Chicago,_IL)*P('NBA'|Chicago,_IL)*P('happy'|Chicago,_IL)

            #word count is <5
            for w in testline.split(' '):
                #trying to ignore first word which is city name 
                if w.find(',_') == -1:
                    try:
                        p_w_in_l_num = self.train_dict[location+w] + 1.0
                    except:
                        p_w_in_l_num = 1.0
                    p_each_w_in_l = (p_w_in_l_num) / float((self.winlvalcnt[location] + len(self.vocabulary)) * 1.0)
                    p_w_in_l *= p_each_w_in_l
                    #Store probabilities of all words in a given location(EX: P(Chicago,IL | w)to display top 5 words in output.
                    self.ptopwinl[location+' '+w] = p_each_w_in_l
                else:
                    continue
            #Final probability for P(each word in test | each Location) * P(each Location)
            #Ex: P('game'|Chicago,_IL)*P('NBA'|Chicago,_IL)*P('happy'|Chicago,_IL)*P(Chicago,_IL)
            self.prob_loc[location] = (p_w_in_l * p_l * 1.0)
        #Return the location with highest probability
        label = sorted(self.prob_loc.items(), key=operator.itemgetter(1), reverse=True)
        return label[0][0]


    #Calculate accuracy of the classified locations
    def accuracy_writeout(self, results, file_output):
        #Write output to output file
        filetest = open(file_test, 'r')
        fileout = open(file_output, 'a')
        linetest = filetest.readlines()
        count = 0
        for line in linetest:
            try:
                linesplit = line.strip().split()
                if linesplit[0].find(',_') != -1:
                    fileout.write(str(self.classified_loc[count][0])+' '+line)
                    count = count + 1
                else:
                    fileout.write(line)
            except IndexError:
                fileout.write(line)
                continue
        #Calculae accuracy
        correct = 0
        for r in results:
            if len(set(r)) == 1:
                correct = correct + 1
        accu_per = (correct / float(self.test_len)) * 100
        print ('Accuracy Score: ', accu_per)

#call the class
g = geolocate()
#Get train file name, test file name and output file names as input from user
'''file_train = sys.argv[1]
file_test = sys.argv[2]
file_output = sys.argv[3]'''

#file_train = "tweets.train.clean.txt"
file_train = "tweets.train.clean.txt"
file_test = "tweets.test.clean.txt"
file_output = "output.txt"
t1 = time.time()
pre_procsd_train = g.load_file(file_train)
print(pre_procsd_train)
print('Time taken for Train data cleaning: ', time.time()-t1)
t2 = time.time()
pre_procsd_test = g.load_file(file_test)
print('Time taken for Test data cleaning: ', time.time()-t2)
t3 = time.time()
g.model_train(pre_procsd_train)
print('Time taken to train Model: ', time.time()-t3 )
t4 = time.time()
results = g.test_run(pre_procsd_test)
print('Time taken for Test Run: ', time.time()-t4)
g.accuracy_writeout(results, file_output)
