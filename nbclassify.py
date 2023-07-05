import glob
import sys
import os
import fnmatch
import re
from collections import defaultdict
import math


class naive_bayes_testing:
	def __init__(self,vocabulary,prior_probablities,cond_prob_positive,cond_prob_negative,cond_prob_truthful,cond_prob_deceptive):
		self.vocabulary=vocabulary
		self.prior_probablities=prior_probablities
		self.cond_prob_positive=defaultdict(int,cond_prob_positive)
		self.cond_prob_negative=defaultdict(int,cond_prob_negative)
		self.cond_prob_truthful=defaultdict(int,cond_prob_truthful)
		self.cond_prob_deceptive=defaultdict(int,cond_prob_deceptive)
		self.stop_words ={' ','','\n',"you'd", 'o', "it's", 'had','because', "should've", "hadn't", 'about', 'where','it', 'him', 'my', 'are', 'hadn', 'who', 'not', 'don', "haven't", 'and', 'am', 'has', 'now', 'into', 'against','further', 'doing', 'wouldn', 'his', 'were', 'for', 'down', 'why', 'most', 'so', 'just', 'i', 'of', 'themselves', 're', "couldn't",'if', 'they', 'herself', 'them', 'after', 'mightn', 'ma', 'by', 'weren', "you're",'d', 'this', "hasn't", "wasn't", 'aren', 'all', 'how', 'mustn', 'only', 'hasn', 'on', 'no', 'ours', "needn't", 'a', "shan't", 'having', 'over', 'than', 'from', 's', 'itself', 'hers', 'do', 'yours', 'to', 'their', 'me', 'our', 'below', "won't", 'needn', 'during', 'very', 'm', 'or', 'should', 'y', 'will', 'those', 'shan', 'off', "isn't", "mightn't", 'ourselves', 'yourselves', 'under', 'been', 'other', 'nor', 'll', 'above', "that'll", 'few', 'that', "doesn't", 'your', 'both', 'doesn', 'didn', "you've", "shouldn't", 'ain', 'what', "she's", 'these', 'as', 'through', 'theirs', 'before', 'couldn', "weren't", 'himself', 'her', 'while', 'again', 'too', 'more', 'myself', 'here', 'until', 'isn', 'whom', 'own', 've', 'did', "mustn't", 'in', 'there', 'but', 'any', 'can', "wouldn't", 'does', 'be', 'wasn', "you'll", "don't", "aren't", "didn't", 'won', 'she', 'the', 'once', 'then', 'being', 'between', 'you', 'was', 'such', 'when', 'same', 'an', 'shouldn', 'haven', 'which', 'yourself', 'at', 't', 'its', 'have', 'out', 'some', 'he', 'each', 'up', 'with', 'is', 'we'}

	def classify(self,paths):
		
		f1 = open('nboutput.txt',"w")

		for file_path in paths:
			#print(file_path,end='\n\n')
			f2=open(file_path,'r')
			token_list =f2.read().lower().split()
			#print(token_list)
			tokens_in_the_file = []
			temp_tokens =[]
			for token in token_list:
				if token not in self.stop_words:
					token = re.sub('[^a-zA-Z0-9]',"",token)
					token = re.sub('-','',token)
					if token not in self.stop_words:
						temp_tokens.append(token)
						tokens_in_the_file.append(token)
			f2.close()
			
			positive_score,negative_score,truthful_score,deceptive_score = 0,0,0,0
			
			positive_score+=math.log(self.prior_probablities['positive'],10)
			negative_score+=math.log(self.prior_probablities['negative'],10)
			truthful_score+=math.log(self.prior_probablities['truthful'],10)
			deceptive_score+=math.log(self.prior_probablities['deceptive'],10)


			for token in tokens_in_the_file:
				
				if self.cond_prob_positive[token]!=0:
					positive_score+=math.log(self.cond_prob_positive[token],10)
				
				if self.cond_prob_negative[token]!=0:
					negative_score+=math.log(self.cond_prob_negative[token],10)
				
				if self.cond_prob_truthful[token]!=0:
					truthful_score+=math.log(self.cond_prob_truthful[token],10)
				
				if self.cond_prob_deceptive[token]!=0:
					deceptive_score+=math.log(self.cond_prob_deceptive[token],10)

			label_a,label_b='',''
			#print('positive_score',positive_score)
			#print('negative_score',negative_score)
			#print('truthful_score',truthful_score)
			#print('deceptive_score',deceptive_score)
			#print('\n\n\n\n')
			if truthful_score>=deceptive_score:
				label_a = 'truthful'
			else:
				label_a='deceptive'
			
			if positive_score>=negative_score:
				label_b='positive'
			else:
				label_b='negative'

			f1.write(label_a+' '+label_b+' '+file_path+'\n')
		
		f1.close()

def main():
	
	f1=open('nbmodel.txt','r')
	vocabulary = eval(f1.readline())
	prior_probablities= eval(f1.readline())
	cond_prob_positive=eval(f1.readline())
	cond_prob_negative=eval(f1.readline())
	cond_prob_truthful=eval(f1.readline())
	cond_prob_deceptive=eval(f1.readline())
	f1.close()

	nb = naive_bayes_testing(vocabulary,prior_probablities,cond_prob_positive,cond_prob_negative,cond_prob_truthful,cond_prob_deceptive)

	input_path = sys.argv[1]
	paths = glob.glob(os.path.join(input_path,'*/*/*/*.txt'))
	nb.classify(paths)

main()

