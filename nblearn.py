import sys
import os
import fnmatch
import re
from collections import defaultdict

class naive_bayes_training:
	def __init__(self):
		self.vocabulary = set()
		
		self.prior_probablities={"positive":0,"negative":0,"truthful":0,"deceptive":0}
		
		self.positive_filepath=[]
		self.negative_filepath=[]
		self.truthful_filepath=[]
		self.deceptive_filepath=[]

		self.positive_words=defaultdict(int)
		self.negative_words=defaultdict(int)
		self.truthful_words=defaultdict(int)
		self.deceptive_words=defaultdict(int)

		self.cond_prob_positive={}
		self.cond_prob_negative={}
		self.cond_prob_truthful={}
		self.cond_prob_deceptive={}

		self.stop_words ={' ','','\n',"you'd", 'o', "it's", 'had','because', "should've", "hadn't", 'about', 'where','it', 'him', 'my', 'are', 'hadn', 'who', 'not', 'don', "haven't", 'and', 'am', 'has', 'now', 'into', 'against','further', 'doing', 'wouldn', 'his', 'were', 'for', 'down', 'why', 'most', 'so', 'just', 'i', 'of', 'themselves', 're', "couldn't",'if', 'they', 'herself', 'them', 'after', 'mightn', 'ma', 'by', 'weren', "you're",'d', 'this', "hasn't", "wasn't", 'aren', 'all', 'how', 'mustn', 'only', 'hasn', 'on', 'no', 'ours', "needn't", 'a', "shan't", 'having', 'over', 'than', 'from', 's', 'itself', 'hers', 'do', 'yours', 'to', 'their', 'me', 'our', 'below', "won't", 'needn', 'during', 'very', 'm', 'or', 'should', 'y', 'will', 'those', 'shan', 'off', "isn't", "mightn't", 'ourselves', 'yourselves', 'under', 'been', 'other', 'nor', 'll', 'above', "that'll", 'few', 'that', "doesn't", 'your', 'both', 'doesn', 'didn', "you've", "shouldn't", 'ain', 'what', "she's", 'these', 'as', 'through', 'theirs', 'before', 'couldn', "weren't", 'himself', 'her', 'while', 'again', 'too', 'more', 'myself', 'here', 'until', 'isn', 'whom', 'own', 've', 'did', "mustn't", 'in', 'there', 'but', 'any', 'can', "wouldn't", 'does', 'be', 'wasn', "you'll", "don't", "aren't", "didn't", 'won', 'she', 'the', 'once', 'then', 'being', 'between', 'you', 'was', 'such', 'when', 'same', 'an', 'shouldn', 'haven', 'which', 'yourself', 'at', 't', 'its', 'have', 'out', 'some', 'he', 'each', 'up', 'with', 'is', 'we'}
		self.num_positive=0
		self.num_negative=0
		self.num_truthful=0
		self.num_deceptive=0
		self.num_docs=0

		
	def get_data(self,paths):
		for path in paths["p"]:
			for root,directory_names,file_names in os.walk(path):
					for file in fnmatch.filter(file_names, '*.txt'):
						f_path = os.path.join(root, file)
						self.positive_filepath.append(os.path.join(root, file))
						self.num_positive+=1
		for path in paths["n"]:
			for root,directory_names,file_names in os.walk(path):
					for file in fnmatch.filter(file_names, '*.txt'):
						self.negative_filepath.append(os.path.join(root, file))
						self.num_negative+=1
		for path in paths["t"]:
			for root,directory_names,file_names in os.walk(path):
					for file in fnmatch.filter(file_names, '*.txt'):
						self.truthful_filepath.append(os.path.join(root, file))
						self.num_truthful+=1
		for path in paths["d"]:
			for root,directory_names,file_names in os.walk(path):
					for file in fnmatch.filter(file_names, '*.txt'):
						self.deceptive_filepath.append(os.path.join(root, file))
						self.num_deceptive+=1

		#print(self.num_positive)
		#print(self.num_negative)
		#print(self.num_truthful)
		#print(self.num_deceptive)
		self.num_docs=int((self.num_positive+self.num_negative+self.num_truthful+self.num_deceptive)/2)

		
	def get_tokens(self):
		for file_path in self.positive_filepath:
			#print(file_path)
			f1 = open(file_path, "r")
			token_list = f1.read().lower().split()
			#print(token_list,end='\n\n')
			#print('\n\n')
			#print(file_path,end='\n\n')
			
			for token in token_list:
				if token:
					token = re.sub('[^a-zA-Z0-9]',"",token)
					token = re.sub('-','',token)
					if token not in self.stop_words:
						self.positive_words[token]+=1
						#temp_token_list.append(token)
						self.vocabulary.add(token)
			f1.close()
		#print(len(self.vocabulary))
		#print(len(self.positive_words))
		#print(self.positive_words)
		for file_path in self.negative_filepath:
			#print(file_path)
			f1 = open(file_path, "r")
			token_list = f1.read().lower().split()
			#print(token_list,end='\n\n')
			for token in token_list:
				if token:
					token = re.sub('[^a-zA-Z0-9]',"",token)
					token = re.sub('-','',token)
					if token not in self.stop_words:
						self.negative_words[token]+=1
						self.vocabulary.add(token)
			f1.close()
		for file_path in self.truthful_filepath:
			#print(file_path)
			f1 = open(file_path, "r")
			token_list = f1.read().lower().split()
			#print(token_list,end='\n\n')
			for token in token_list:
				if token:
					token = re.sub('[^a-zA-Z0-9]',"",token)
					token = re.sub('-','',token)
					if token not in self.stop_words:
						self.truthful_words[token]+=1
						self.vocabulary.add(token)
			f1.close()
		for file_path in self.deceptive_filepath:
			#print(file_path)
			f1 = open(file_path, "r")
			token_list = f1.read().lower().split()
			#print(token_list,end='\n\n')
			for token in token_list:
				
				if token:
					token = re.sub('[^a-zA-Z0-9]',"",token)
					token = re.sub("-",'',token)
					
					if token not in self.stop_words:
						#print(token)
						self.deceptive_words[token]+=1
						self.vocabulary.add(token)
			f1.close()
		#print(len(self.vocabulary))
		#print(self.deceptive_words)
		#print(self.negative_words)
		#print(self.positive_words)
		#print(self.truthful_words)
	def get_count_of_dict_values(self,dict_):
		count =0
		for i in dict_:
			count+=dict_[i]
		return count

	def get_probablities(self):
		self.prior_probablities["positive"]=self.num_positive/self.num_docs
		self.prior_probablities["negative"]=self.num_negative/self.num_docs
		self.prior_probablities["truthful"]=self.num_truthful/self.num_docs
		self.prior_probablities["deceptive"]=self.num_deceptive/self.num_docs
		#print(self.prior_probablities)
		V = len(self.vocabulary)
		p_total,n_total,t_total,d_total= self.get_count_of_dict_values(self.positive_words),self.get_count_of_dict_values(self.negative_words),self.get_count_of_dict_values(self.truthful_words),self.get_count_of_dict_values(self.deceptive_words)

		for word in self.vocabulary:
			self.cond_prob_positive[word]=(self.positive_words[word]+1)/(p_total+V)
			self.cond_prob_negative[word]=(self.negative_words[word]+1)/(n_total+V)
			self.cond_prob_truthful[word]=(self.truthful_words[word]+1)/(t_total+V)
			self.cond_prob_deceptive[word]=(self.deceptive_words[word]+1)/(d_total+V)
		#print(self.cond_prob_deceptive)
		f1=open('nbmodel.txt','w')
		#print(len(self.vocabulary))
		f1.write(str(self.vocabulary))
		f1.write('\n')
		f1.write(str(self.prior_probablities))
		f1.write('\n')
		f1.write(str(self.cond_prob_positive))
		f1.write('\n')
		f1.write(str(self.cond_prob_negative))
		f1.write('\n')
		f1.write(str(self.cond_prob_truthful))
		f1.write('\n')
		f1.write(str(self.cond_prob_deceptive))
		f1.close()


def main():				
	input_path = sys.argv[1]
	positive_truthful = input_path+"/positive_polarity/truthful_from_TripAdvisor/"
	positive_deceptive = input_path+"/positive_polarity/deceptive_from_MTurk/"
	negative_truthful = input_path+"/negative_polarity/truthful_from_Web/"
	negative_deceptive = input_path+"/negative_polarity/deceptive_from_MTurk/"

	paths ={"p":[positive_truthful,positive_deceptive],"n":[negative_truthful,negative_deceptive],"t":[positive_truthful,negative_truthful],"d":[positive_deceptive,negative_deceptive]}

	nb = naive_bayes_training()
	nb.get_data(paths)
	nb.get_tokens()
	nb.get_probablities()
	#print(len(nb.vocabulary))

main()






