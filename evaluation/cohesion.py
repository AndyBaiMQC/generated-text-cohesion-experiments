'''
Code used to evaluate text generation ouput.

November 29th, 2020

Author: Benjamin LeBrun
'''
import nltk
from nltk.util import ngrams
import numpy as np
import re, math
import editdistance
from random import sample
from scipy.stats import hypergeom
from collections import Counter
from itertools import chain
from nltk.corpus.reader import wordnet as wn
from scipy.stats import entropy
from tqdm.notebook import tqdm

# tokenize and pos tag
import spacy
tokenize = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
tokenizer.add_pipe(tokenizer.create_pipe('sentencizer'))

content_words = r'JJ[RS]*$|NN[PS]*$|RB[RS]*$|VB[DGNPZ]*$'

class Document:
	def __init__(self, text):
		self.doc = tokenize(text)
		self.sentences = [[w for w in s] for s in doc.sents]
		self.tokens = list(chain(*[w.text.lower() for w in doc if w.pos_ != 'PUNCT']))
        self.lemmas = [t.lemma_ for t in self.doc if t.pos_ != 'PUNCT']
        self.V = np.unique(self.tokens)
        self.len = len(self.tokens)
        self.freqs = Counter(self.tokens)
        self.freq_spectrum = dict(Counter(self.freqs.values()))

	def get_ngrams(self, n=2, unit='s'):
        if unit == 's':
            return [g for g in ngrams(self.sentences, n)]
        else:
            return [w for w in ngrams(self.tokens, n)]

    ######################### Lexical diversity measures #########################

    def kl_divergence(self, true_dist):
    	''' compare this document's word freq dist to a true dist 
    	using KL-divegence '''
    	if isinstance(true_dist, dict):
    		return entropy(self.freq_spectrum, true_dist)
    	else:
    		return entropy(self.freq_spectrum, dict(Counter(Counter(true_dist).values())))

    def get_entropy(self, seq):
        ''' entropy of a sequence of symbols '''
        freqs = Counter(seq)
        total = sum(freqs.values())
        return sum([-(lambda p: p*math.log(p,2) if p != 1 else 0) (freqs[k]/total) for k in freqs.keys()])

    def evenness(self):
        ''' normalized entropy '''
        return self.get_entropy(self.tokens)/math.log(self.len, 2)

    def get_ttr(self, tokens):
        ''' type-token ratio '''
        return len(np.unique(tokens))/len(tokens)

    def mattr(self, window):
        ''' moving average ttr '''
        def run(tokens):
            return np.nanmean([self.get_ttr(tokens[i:i+window]) 
                                for i in range(0, self.len-window)])
        return np.mean([run(self.tokens), run(list(reversed(self.tokens)))])

    def mtld(self, thrsh=.72):
        ''' mean len of seq of tokens that maintains a ttr '''
        def run(tokens, threshold):    
            ttr, factors, factor_tokens = 1.0, 0, []
            for token in tokens:
                factor_tokens.append(token)
                ttr = self.get_ttr(factor_tokens)
                if ttr >= threshold:
                    continue
                else:
                    factors += 1
                    ttr = 1.0
                    factor_tokens = []
                # if we reach the end and there are still tokens remaining
                if len(factor_tokens) > 0:
                    factors += (1-self.get_ttr(factor_tokens))/(1-threshold)
            return len(tokens)/factors
        return (run(self.tokens, thrsh) + run(list(reversed(self.tokens)), thrsh))/2
        
    def mtld_dist(self, thrsh=.72):
        ''' dist of seq of tokens which maintain a given ttr '''
        def run(tokens, threshold):    
            ttr, factors, factor_tokens, lens = 1.0, 0, [], []
            for token in tokens:
                factor_tokens.append(token)
                ttr = self.get_ttr(factor_tokens)
                if ttr >= threshold:
                    continue
                else:
                    factors += 1
                    ttr = 1.0
                    lens.append(len(factor_tokens))
                    factor_tokens = []
            return lens
        return run(self.tokens, thrsh)

    def hdd(self, sample=42, dist=False):
        ''' HD-D:  a measure of lexical diversity '''
        def get_p_t(t):    
            M, n, N = len(self.tokens), len(self.tokens[self.tokens == t]), sample
            rv = hypergeom(M, n, N)
            return 1-rv.pmf(0)
        if isinstance(self.hdd_vals, bool):
            self.hdd_vals = [get_p_t(w) for w in tqdm(self.V)]
        if dist:
            return self.hdd_vals
        else:
            return np.sum(self.hdd_vals)
        
    def simpsons(self):
        ''' Simpson's D index '''
        return sum([n*(n-1)/(self.len*(self.len-1)) for n in self.freqs.values()])

    ######################### Sentence bigram based measures #########################

    def symed(self, unit):
        ''' levenshtein distance between adjacent sentences 
        unit can be lemmas, words, or POS tags '''
        def distance(bigram, unit):
            if unit == 'word':
                return editdistance.eval([t.text for t in bigram[0]], [t.text for t in bigram[1]])
            elif unit == 'lemma':
                return editdistance.eval([t.lemma_ for t in bigram[0]], [t.lemma_ for t in bigram[1]])
            elif unit == 'pos':
                return editdistance.eval([w.pos_ for w in bigram[0]], [w.pos_ for w in bigram[1]])
        # calculate distance for each pair of adjacent sentences
        return [distance(bigram, unit=unit) for bigram in self.get_ngrams(n=2, unit='s')]

    def overlap(self, variant):
        ''' This function calculates referential cohesion through lexical overlap.
        variant is one of noun, argument, stem, content '''

        variants = {
        	'noun':(r'NN[PS]*', r'NN[PS]*', True),
        	'argument': (r'NN[PS]*|PRP(\$)*', r'NN[PS]*', True),
        	'content': (r'NN[PS]*|RB[RS]*|VB[DGNPZ]*', r'NN[PS]*|RB[RS]*|VB[DGNPZ]*', False),
        	'stem': (r'NN[PS]*|RB[RS]*|VB[DGNPZ]*', r'NN[PS]*', True)
        }

        def overlap(b1, b2, variant):
	        tags1, tags2, binary = variants[variant]
	        if variant != 'stem':
	        	s1 = [w.text.lower() for w in b1 if re.match(tags1, w.tag_)]
	        	s2 = [w.text.lower() for w in b2 if re.match(tags2, w.tag_)]
	        else:
	        	s1 = [w.lemma_.lower() for w in b1 if re.match(tags1, w.tag_)]
	        	s2 = [w.text.lower() for w in b2 if re.match(tags2, w.tag_)]
	        	
	        if binary:
	            return len(list(set(s1).intersection(s2))) > 0
	        else:
	        	return len(list(set(s1).intersection(s2)))/len(np.unique(sum([s1, s2], [])))
        
        return [overlap(bi[0], bi[1], variant) for bi in self.get_ngrams(n=2, unit='s')]

    ######################### word information measures #########################   
    
    def polysemy(self):
        ''' returns distribution of the number of synsets for each token '''
        return [len(wn.synsets(token)) for token in self.tokens if len(wn.synsets(token)) != 0]
        
    def hypernymy(self):
        ''' Feng et al. (2011). Use WordNet to calculate the hypernym distance of a word.
        This is defined as the average length of the path from each sense to the root hypernym 'entity'.
        This function should only be used for nouns. '''
        def get_hypernymy(text):
            entity = wn.synset('entity.n.01')
            distances = [np.nanmean([t[1] for t in [x for x in synset.hypernym_distances() if x[0] == wn.synset('entity.n.01')]]) 
                            for synset in wn.synsets(self.text) if synset.pos() == 'n']
            return np.nanmean(distances)
        return [get_hypernymy(w['lemma']) for w in self.words if re.match(w['xpos'], r'NN[PS]*')]



