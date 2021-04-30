import pickle
import nltk
import spacy
from tqdm import tqdm


class LanguageModel():
    def __init__(self, traininig_from_scratch=False, language_corpus='ru2_nerus_800ks_96'):

        self.default_path = 'data/language_model_final' # the path to the language model
        self.nlp = spacy.load(language_corpus)
        self.invalid_pos = ('NUM', 'PUNCT', 'SYM', 'X', 'SPACE', 'NO_TAG') # the words of these POS will not be included in the creation of the staistic model

        if traininig_from_scratch:
            self.unigrams = nltk.FreqDist()
            self.backward_bigrams = nltk.FreqDist()
            self.forward_bigrams = nltk.FreqDist()
            self.trigrams = nltk.FreqDist()
            self.word_registers = {}

            self.unigrams_hop = 0 # the variable indicates the starting point in the training corpus for unigrams
            self.ngrams_hop = 0 # the variable indicates the starting point in the training corpus for ngrams
        else:
            print('Loading model..')
            self._load_object()
            print('Done!')
      

    def _keep_register(self, word, lemma):
        '''
            Application of lemmatization results in the loss of the case. 
            The function allows you to restore the register of a lemma.
        '''
        for i in range(min(len(word), len(lemma))):
            if word[i].isupper():
                lemma = lemma[:i:] + lemma[i].upper() + lemma[i+1::]
        
        return lemma
    

    def _save_object(self, path=None):
        if path == None:
            path = self.default_path
        
        pickle_dict = {
                       'unigrams': self.unigrams,
                       'backward_bigrams': self.backward_bigrams,
                       'forward_bigrams': self.forward_bigrams,
                       'trigrams': self.trigrams,
                       'word_registers': self.word_registers,
                       'unigrams_hop': self.unigrams_hop,
                       'ngrams_hop': self.ngrams_hop
                      }

        with open(path, "wb") as fp:
            pickle.dump(pickle_dict, fp)
      

    def _load_object(self, path=None):
        if path == None:
            path = self.default_path
  
        with open(path, "rb") as language_model:
            pickle_dict = pickle.load(language_model)
            self.unigrams = pickle_dict['unigrams']
            self.backward_bigrams = pickle_dict['backward_bigrams']
            self.forward_bigrams = pickle_dict['forward_bigrams']
            self.trigrams = pickle_dict['trigrams']
            self.word_registers = pickle_dict['word_registers']

            self.unigrams_hop = pickle_dict['unigrams_hop']
            self.ngrams_hop = pickle_dict['ngrams_hop']


    def preprocessing(self, company_name):
        '''
            The preprocessing function tokenizes the company name and 
            applies lemmatization for each token. 
        '''
        doc = self.nlp(company_name)

        tokenized_company_name = [ token.text for token in doc ] # the list contains non-modified tokens
        lemmatized_company_name = [] # the list contains lemmatized tokens
        valid_token_indices = [] # the list contains indexes of tokens that will participate in the generation of ngrams

        for i, token in enumerate(doc):
            if token.pos_ not in self.invalid_pos:
                lemma = self._keep_register(token.text, token.lemma_)
                lemmatized_company_name.append(lemma)
                valid_token_indices.append(i)
            else:
                lemmatized_company_name.append(token.text)

        return {
                'tokenized_company_name': tokenized_company_name, 
                'lemmatized_company_name': lemmatized_company_name,
                'valid_token_indices': valid_token_indices
               }
      

    def generate_unigram(self, processed_company_name):
        '''
            The function counts the occurrence of each token in the training corpus 
            and also generates sets for each word containing the different cases for word.
        '''
        for index in processed_company_name['valid_token_indices']:
            lemma = processed_company_name['lemmatized_company_name'][index]

            self.unigrams[lemma] += 1

            if lemma.lower() not in self.word_registers:
                self.word_registers[lemma.lower()] = set()
            self.word_registers[lemma.lower()].add(lemma)


    def generate_ngrams(self, processed_company_name):
        '''
            The function counts the occurrence of forward bigrams, backward bigrams and trigrams.
        '''
        for index in processed_company_name['valid_token_indices']:

            current_token = processed_company_name['lemmatized_company_name'][index]

            if current_token.lower() not in self.word_registers or len(self.word_registers[current_token.lower()]) < 2:
                continue

            # generating backward_bigrams
            if index != 0:
                prev_token = processed_company_name['lemmatized_company_name'][index-1]
                backward_bigram_token = prev_token + '_' + current_token
                self.backward_bigrams[backward_bigram_token] += 1

            # generating forward_bigrams
            if index < len(processed_company_name['tokenized_company_name']) - 1:
                next_token = processed_company_name['tokenized_company_name'][index+1].lower()
                forward_bigram_token = current_token + '_' + next_token
                self.forward_bigrams[forward_bigram_token] += 1
            
            # generating trigrams
            if index != 0 and index < len(processed_company_name['tokenized_company_name']) - 1:
                trigram_token = prev_token + '_' + current_token + '_' + next_token
                self.trigrams[trigram_token] += 1


    def train(self, train_corpus_path, unigrams=True):

        # generating unigrams and sets of possible cases for each lemmatized token
        if unigrams:
            with open(train_corpus_path) as company_names:

                print(f'Starting from {self.unigrams_hop}')
                for i in tqdm(range(self.unigrams_hop)):
                    next(company_names)
                  
                print('Training..')
                for i, company_name in enumerate(tqdm(company_names)):
                    self.unigrams_hop += 1

                    if self.unigrams_hop % 10000 == 0:
                        self._save_object()

                    processed_company_name = self.preprocessing(company_name)         
                    self.generate_unigram(processed_company_name)
                
                self._save_object()
        
        # generating bigrams and trigrams
        else:
            with open(train_corpus_path) as company_names:

                print(f'Starting from {self.ngrams_hop}')
                for i in tqdm(range(self.ngrams_hop)):
                    next(company_names)

                print('Training..')
                for i, company_name in enumerate(tqdm(company_names)):
                    self.ngrams_hop += 1

                    if self.ngrams_hop % 100000 == 0:
                        self._save_object()
                    
                    processed_company_name = self.preprocessing(company_name)         
                    self.generate_ngrams(processed_company_name)
                
                self._save_object()


# language_model = LanguageModel()
# print(language_model.unigrams_hop)
# print(language_model.ngrams_hop)
# language_model.train('data/train.txt', False)