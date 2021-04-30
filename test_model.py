from train_model import LanguageModel
from tqdm import tqdm
import numpy as np
from copy import copy
from f1_score import F1


class Truecasing():
    def __init__(self, language_model, validation, validation_split=None):
        self.language_model = language_model
        self.validation = False
        if validation:
            self.validation = True
            self.validation_split = validation_split
      

    def get_score(self, prev_token, target_token, next_token):
        pseudo_count = 5.0

        # calculating unigram score
        nominator = self.language_model.unigrams[target_token] + pseudo_count
        denominator = 0
        for alternative_token in self.language_model.word_registers[target_token.lower()]:
            denominator += self.language_model.unigrams[alternative_token] + pseudo_count

        unigram_score = nominator / denominator

        # calculating backward score
        bigram_backward_score = 1
        if prev_token is not None:
            nominator = self.language_model.backward_bigrams[prev_token + '_' + target_token] + pseudo_count
            denominator = 0
            for alternative_token in self.language_model.word_registers[target_token.lower()]:
                denominator += self.language_model.backward_bigrams[prev_token + '_' + alternative_token] + pseudo_count

            bigram_backward_score = nominator / denominator

        # calculating forward score
        bigram_forward_score = 1
        if next_token is not None:
            next_token = next_token.lower()  # ensure it is lower case
            nominator = self.language_model.forward_bigrams[target_token + '_' + next_token] + pseudo_count
            denominator = 0
            for alternative_token in self.language_model.word_registers[target_token.lower()]:
                denominator += self.language_model.forward_bigrams[alternative_token + '_' + next_token] + pseudo_count

            bigram_forward_score = nominator / denominator

        # calculating trigram score
        trigram_score = 1
        if prev_token is not None and next_token is not None:
            next_token = next_token.lower()  # ensure it is lower case
            nominator = self.language_model.trigrams[prev_token + "_" + target_token + "_" + next_token] + pseudo_count
            denominator = 0
            for alternative_token in self.language_model.word_registers[target_token.lower()]:
                denominator += self.language_model.trigrams[prev_token + '_' + alternative_token + '_' + next_token] + pseudo_count

            trigram_score = nominator / denominator
        
        unigram_coef = 1
        bigram_backward_coef = 2
        bigram_forward_score = 2
        trigram_coef = 3

        total_score = unigram_coef * unigram_score + bigram_backward_coef * bigram_backward_score + bigram_forward_score * bigram_forward_score + trigram_coef * trigram_score
        # total_score = np.log(unigram_score) + np.log(bigram_backward_score) + np.log(bigram_forward_score) + np.log(trigram_score)
        return total_score

      
    def get_true_case(self, processed_company_name):

        new_tokens = []
        for i in processed_company_name['valid_token_indices']:
            token = processed_company_name['lemmatized_company_name'][i]
            token = token.lower()

            if token in self.language_model.word_registers:
                if len(self.language_model.word_registers[token]) == 1:
                    best_token = list(self.language_model.word_registers[token])[0]
                else:
                    prev_token = (processed_company_name['lemmatized_company_name'][i - 1] if i > 0 else None)
                    next_token = (processed_company_name['lemmatized_company_name'][i + 1] if i < len(processed_company_name['lemmatized_company_name']) - 1 else None)

                    best_token = None
                    highest_score = -np.inf

                    for case in self.language_model.word_registers[token]:
                        score = self.get_score(prev_token, case, next_token)

                        if score > highest_score:
                            best_token = case
                            highest_score = score

            else:
                initial_token = processed_company_name['tokenized_company_name'][i]
                
                print('New: ', processed_company_name['tokenized_company_name'][i])
                if initial_token in self.language_model.word_registers:

                    if len(self.language_model.word_registers[initial_token]) == 1:
                        best_token = list(self.language_model.word_registers[initial_token])[0]
                    else:
                        prev_token = (processed_company_name['lemmatized_company_name'][i - 1] if i > 0 else None)
                        next_token = (processed_company_name['lemmatized_company_name'][i + 1] if i < len(processed_company_name['lemmatized_company_name']) - 1 else None)

                        best_token = None
                        highest_score = -np.inf

                        for case in self.language_model.word_registers[initial_token]:
                            score = self.get_score(prev_token, case, next_token)

                            if score > highest_score:
                                best_token = case
                                highest_score = score
                else:
                    new_tokens.append(i)
                    best_token = initial_token

            processed_company_name['tokenized_company_name'][i] = self.language_model._keep_register(best_token, processed_company_name['tokenized_company_name'][i].lower())

        return processed_company_name['tokenized_company_name'], new_tokens
      

    def _restore_punct(self, initial_company_name, obtained_tokens):
        pointer = 0
        modified_company_name = copy(initial_company_name)
        for token in obtained_tokens:
            # print(f'modified_company_name: {modified_company_name}')
            start = modified_company_name.find(token.lower(), pointer)
            # print(token, start)
            end = start + len(token)
            modified_company_name = modified_company_name[:start:] + token + modified_company_name[end::]
            pointer = end
        
        # print(f'modified_company_name: {modified_company_name}')
        return modified_company_name

    
    def out_of_vocabulary(self, word, left_pointer=0, result_tokens=''):
        new_token = ''
        flag=False
        right_pointer = len(word)

        current_right = right_pointer
        current_unigram_score = -1

        if left_pointer == right_pointer:
            return result_tokens

        while right_pointer - left_pointer >= 3:
            if word[left_pointer:right_pointer:].lower() in self.language_model.word_registers:
                # if self.language_model.unigrams[word[left_pointer:right_pointer:].capitalize()] > current_unigram_score:
                #     current_unigram_score = self.language_model.unigrams[word[left_pointer:right_pointer:].capitalize()]
                #     current_right = right_pointer
                #     new_token = word[left_pointer:right_pointer:].capitalize()
                new_token = word[left_pointer:right_pointer:].capitalize()
                current_right = right_pointer
                flag = True
                break

            right_pointer -= 1
        
        if not flag:
            return word.capitalize()
        else:
            result_tokens += new_token
            return self.out_of_vocabulary(word, current_right, result_tokens)


    def postprocessing(self, tokenized_company_name, out_of_vocabulary):
        prev_token = ''
        for i, token in enumerate(tokenized_company_name):
            if i == 0:
                tokenized_company_name[i] = token.capitalize()
            if prev_token == '"' or prev_token == "'" or prev_token == '.':
                tokenized_company_name[i] = token.capitalize()

            if len(token) == 2 or len(token) == 3:
                tokenized_company_name[i] = token.upper()
            elif i in out_of_vocabulary:
                tokenized_company_name[i] = self.out_of_vocabulary(token)
            
            prev_token = token
        
        return tokenized_company_name


    def predict(self, corpus_path):
        if self.validation:
            # company_name = 'общество с ограниченной ответственностью "искусство копчения"'
            # true_name = 'Общество с ограниченной ответственностью "ИСКУССТВО КОПЧЕНИЯ"'
            # processed_company_name = self.language_model.preprocessing(company_name)
            # print(processed_company_name)
            # true_cased_name = self.get_true_case(processed_company_name)
            # result = self._restore_punct(company_name, true_cased_name)
            # # print(result)

            # # print(len(result), len(true_name))
            # score = Test(true_name, result)
            # print(f'SCORE: {score.f1_score()}')

            total_score = 0
            entries = 0

            with open(corpus_path) as company_names:

                print(f'Starting from {self.validation_split}')
                for i in tqdm(range(self.validation_split)):
                    next(company_names)
                  
                print('Validation..')
                for i, company_name in enumerate(tqdm(company_names)):
                    
                    company_name_lower = company_name.lower()
                    processed_company_name = self.language_model.preprocessing(company_name_lower)
                    true_cased_name, new_tokens = self.get_true_case(processed_company_name)
                    postprocessed = self.postprocessing(true_cased_name, new_tokens)
                    result = self._restore_punct(company_name_lower, postprocessed)

                    # print(f'Valid: {company_name}, {len(company_name)}; Result: {result}, {len(result)}')
                    print('Predicted: ', result)
                    print('Actual: ', company_name)
                    print('######')
                    f1 = F1(company_name, result)
                    score = f1.f1_score()
                    total_score += score
                    entries += 1

                    if entries % 1000 == 0:
                        mean_score = total_score / entries
                        print(f'Avg. F1-score: {mean_score}')
        else:
            test = open('result.txt', 'a')
            with open(corpus_path) as company_names:
                for i, company_name in enumerate(tqdm(company_names)):
                    company_name_lower = company_name.lower()
                    processed_company_name = self.language_model.preprocessing(company_name_lower)
                    true_cased_name, new_tokens = self.get_true_case(processed_company_name)
                    postprocessed = self.postprocessing(true_cased_name, new_tokens)
                    result = self._restore_punct(company_name_lower, postprocessed)
                    test.write(result)
            
            test.close()



language_model = LanguageModel()
truecasing = Truecasing(language_model, False)
# print(truecasing.out_of_vocabulary('тд'))
truecasing.predict('data/test.txt')