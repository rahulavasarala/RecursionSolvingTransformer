import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import random as rand
from serialization import random_serialize

#Input: is the paths to the recursion sentences and solution sentences
#Assume that the padding, start and end tokens are all the same ''
class RecursionDatasetCreator():
    def __init__(self, path_to_recursions, path_to_solutions, filter_percentile, limit, recursion_vocab, solution_vocab):

        self.recursions, self.solutions = self.extract_sentences(limit, path_to_recursions, path_to_solutions)
        self.recursion_vocab = recursion_vocab
        self.recursion_to_index = {vocab : index for index, vocab in enumerate(recursion_vocab)}
        self.solution_vocab = solution_vocab
        self.solution_to_index = {vocab : index for index, vocab in enumerate(solution_vocab)}
        self.filter_percentile = filter_percentile
        self.filter_by_length()
        self.filter_by_validity()

    def extract_sentences(self, limit, path_to_recursions, path_to_solutions):
        recursions, solutions = [], []
        with open(path_to_recursions, 'r') as file:
            recursions = file.readlines()

        with open(path_to_solutions, 'r') as file:
            solutions = file.readlines()

        recursions = [sentence.rstrip('\n') for sentence in recursions]
        solutions = [sentence.rstrip('\n') for sentence in solutions]

        return recursions[:limit], solutions[:limit]

    def filter_by_length(self):
        max_length_recursion = np.percentile([len(sentence) for sentence in self.recursions], self.filter_percentile)
        max_length_solution = np.percentile([len(sentence) for sentence in self.solutions], self.filter_percentile)

        max_length = min(max_length_recursion, max_length_solution)
        print(max_length)

        valid_indices = []

        for i in range(len(self.recursions)):
            if len(self.recursions[i]) <= max_length and len(self.solutions[i]) <= max_length:
                valid_indices.append(i)

        self.recursions = [self.recursions[i] for i in valid_indices]
        self.solutions = [self.solutions[i] for i in valid_indices]

    def is_sentence_valid(self, sentence, vocab_to_index):
        for char in sentence:
            if char not in vocab_to_index:
                return False

        return True
        
    def filter_by_validity(self):
        valid_indices = []
        for i in range(len(self.recursions)):
            if (self.is_sentence_valid(self.recursions[i], self.recursion_to_index)) and \
                (self.is_sentence_valid(self.solutions[i],self.solution_to_index)):

                valid_indices.append(i)

        self.recursions = [self.recursions[i] for i in valid_indices]
        self.solutions = [self.solutions[i] for i in valid_indices]

    def create_recursion_dataset(self):
        return RecursionDataset(self.recursions, self.solutions)

    def find_max_sequence_length(self):
        max_recursion = max(len(sentence) for sentence in self.recursions)
        max_solution = max(len(sentence) for sentence in self.soutions)

        return max(max_recursion, max_solution) + 2


class RecursionDataset(Dataset):

    def __init__(self, recursion_data, solution_data):
        self.recursion_data = recursion_data
        self.solution_data = solution_data

    def __len__(self):
        return len(self.recursion_data)

    def __getitem__(self, idx):
        return self.recursion_data[idx], self.solution_data[idx]

def init_recursion_solution_files():

    #pick a combination of a, b, and c from 0-9
    # a will have to be non 0 for the solution to be generated

    #generate the solution by writing it in an acceptable form
    # for each pair a - b, randomly serialize the data each time,
    #and create copies of the  input data
    #repeat this until the dataset is 100 in size
    #export the data and put it in the training cycle.

    recursion_data = []
    solution_data = []
    
    combo_map = {}

    while len(recursion_data) < 100: 

        a = rand.randint(1,9)
        if a == 1:
            a = 0
        b = rand.randint(0,9)

        c = rand.randint(0,9)
    
        combo = 100*a + 10 * b + c
    
        if combo in combo_map:
            continue
        else:
            combo_map[combo] = True

        recursion = "T_(n+1) = {} * T_n + {} * n + {}".format(a,b,c)

        solution = "T_n = ({b}/(1-{a})) * n + ({c}/(1-{a})) - {b}/(1-{a})^2 + {d} * a ^ n".format(a = a, b = b, c = c, d = "d")

        recursion_map = {}

        batch_specific_recursions = []
        batch_specific_solutions = []

        while len(batch_specific_recursions) < 3:

            rec_ser = random_serialize(recursion, random = True)
            sol_ser = random_serialize(solution, random = True)
            if rec_ser in recursion_map:
                continue
            else:
                recursion_map[rec_ser] = True
                batch_specific_recursions.append(rec_ser)
                batch_specific_solutions.append(sol_ser)

        recursion_data = recursion_data + batch_specific_recursions
        solution_data = solution_data + batch_specific_solutions

                

    with open('recursions.txt', 'w') as f:
        for line in recursion_data:
            f.write("{}\n".format(line))

    with open('solutions.txt', 'w') as f:
        for line in solution_data:
            f.write("{}\n".format(line))

    return recursion_data, solution

        