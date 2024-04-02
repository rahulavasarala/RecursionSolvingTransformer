import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformer import generate_masks_tokenized

def run_beam_search(input_sentences, transformer, input_tokenizer, output_tokenizer, num_beams = 5):

    transformer.eval()

    max_seq_length = transformer.max_seq_length
    recursion_padding_index = input_tokenizer.vocab_to_index[input_tokenizer.PAD]
    solution_padding_index = output_tokenizer.vocab_to_index[output_tokenizer.PAD]
    solution_eos_index = output_tokenizer.vocab_to_index[output_tokenizer.END]
    solution_start_index = output_tokenizer.vocab_to_index[output_tokenizer.START]

    base_input = input_tokenizer.tokenize(input_sentences)
    base_input = input_tokenizer.pad(base_input, max_seq_length, start = False, end = False)
    scaled_input = base_input.clone()

    tok_output = [[solution_start_index]]
    tok_output = output_tokenizer.pad(tok_output, max_seq_length, start = False, end = False)

    next_batch_size = num_beams
    curr_batch_size = 1
    curr_log_likelihoods = np.array([0 for _ in range(1)])

    final_beam_predictions = []

    for i in range(max_seq_length):
        
        if scaled_input.size()[0] != curr_batch_size:
            scaled_input = torch.ones((curr_batch_size, max_seq_length), dtype = int)
            for i in range(curr_batch_size):
                scaled_input[i] = base_input

        enc_mask, dec_mask, cross_mask = generate_masks_tokenized(scaled_input, tok_output, recursion_padding_index, 
                                                                  solution_padding_index)
        predictions = transformer(scaled_input, tok_output, enc_mask, dec_mask, cross_mask)

        group_likelihoods = []
        group_tokens = []


        for sentence_num in range(curr_batch_size):

            logits = predictions[sentence_num][i].clone()
            #take the negative log of all the probabilities and sort
            np_logits = np.array(logits.detach().numpy())
            np_logits = -1*np.log(np_logits)
            
            most_probable_indices = np.argsort(np_logits)
            most_probable_indices = most_probable_indices[0:next_batch_size]
            token_likelihoods = np_logits[most_probable_indices]
            sequence_likelihoods = curr_log_likelihoods[sentence_num] + token_likelihoods

            group_likelihoods = group_likelihoods + sequence_likelihoods.tolist()
            group_tokens = group_tokens + most_probable_indices.tolist()

        most_probable_indices = np.argsort(group_likelihoods)
        most_probable_indices = most_probable_indices[0:next_batch_size]
        group_likelihoods = np.array(group_likelihoods)
        group_tokens = np.array(group_tokens)
        group_likelihoods = group_likelihoods[most_probable_indices]#This is length 5
        group_tokens = group_tokens[most_probable_indices]#this is length 5

        new_tok_output = torch.empty((0,max_seq_length), dtype = int)
        new_log_likelihoods = []

        for idx in range(len(most_probable_indices)):
            sentence_number = most_probable_indices[idx]//next_batch_size
            token = group_tokens[idx]
            likelihood = group_likelihoods[idx]
            
            if token == solution_eos_index:
                s = tok_output[sentence_number].tolist()
                s.append(solution_eos_index)
                final_beam_predictions.append(s)
                continue

            row_to_add = tok_output[sentence_number].clone()
            if i == max_seq_length -1:
                s = torch.zeros((max_seq_length + 1))
                s[0:max_seq_length] = row_to_add
                s[max_seq_length] = token
                s = s.tolist()
                final_beam_predictions.append(s)
            else:
                row_to_add[i+1] = token
                row_to_add = torch.unsqueeze(row_to_add, dim=0)
                new_tok_output = torch.cat((new_tok_output, row_to_add), dim = 0)
                new_log_likelihoods.append(likelihood)

        tok_output = new_tok_output
        curr_batch_size = tok_output.size()[0]
        next_batch_size = curr_batch_size
        curr_log_likelihoods = np.array(new_log_likelihoods)

        if curr_batch_size == 0:
            break

    out_sentences = output_tokenizer.untokenize(final_beam_predictions)
        
    return out_sentences

def run_inference(input_sentences, transformer, input_tokenizer, output_tokenizer):

    transformer.eval()

    batch_size = len(input_sentences)
    max_seq_length = transformer.max_seq_length
    recursion_padding_index = input_tokenizer.vocab_to_index[input_tokenizer.PAD]
    solution_padding_index = output_tokenizer.vocab_to_index[output_tokenizer.PAD]

    tok_input = input_tokenizer.tokenize(input_sentences)
    tok_input = input_tokenizer.pad(tok_input, max_seq_length, start = False, end = False)

    output_sentences = [[] for _ in range(batch_size)]
    tok_output = output_tokenizer.pad([[0] for _ in range(batch_size)], 
                                      max_seq_length, start = False, end = False)

    #Assume that the output token chain will grow

    completed_outputs = [False for _ in range(batch_size)]

    for i in range(max_seq_length):
        enc_mask, dec_mask, cross_mask = generate_masks_tokenized(tok_input, tok_output, recursion_padding_index, 
                                                                  solution_padding_index)
        predictions = transformer(tok_input, tok_output, enc_mask, dec_mask, cross_mask)

        for batch_num in range(batch_size):

            if completed_outputs[batch_num]:
                continue

            logits = predictions[batch_num][i]
            value , max_index = torch.max(logits, dim = -1)
            max_idx = max_index.item()
            if output_tokenizer.index_to_vocab[max_idx] == output_tokenizer.END:
                completed_outputs[batch_num] = True

            output_sentences[batch_num].append(max_idx)
            if i < max_seq_length -1:
                tok_output[batch_num][i+1] = max_index

    #Now you need to untokenize each of the lists
    out_sentences = output_tokenizer.untokenize(output_sentences)

    return out_sentences