import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader

#This is the get_device method which checks whether a cuda powered gpu could be used
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#define the scaled dot product function
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim = -1)
    values = torch.matmul(attention, v)
    return values, attention

#define the multiheaded attention module here- This will apply to the encoder and decoder
class MultiheadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, identity = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model//num_heads
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.identity = identity
        self.extra_linear_layer = nn.Linear(d_model, d_model)
        if identity:
            self.make_linear_layers_identity()

    def make_linear_layers_identity(self): # This is for unit testing this module
        i = torch.eye(self.d_model)
        qkv_weights = torch.cat((i,i,i), dim = 1)
        self.qkv_layer.weight.data.copy_(qkv_weights.T)
        qkv_biases = torch.zeros(3* self.d_model)
        self.qkv_layer.bias.data.copy_(qkv_biases)
        linear_weights = i
        linear_biases = torch.zeros(self.d_model)
        self.extra_linear_layer.weight.data.copy_(linear_weights)
        self.extra_linear_layer.bias.data.copy_(linear_biases)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim = -1)
        values, attention = scaled_dot_product_attention(q,k,v, mask)
        values = values.reshape(batch_size, sequence_length, self.d_model) ##easier to understand
        out = self.extra_linear_layer(values)
        return out

#Define the Mutiheaded Cross Attention here- this will apply to the decoder, when taking decoder q and taking encoder k and v
class MultiheadedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, identity = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model//num_heads
        self.kv_layer = nn.Linear(d_model, 2*d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.extra_linear_layer = nn.Linear(d_model, d_model)
        if identity:
            self.make_linear_layers_identity()

    def make_linear_layers_identity(self): #This is for unit testing this module
        i = torch.eye(self.d_model)
        kv_weights = torch.cat((i,i), dim = 1)
        self.kv_layer.weight.data.copy_(kv_weights.T)
        kv_biases = torch.zeros(2* self.d_model)
        self.kv_layer.bias.data.copy_(kv_biases)
        linear_weights = i
        linear_biases = torch.zeros(self.d_model)
        self.extra_linear_layer.weight.data.copy_(linear_weights)
        self.extra_linear_layer.bias.data.copy_(linear_biases)
        

    def forward(self, x, y, mask=None):#x is from encoder, y is from decoder
        batch_size, sequence_length, input_dim = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2*self.head_dim)
        kv = kv.permute(0,2,1,3)
        k, v = kv.chunk(2 , dim = -1)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        q = q.permute(0,2,1,3)
        values, attention = scaled_dot_product_attention(q,k,v, mask)
        values = values.reshape(batch_size, sequence_length, self.d_model) ##easier to understand
        out = self.extra_linear_layer(values)
        return out

#Define the SentenceEmbedding Class, that can take a batch of input and output sentences and embed them
#This class takes in a vocabulary to id dictionary so that the sentence embedder can start encoding the 
#sentences as strings of numbers, and then these strings of numbers are converted into actual embeddings
# with maximum dim d_model and maximum sentence length max_sequence_length
class Embedder(nn.Module):
    def __init__(self,len_vocab, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(len_vocab + 1, d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.pos_encoder = PositionalEncoder(max_seq_length, d_model)

    def forward(self, x, pos = True): # sentence
        x = self.embeddings(x)
        if pos: 
            pos = self.pos_encoder.generate()
            x = self.dropout(x + pos)
        return x.to(torch.float32)

class Tokenizer():
    def __init__(self, vocab, START, PAD, END):
        self.vocab = vocab
        self.vocab_to_index = {vocab : index for index, vocab in enumerate(vocab)}
        self.index_to_vocab = {index : vocab for index, vocab in enumerate(vocab)}
        self.START = START
        self.PAD = PAD
        self.END = END

    def tokenize(self, sentences):

        tokenized_sentences = []
        for i in range(len(sentences)):
            tokenized_chain = []
            for char in sentences[i]:
                tokenized_chain.append(self.vocab_to_index[char])

            tokenized_sentences.append(tokenized_chain)

        return tokenized_sentences

    def untokenize(self, tok_output):

        sentence_output = []

        for i in range(len(tok_output)):

            sentence = ""
            for num in tok_output[i]:
                sentence = sentence + self.index_to_vocab[num]

            sentence_output.append(sentence)

        return sentence_output

    def pad(self, tokenized_sentences, max_sequence_length, start = True, end = True):
        #This method here should turn the input into what can go into the transformer

        for i in range(len(tokenized_sentences)):
            if start:
                tokenized_sentences[i].insert(0, self.vocab_to_index[self.START])
            if end:
                tokenized_sentences[i].append(self.vocab_to_index[self.END])
            while len(tokenized_sentences[i]) < max_sequence_length:
                tokenized_sentences[i].append(self.vocab_to_index[self.PAD])

        tokenized_sentences = torch.tensor(tokenized_sentences)
        print(tokenized_sentences.size())

        return tokenized_sentences
            

NEG_INFTY = -1e9

def generate_masks_tokenized(tok_input, tok_output, input_pad, output_pad):

    batch_size, max_sequence_length = tok_input.size()
    
    enc_mask = torch.zeros([batch_size, max_sequence_length, max_sequence_length])
    dec_mask = torch.zeros([batch_size, max_sequence_length, max_sequence_length])
    cross_mask = torch.zeros([batch_size, max_sequence_length, max_sequence_length])

    for i in range(batch_size):
        #Find the padding token index in for the input sentence
        in_padding = torch.where(tok_input[i] == input_pad)
        in_pad_start = max_sequence_length
        if len(in_padding[0]) > 0:
            in_pad_start = in_padding[0][0].item()
        enc_mask[i, in_pad_start:max_sequence_length, :] = 1
        enc_mask[i, :, in_pad_start:max_sequence_length] = 1
        #Create a upper triangular

        ones = torch.ones([max_sequence_length, max_sequence_length])
        upper = torch.triu(ones)
        upper = upper - torch.eye(max_sequence_length)
        dec_mask[i] = upper
        
        #Find the padding token index for the output sentence
        out_padding = torch.where(tok_output[i] == output_pad)
        out_pad_start = max_sequence_length
        if len(out_padding[0]) > 0:
            out_pad_start  = out_padding[0][0].item()
        dec_mask[i, out_pad_start:max_sequence_length, :] = 1
        dec_mask[i, :, out_pad_start:max_sequence_length] = 1

        #Figure out the cross mask
        cross_mask[i, :, in_pad_start:max_sequence_length] = 1
        cross_mask[i, out_pad_start:max_sequence_length, :] = 1

    enc_mask = enc_mask * NEG_INFTY
    dec_mask = dec_mask * NEG_INFTY
    cross_mask = cross_mask * NEG_INFTY

    return enc_mask, dec_mask, cross_mask

class PositionalEncoder():
    def  __init__(self, max_sequence_length, d_model):
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    
    def generate(self):
        #Generate the positional encodings
        consecutive = np.arange(0,self.d_model)
        odd_indices = consecutive[consecutive % 2 == 1]
        even_indices = consecutive[consecutive % 2 == 0]
        pos_encoding = np.zeros((self.max_sequence_length, self.d_model))
        
        for pos in range(self.max_sequence_length):
            sin_values = [np.sin(pos/(10000**(i/self.d_model))) for i in even_indices]
            cos_values =  [np.cos(pos/(10000**((i-1)/self.d_model))) for i in odd_indices]
            pos_encoding[pos][odd_indices] = cos_values
            pos_encoding[pos][even_indices] = sin_values

        pos_encoding = pos_encoding[np.newaxis, :, :]
        pos_encoding = torch.tensor(pos_encoding)

        return pos_encoding


class LayerNorm(nn.Module):
    def __init__(self,d_model, eps=1e-5):
        ##NIGGAS AND SHIT
        super().__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim = True)
        var = ((inputs - mean)**2).mean(dim=-1, keepdim = True)
        std = (var + self.eps).sqrt()
        scaled = (inputs - mean)/std
        out = self.gamma * scaled + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_hidden = ffn_hidden
        self.drop_prob = drop_prob

        #These are the layers of the encoder
        self.attention = MultiheadedAttention(num_heads = num_heads, d_model = d_model)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)

    def forward(self, x, encoder_self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask = encoder_self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

#
class EncoderChain(nn.Sequential):
    def forward(self, *inputs):
        x, encoder_self_attention_mask = inputs

        for module in self._modules.values():
            x = module(x, encoder_self_attention_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob):
        super().__init__()
        self.masked_attention = MultiheadedAttention(num_heads = num_heads, d_model = d_model)
        self.cross_attention = MultiheadedCrossAttention(num_heads = num_heads, d_model = d_model)
        self.norm1 = LayerNorm(d_model = d_model)
        self.norm2 = LayerNorm(d_model = d_model)
        self.norm3 = LayerNorm(d_model = d_model)
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.dropout3 = nn.Dropout(p = drop_prob)
        
    def forward(self, x, encoder_input, masked_mask, cross_mask):
        residual_x =  x.clone()
        x = self.masked_attention(x, mask = masked_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.cross_attention(encoder_input, x, mask = cross_mask)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + residual_x)
        return x
        
class DecoderChain(nn.Sequential):
    def forward(self, *inputs):
        x, encoder_input, masked_mask, cross_mask = inputs

        for module in self._modules.values():
            x = module(x, encoder_input, masked_mask, cross_mask)

        return x

class Transformer(nn.Module):

    def __init__(self, d_model, num_heads, chain_length, max_seq_length, ffn_hidden, drop_prob, input_vocab_size, output_vocab_size):
        
        super().__init__()
        self.input_embedder = Embedder(input_vocab_size + 1, d_model, max_seq_length)
        self.output_embedder = Embedder(output_vocab_size + 1, d_model, max_seq_length)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_seq_length = max_seq_length
        
        self.encoder_chain = EncoderChain(
            *[Encoder(
                d_model=d_model,
                num_heads=num_heads,
                ffn_hidden=ffn_hidden,
                drop_prob=drop_prob
            ) for _ in range(chain_length)]
        )
        
        self.decoder_chain = DecoderChain(
            *[Decoder(
                d_model=d_model,
                num_heads=num_heads,
                ffn_hidden=ffn_hidden,
                drop_prob=drop_prob
            ) for _ in range(chain_length)]
        )

        self.device = get_device()
        self.linear = nn.Linear(d_model, output_vocab_size)
        self.reset_params()

    def reset_params(self): 
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x, y, enc_mask, dec_mask, cross_mask): 
        x = self.input_embedder(x)
        y = self.output_embedder(y)
        x = self.encoder_chain(x, enc_mask)
        out = self.decoder_chain(y, x, dec_mask, cross_mask)
        out = self.linear(out)
        out = F.softmax(out, dim = -1)
        return out