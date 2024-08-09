import numpy as np


class HelperFunction:
  @staticmethod
  def relu(x):
    return np.maximum(0, x)

  @staticmethod
  def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

  @staticmethod
  def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = np.random.rand(out_features, in_features)
    self.bias = np.random.rand(out_features)

  def forward(self, x):
    '''
    input: (N, in_features)
    output: (N, out_features)
    '''
    x = np.dot(x, self.weight.T) + self.bias
    x = HelperFunction.relu(x)
    return x


class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) # (out_channels, in_channels, K, K)
    self.bais = np.random.randn(out_channels)

  def forward(self, x):
    '''
    input: (N, C_in, H, W)
    output: (N, C_out, new_h, new_W)
    '''
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    if self.padding > 0:
      x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

    new_H = int((H + 2*self.padding - self.kernel_size) / self.stride + 1)
    new_W = int((W + 2*self.padding - self.kernel_size) / self.stride + 1)

    filters = np.repeat(self.filters[np.newaxis, :, :, :, :], N, axis=0)
    result = np.zeros((N, self.out_channels, new_H, new_W))
    for h in range(0, new_H, self.stride):
      for w in range(0, new_W, self.stride):
        curr = x[:, :, h:h+self.kernel_size, w:w+self.kernel_size]  # (N, C_in, K, K)
        modified_curr = np.repeat(curr[:, np.newaxis, :, :, :], self.out_channels, axis=1) # (N, C_out, C_in, K, K)
        curr_res = modified_curr * filters # (N, C_out, C_in, K, K)
        curr_conv = np.max(curr_res, axis=(2, 3, 4), keepdims=False) + self.bais # (N, C_out)
        result[:, :, h:h+1, w:w+1] = np.expand_dims(np.expand_dims(curr_conv, axis=-1), axis=-1) # (N, C_out, 1, 1)
  
    result = HelperFunction.relu(result)
    return result



class ConvTranspose2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) # (out_channels, in_channels, K, K)
    self.bais = np.random.randn(out_channels)

  def forward(self, x):
    '''
    input: (N, C_in, H, W)
    output: (N, C_out, new_h, new_W)
    '''
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    if self.padding > 0:
      x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
    
    new_H = int(self.stride*(H - 1) - 2*self.padding + self.kernel_size)
    new_W = int(self.stride*(W - 1) - 2*self.padding + self.kernel_size)

    filters = np.repeat(self.filters[np.newaxis, :, :, :, :], N, axis=0)
    result = np.zeros((N, self.out_channels, new_H, new_W))
    for h in range(0, H, self.stride):
      for w in range(0, W, self.stride):
        curr = x[:, :, h:h+1, w:w+1]  # (N, C_in, 1, 1)
        curr = np.repeat(np.repeat(curr, self.kernel_size, axis=2), self.kernel_size, axis=3) # (N, C_in, K, K)
        modified_curr = np.repeat(curr[:, np.newaxis, :, :, :], self.out_channels, axis=1) # (N, C_out, C_in, K, K)
        curr_res = modified_curr * filters # (N, C_out, C_in, K, K)
        curr_trans = np.sum(curr_res, axis=2)
        result[:, :, h:h+self.kernel_size, w:w+self.kernel_size] += curr_trans
        return result
    
    
class MaxPool2d:
  def __init__(self, kernel_size, stride, padding):
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    assert self.padding < self.kernel_size / 2, "Padding must be smaller than half of the kernel size"

  def forward(self, x):
    '''
    input:
      x: (N, C, H, W)
    output:
      result: (N, C, new_H, new_W)
    '''
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    # add padding to input x if self.padding is bigger than 0
    # (N, C, H, W) -> (N, C, H+2*self.padding, W+2*self.padding)
    if self.padding > 0:
      x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
    
    new_H = int((H + 2*self.padding - self.kernel_size) / self.stride + 1)
    new_W = int((W + 2*self.padding - self.kernel_size) / self.stride + 1)

    output = np.zeros((N, C, new_H, new_W))

    for h in range(0, new_H, self.stride):
      for w in range(0, new_W, self.stride):
        # (N C, K, K) -> (N C, 1, 1) and then put this into right positioj
        curr_selection = x[:, :, h:h+self.kernel_size, w:w+self.kernel_size]
        pooled = np.max(curr_selection, axis=(2,3), keepdims=True)
        output[:, :, h:h+1, w:w+1] = pooled

    return output


class Embedding:
  def __init__(self, vocab_size, embedding_dim):
    self.E = np.random.randn(vocab_size, embedding_dim)

  def forward(self, x):
    '''
    Input:
      x: (N, L)
    Output:
      result: (N, L, embdding_dim)
    '''
    res = []
    for i in range(x.shape[0]):
        curr = self.E[x[i], :] # (L, embedding_dim)
        res.append(curr)
    result = np.stack(res, axis=0)
    return result


class RNN:
  def __init__(self, embedding_dim, hidden_dim):
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.W_xh = np.random.rand(hidden_dim, embedding_dim)
    self.W_hh = np.random.rand(hidden_dim, hidden_dim)
    self.b_xh = np.random.rand(hidden_dim)
    self.b_hh = np.random.rand(hidden_dim)


  def forward(self, x):
    '''
    input:
      x: (N, L, embeeding_dim)
    outputs:
      output: (N, L, hidden_dim)
      last_hidden_state: (1, N, hidden_dim)
    '''
    batch_size, seq_length = x.shape[0], x.shape[1]
    h_0 = np.zeros((batch_size, self.hidden_dim))
    hidden_states = []
    hidden_states.append(h_0)
    for i in range(seq_length):
        next_h = np.dot(x[:, i, :], self.W_xh.T) + self.b_xh + np.dot(hidden_states[-1], self.W_hh.T) + self.b_hh
        next_h = HelperFunction.tanh(next_h) # (N, hidden_dim)
        hidden_states.append(next_h)

    output = np.stack(hidden_states[1:], axis=1)
    last_hidden_state = hidden_states[-1]

    return output, last_hidden_state[np.newaxis, :]


class LSTM:
  def __init__(self, input_size, hidden_size):
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.W_ii = np.random.rand(hidden_size, input_size)
    self.W_hi = np.random.rand(hidden_size, hidden_size)
    self.W_if = np.random.rand(hidden_size, input_size)
    self.W_hf = np.random.rand(hidden_size, hidden_size)
    self.W_ig = np.random.rand(hidden_size, input_size)
    self.W_hg = np.random.rand(hidden_size, hidden_size)
    self.W_io = np.random.rand(hidden_size, input_size)
    self.W_ho = np.random.rand(hidden_size, hidden_size)

    self.b_ii = np.random.rand(hidden_size)
    self.b_hi = np.random.rand(hidden_size)
    self.b_if = np.random.rand(hidden_size)
    self.b_hf = np.random.rand(hidden_size)
    self.b_ig = np.random.rand(hidden_size)
    self.b_hg = np.random.rand(hidden_size)
    self.b_io = np.random.rand(hidden_size)
    self.b_ho = np.random.rand(hidden_size)

  def one_cell(self, x_t, c_prev, h_prev):
    '''
    Inputs:
      x_t: (N, 1, input_size)
      c_prev: (N, hidden_size)
      h_prev: (N, hidden_size)

    Outputs:
      c_t: current cell state, (N, hidden_dize)
      h_t: current hidden state, (N, hidden_size)

    '''
    i_t = HelperFunction.sigmoid(np.dot(x_t, self.W_ii.T) + self.b_ii + np.dot(h_prev, self.W_hi.T) + self.b_hi) # (N, hidden_size)
    f_t = HelperFunction.sigmoid(np.dot(x_t, self.W_if.T) + self.b_if + np.dot(h_prev, self.W_hf.T) + self.b_hf) # (N, hidden_size)
    g_t = HelperFunction.sigmoid(np.dot(x_t, self.W_ig.T) + self.b_ig + np.dot(h_prev, self.W_hg.T) + self.b_hg) # (N, hidden_size)
    o_t = HelperFunction.sigmoid(np.dot(x_t, self.W_io.T) + self.b_io + np.dot(h_prev, self.W_ho.T) + self.b_ho) # (N, hidden_size)

    c_t = np.multiply(f_t, c_prev) + np.multiply(i_t, g_t)
    h_t = np.multiply(o_t, np.tanh(c_t))

    return c_t, h_t

  def forward(self, x):
    '''
    Input:
      x: (N, L, input_size)

    Outputs:
      output: (N, L, hidden_size)
      last_hidden_state: (1, N, hidden_size)
      last_cell_state: (1, N, hidden_size)
    '''
    batch_size, seq_length = x.shape[0], x.shape[1]

    h_0 = np.zeros((batch_size, self.hidden_size))
    c_0 = np.zeros((batch_size, self.hidden_size))
    h_records = []
    c_records = []

    h_records.append(h_0)
    c_records.append(c_0)
    for i in range(seq_length):
      x_t = x[:, i, :]
      c_t, h_t = self.one_cell(x_t, c_records[-1], h_records[-1])
      h_records.append(h_t)
      c_records.append(c_t)

    output = np.stack(h_records[1:], axis=1)

    last_hidden_state =  h_records[-1]
    last_cell_state = c_records[-1]


    return output, last_hidden_state[np.newaxis, :], last_cell_state[np.newaxis, :]


class MultiHeadAttention:
  def __init__(self, embedding_dim, num_heads, source_length, target_length):
    self.d_k = embedding_dim // num_heads
    self.d_v = embedding_dim // num_heads
    self.num_heads = num_heads
    self.source_length = source_length
    self.target_length = target_length

    self.casual_attn_mask = self.create_attn_mask()

    self.W_Qs = [np.random.rand(embedding_dim, self.d_k) for _ in range(num_heads)]
    self.W_Ks = [np.random.rand(embedding_dim, self.d_k) for _ in range(num_heads)]
    self.W_Vs = [np.random.rand(embedding_dim, self.d_v) for _ in range(num_heads)]
    self.W_O = np.random.rand(num_heads*self.d_v, embedding_dim)


  def create_attn_mask(self):
    ''' Create a casual attention mask for LookAhead MultiHeadAttenion Block in Decoder
    Shape of attn mask is (target_length, source_length)
    '''
    if self.target_length != 0:
      base = np.zeros((self.target_length, self.source_length))
    else:
      base = np.zeros((self.source_length, self.source_length))
    row, col = base.shape[0], base.shape[1]

    for r in range(row-1):
      for c in range(col):
        if c > r:
          base[r, c] = np.NINF
    return base # (target_length, source_length)

  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

  def forward(self, Q, K, V, casual=False):
    '''
    Q: (N, source_length, embedding_dim)
    K: (N, target_length, embedding_dim)
    V: (N, target_length, embedding_dim)
    '''
    attentions = []
    for i in range(self.num_heads):
      QW = np.dot(Q, self.W_Qs[i]) # (N, target_length, d_k)
      KW = np.dot(K, self.W_Ks[i]) # (N, source_length, d_k)
      VW = np.dot(V, self.W_Vs[i]) # (N, source_length, d_v)

      curr_attention = np.matmul(QW, KW.transpose(0, 2, 1)) / np.sqrt(self.d_k) # (N, target_length, source_length)

      if casual:
        curr_attention += self.casual_attn_mask # (N, target_length, source_length)

      curr_attention = self.softmax(curr_attention) # (N, target_length, source_length)
      curr_attention = np.matmul(curr_attention, VW) # (N, target_length, source_length)@(N, source_length, d_v) => (N, target_length, d_v)
      attentions.append(curr_attention)

    attentions = np.concatenate(attentions, axis=-1)
    result = np.dot(attentions, self.W_O) # (N, target_length, num_heads*d_v) @ (num_heads*d_v, embedding_dim) => (N, target_length, embedding_dim)
    return result #(N, target_length, embedding_dim)