import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Embedding, Reshape, Dot, Concatenate
from tensorflow.keras.layers import GRU, SpatialDropout1D, Conv1D, GlobalMaxPooling1D,Multiply, Lambda, Softmax, Flatten, BatchNormalization, Bidirectional, dot, concatenate
from tensorflow.keras.layers import AdditiveAttention, Attention


class ILotto(tf.keras.layers.Layer):
    def __init__(self,
                 numbers = 37, 
                 embed_dim = 30,
                 dropout_rate = 0.5,
                 spatial_dropout_rate = 0.5,
                 steps_before = 10,
                 steps_after = 7,
                 hidden_neurons = [64, 32],
                 bidirectional = True,
                 attention_style = 'Bahdanau'
                 ):
        super(ILotto, self).__init__(name='')
        self.numbers = numbers
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.steps_before = steps_before
        self.steps_after = steps_after
        self.feature_count = embed_dim * steps_after
        self.hidden_neurons = hidden_neurons 
        self.bidirectional = bidirectional 
        self.attention_style = attention_style
    
    def build(self, input_shape):
        super(ILotto, self).build(input_shape)
            
    def call(self, inputs, training=None, mask=None):
        
        inp1 = Lambda(lambda x: x[:, :, 0])(inputs)
        inp1 = Embedding(self.numbers, self.embed_dim)(inp1)
        inp1 = SpatialDropout1D(self.spatial_dropout_rate)(inp1)
        
        inp2 = Lambda(lambda x: x[:, :, 1])(inputs)
        inp2 = Embedding(self.numbers, self.embed_dim)(inp2)
        inp2 = SpatialDropout1D(self.spatial_dropout_rate)(inp2)
        
        inp3 = Lambda(lambda x: x[:, :, 2])(inputs)
        inp3 = Embedding(self.numbers, self.embed_dim)(inp3)
        inp3 = SpatialDropout1D(self.spatial_dropout_rate)(inp3)
        
        inp4 = Lambda(lambda x: x[:, :, 3])(inputs)
        inp4 = Embedding(self.numbers, self.embed_dim)(inp4)
        inp4 = SpatialDropout1D(self.spatial_dropout_rate)(inp4)
        
        inp5 = Lambda(lambda x: x[:, :, 4])(inputs)
        inp5 = Embedding(self.numbers, self.embed_dim)(inp5)
        inp5 = SpatialDropout1D(self.spatial_dropout_rate)(inp5)    
        
        inp6 = Lambda(lambda x: x[:, :, 5])(inputs)
        inp6 = Embedding(self.numbers, self.embed_dim)(inp6)
        inp6 = SpatialDropout1D(self.spatial_dropout_rate)(inp6)
        
        inp7 = Lambda(lambda x: x[:, :, 6])(inputs)
        inp7 = Embedding(self.numbers, self.embed_dim)(inp7)
        inp7 = SpatialDropout1D(self.spatial_dropout_rate)(inp7)
        
        inp = Concatenate()([inp1, inp2, inp3, inp4, inp5, inp6, inp7])

        # Seq2Seq model with attention or bidirectional encoder
    
        num_layers = len(self.hidden_neurons)
        
        sh_list, h_list, c_list = [inp], [], []
        
        if self.bidirectional:
            for i in range(num_layers):
    
                sh, fh, fc, bh, bc = Bidirectional(LSTM(self.hidden_neurons[i],
                                                        dropout = self.dropout_rate, 
                                                        return_state = True, 
                                                        return_sequences = True))(sh_list[-1])
            
                h = Concatenate()([fh, bh])
                c = Concatenate()([fc, bc]) 

                sh_list.append(sh)
                h_list.append(h)
                c_list.append(c)
        
        else:
            for i in range(num_layers):

                sh, h, c = LSTM(self.hidden_neurons[i], 
                                dropout = self.dropout_rate,
                                return_state = True, 
                                return_sequences = True)(sh_list[-1])

                sh_list.append(sh)
                h_list.append(h)
                c_list.append(c)
        
        decoder = RepeatVector(self.steps_after)(h_list[-1])
    
        if self.bidirectional:
            
            decoder_hidden_neurons = [hn * 2 for hn in self.hidden_neurons]
            
        else:
            
            decoder_hidden_neurons = self.hidden_neurons
        
        for i in range(num_layers):
            decoder = LSTM(decoder_hidden_neurons[i],
                       dropout = self.dropout_rate, 
                       return_sequences = True)(decoder, initial_state = [h_list[i], c_list[i]])
       
        if self.attention_style == 'Bahdanau':
            
            context = AdditiveAttention(dropout = self.dropout_rate)([decoder, sh_list[-1]])
            
            decoder = concatenate([context, decoder])
            
        elif self.attention_style == 'Luong':
            context = Attention(dropout = self.dropout_rate)([decoder, sh_list[-1]])
        
        decoder = concatenate([context, decoder])
    
        out = Dense(self.numbers, activation = 'softmax')(decoder)

        return out


if __name__ == "__main__":
    
    inputs = Input(shape = (10, 7))
    output = ILotto()(inputs)
    model = tf.keras.Model( inputs, output)
    
    sparse_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(k = 5, name = 'sparse_top_k')
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [sparse_top_k])
    model.summary()