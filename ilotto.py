import math
import tensorflow as tf

from tensorflow.keras import callbacks
from tensorflow.keras import backend, saving
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, Embedding, Concatenate
from tensorflow.keras.layers import SpatialDropout1D, Lambda, Bidirectional, concatenate
from tensorflow.keras.layers import AdditiveAttention, Attention


@saving.register_keras_serializable(package="ILotto")
class ILotto(tf.keras.Model):
    def __init__(
        self,
        numbers=37,
        embed_dim=30,
        dropout_rate=0.5,
        spatial_dropout_rate=0.5,
        steps_before=10,
        steps_after=7,
        hidden_neurons=[64, 32],
        bidirectional=True,
        attention_style="Bahdanau",
        shape=(None, 10, 7),
        **kwargs,  # Add kwargs to support additional arguments like `trainable`

    ):
        super(ILotto, self).__init__(name="")
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
        self.in_shape = shape
        self.embedding = []
        self.spatial_drop = []
        for i in range(shape[-1]):
            self.embedding.append(Embedding(self.numbers, self.embed_dim))
            self.spatial_drop.append(SpatialDropout1D(self.spatial_dropout_rate))

        self.bidirectional_lstm = []
        self.lstm = []
        for i in range(len(self.hidden_neurons)):
            if self.bidirectional:

                self.bidirectional_lstm.append(
                    Bidirectional(
                        LSTM(
                            self.hidden_neurons[i],
                            dropout=self.dropout_rate,
                            return_state=True,
                            return_sequences=True,
                        )
                    )
                )

            else:
                self.lstm.append(
                    LSTM(
                        self.hidden_neurons[i],
                        dropout=self.dropout_rate,
                        return_state=True,
                        return_sequences=True,
                    )
                )

        if bidirectional:
            self.decoder_hidden_neurons = [hn * 2 for hn in hidden_neurons]
        else:
            self.decoder_hidden_neurons = hidden_neurons

        self.lstm_decoder = []
        for i in range(len(self.hidden_neurons)):
            self.lstm_decoder.append(
                LSTM(
                    self.decoder_hidden_neurons[i],
                    dropout=dropout_rate,
                    return_sequences=True,
                )
            )

        self.repeat_vec = RepeatVector(self.steps_after)

        if self.attention_style == "Bahdanau":
            self.additive_attention = AdditiveAttention(dropout=self.dropout_rate)
        else:
            self.attention = Attention(dropout=self.dropout_rate)
        self.dense_out = Dense(self.numbers, activation="softmax")

        self.build(shape)

    def build(self, input_shape):
        super(ILotto, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):

        inp = []
        for i in range(self.in_shape[-1]):
            inp_d = Lambda(lambda x: x[:, :, i])(inputs)
            inp_dd = self.embedding[i](inp_d)
            inp.append(self.spatial_drop[i](inp_dd))

        inp = Concatenate()(inp)

        # Seq2Seq model with attention or bidirectional encoder

        num_layers = len(self.hidden_neurons)

        sh_list, h_list, c_list = [inp], [], []

        for i in range(num_layers):
            if self.bidirectional:

                sh, fh, fc, bh, bc = self.bidirectional_lstm[i](sh_list[-1])

                h = Concatenate()([fh, bh])
                c = Concatenate()([fc, bc])

                sh_list.append(sh)
                h_list.append(h)
                c_list.append(c)

            else:
                sh, h, c = self.lstm[i](sh_list[-1])

                sh_list.append(sh)
                h_list.append(h)
                c_list.append(c)

        decoder = self.repeat_vec(h_list[-1])

        for i in range(num_layers):
            decoder = self.lstm_decoder[i](
                decoder, initial_state=[h_list[i], c_list[i]]
            )

        if self.attention_style == "Bahdanau":

            context = self.additive_attention([decoder, sh_list[-1]])

            decoder = concatenate([context, decoder])

        elif self.attention_style == "Luong":
            context = self.attention([decoder, sh_list[-1]])

        decoder = concatenate([context, decoder])

        out = self.dense_out(decoder)

        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "numbers": self.numbers,
            "embed_dim": self.embed_dim,
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "steps_before": self.steps_before,
            "steps_after": self.steps_after,
            "hidden_neurons": self.hidden_neurons,
            "bidirectional": self.bidirectional,
            "attention_style": self.attention_style,
            "shape": self.in_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)