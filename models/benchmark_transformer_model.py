import tensorflow as tf


###MultiHeadSelfAttention
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads = 8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimensions = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b = True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis = -1)
        output = tf.matmul(weights, value)
        return output, weights
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output
    

###Transformer Keras Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation = "relu"), tf.keras.layers.Dense(embed_dim),])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)


class Transformer_Portfolio(tf.keras.layers.Layer):
    def __init__(self, shape1, shape2, outputShape, headsAttention, dropout, learningRate):
        self.shape1 = shape1
        self.shape2 = shape2
        self.outputShape = outputShape
        self.headsAttention = headsAttention
        self.dropout = dropout
        self.learningRate = learningRate
        self.model = None

    def Transformer_Model(self):
        #Model Structure is defined
        Input = tf.keras.Input(shape = (self.shape1, self.shape2), name = 'Input')
        
        #Pos_Encoding is Used for Vanilla Transformer
        
        #Transformer architecture is implemented
        transformer_block_1 = TransformerBlock(embed_dim = 34, num_heads=self.headsAttention, ff_dim = 8, rate = self.dropout, )
        X = transformer_block_1(Input)

        #Dense layers are used
        X = tf.keras.layers.GlobalAveragePooling1D()(X)
        X = tf.keras.layers.Dense(8, activation=tf.nn.sigmoid)(X)
        X = tf.keras.layers.Dropout(self.dropout)(X)
        Output = tf.keras.layers.Dense(self.outputShape, activation=tf.nn.softmax, name="Output")(X)

        model = tf.keras.Model(inputs=Input, outputs=Output)
        #Optimizer is defined
        Opt = tf.keras.optimizers.Adam(learning_rate=self.learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')

        #Configuring Custom Loss Funciton with Mean Sharpe Ratio
        def sharpe_loss(y_true, y_pred):
            y_true = tf.concat((tf.zeros((1, y_true.shape[1]), tf.float32), y_true[:-1, :]), axis = 0)
            portfolio_returns = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)
            #portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1] 
            sharpe = tf.math.divide(tf.keras.backend.mean(portfolio_returns),tf.keras.backend.std(portfolio_returns))
            return -sharpe

        #Model is compiled
        model.compile(optimizer=Opt, loss= sharpe_loss)
        return model

    def allocation_vanilla_train(self, xtrainRNN, ytrainRNN, Epochs, BatchSize):
        if self.model == None:
            self.model = self.Transformer_Model()
            self.model.fit(xtrainRNN, ytrainRNN, epochs = Epochs, verbose = 0, batch_size = BatchSize)
        else: 
            self.model.fit(xtrainRNN, ytrainRNN, epochs = Epochs, verbose = 0, batch_size = BatchSize)
    
    def allocation_vanilla_test(self, xtestRNN):
            if self.model == None: 
                print('Model is not trained.')
            else: 
                return self.model.predict(xtestRNN)
    def vanilla_save_weights(self, path): 
        return self.model.save_weights(path)

    def vanilla_load_weights(self, path): 
        if self.model == None:
            self.model = self.Transformer_Model()
            self.model.load_weights(path)
        else: 
            self.model.load_weights(path)

if __name__ == '__main__':
    pass


