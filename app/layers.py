# Custom L1Dist (L1 Distance layer) module
# WHY: we need this to load our custom model 
#import dependencies 
import tensorflow as tf
from tensorflow.keras.layers import Layer 

# Custom L1 Distance layer from N/B 
class L1Dist(Layer):
    def __init__(self,**kwargs): 
        super().__init__() #inheritance
    
    # this is the logic here (the difference b/w the input_embedding and the validation_embedding)
    def call(self,input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)