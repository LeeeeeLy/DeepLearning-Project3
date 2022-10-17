import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
from sklearn.metrics import precision_score, recall_score, confusion_matrix
#from tensorflow.keras.utils import to_categorical
import sys
import keras
from keras.models import Sequential,Input,Model
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from os import listdir
from os.path import isfile, join
import tensorflow_addons as tfa

#Load data
def datasets(datas):
    if datas == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10
    elif datas == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        num_classes = 100 
    elif datas == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        num_classes = 10
        #add 1 more dim
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    else:
        print('wrong dataset!')
            
    
    #one hot
    #y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    #y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    return (x_train, y_train), (x_test, y_test), num_classes
#Vision Transformer   
class Patches(layers.Layer):
    
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        return {"patch_size": self.patch_size}

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim=projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.projection_dim}

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def ViTransformer(datas,size):
    #ViTransformer model
    
    #config
    learning_rate = 0.0001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 50
    image_size = 72
    patch_size = 6
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
    #read in data
    (x_train, y_train), (x_test, y_test), num_classes = datasets(datas)
    outputNeurons = num_classes
    trainingData = x_train
    shape = x_train.shape[1:4]
    
    if size == "tiny":
        transformer_layers = 4
    elif size == "small":
        transformer_layers = 6
    elif size == "base":
        transformer_layers = 8
    
    #do some data preprocessing
    data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(trainingData)
    
    #set input layer/shape
    inputs = layers.Input(shape=shape)

    # Augment data
    augmented = data_augmentation(inputs)

    # Create patches
    patches = Patches(patch_size)(augmented)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Final normalization/output
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    #add mlp to transformer
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    #pass features from mlp to final dense layer/classification
    logits = layers.Dense(outputNeurons)(features)

    #create model
    model = keras.Model(inputs=inputs, outputs=logits)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model.summary()
    return model    
    
    
def CNN(datas,size):
    #CNN model
    
    #initialization
    epochs = 50
    batch_size = 64
    Learning_rate = 0.0001
    dropout = 0
    #none, l1, l2
    Regularization = 'none'
    #Adam, SGD, RMSprop
    Optimizerr = 'Adam'
    #sparse_categorical_crossentropy, Poisson, KLDivergence
    Loss_function = 'sparse_categorical_crossentropy'
    (x_train, y_train), (x_test, y_test), num_classes = datasets(datas)
    #normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #build model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=x_train.shape[1:4]))
    model.add(MaxPooling2D((2, 2)))
    if size != 'tiny':
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        if size != 'small':
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
    
    model.add(Dropout(dropout))
    model.add(Flatten())
    if Regularization == 'none':
        model.add(Dense(512, activation='relu'))
    elif Regularization == 'l1':
        model.add(Dense(512, kernel_regularizer=tf.keras.regularizers.l1(0.2), activation='relu'))
    else:
        model.add(Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.2), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    if Optimizerr == 'Adam':
        model.compile(loss=Loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_rate),metrics=['accuracy'])
    elif Optimizerr == 'SGD':
        model.compile(loss=Loss_function, optimizer=tf.keras.optimizers.SGD(learning_rate=Learning_rate),metrics=['accuracy'])
    else:
        model.compile(loss=Loss_function, optimizer=tf.keras.optimizers.RMSprop(learning_rate=Learning_rate),metrics=['accuracy'])
    model.summary()

    #cifar10_train = model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))
    #test_eval = model.evaluate(x_test, y_test, verbose=0)
   
    return model
    
def Train(datas, size):
    #call both transformer and CNN and save both model
    model_ViT = ViTransformer(datas,size)
    model_CNN = CNN(datas,size)
    model_ViT.save("./"+datas+'_'+size+'_vision_transformer'+".h5")
    print("Vision Transformer model saved.")
    model_CNN.save("./"+datas+'_'+size+'_CNN'+".h5")
    print("CNN model saved.")
    #CNN compile
    numbatchs = 128
    numepochs = 50
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath="./"+datas+'_'+size+'_CNN'+".h5", 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max'),
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.01, patience=2)
                  ]
                
    (x_train, y_train), (x_test, y_test), num_classes = datasets(datas)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model_CNN = CNN(datas,size)
    model_CNN.fit(x_train, y_train,
               batch_size=numbatchs,
               epochs=numepochs,
               validation_data=[x_test, y_test], callbacks=checkpoint)
    print("Best CNN model saved.")
    #ViT compile
    numbatchs = 256
    numepochs = 50
    (x_train, y_train), (x_test, y_test), num_classes = datasets(datas)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="./"+datas+'_'+size+'_vision_transformer'+".h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    model_ViT = ViTransformer(datas,size)
    model_ViT.fit(
        x=x_train,
        y=y_train,
        batch_size=numbatchs,
        epochs=numepochs,
        validation_data=[x_test, y_test],
        callbacks=checkpoint_callback
    )
    print("Best Vision Transformer model saved.")
def main():
    Size = sys.argv[1]
    Dataset = sys.argv[2]
   
    Train(Dataset,Size)


if __name__ == "__main__":
    main()