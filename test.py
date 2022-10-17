import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import sys
from train import datasets
from train import Patches
from train import PatchEncoder
from keras import backend as K
from os import listdir
from os.path import isfile, join

def Predict(model,datas):
    (x_train, y_train), (x_test, y_test), num_classes = datasets(datas)
    
    test_eval = model.evaluate(x_test, y_test, verbose=0)
    accuracy = test_eval[1]

    #predict and format output to use with sklearn
    predict = model.predict(x_test)
    #print(np.shape(predict))
    predict = np.argmax(predict, axis=1)
    #print(np.shape(predict))
    #macro precision and recall
    precisionMacro = precision_score(y_test, predict, average='macro')
    recallMacro = recall_score(y_test, predict, average='macro')
    #micro precision and recall
    precisionMicro = precision_score(y_test, predict, average='micro')
    recallMicro = recall_score(y_test, predict, average='micro')
    #Confusion Matrix 
    confMat = confusion_matrix(y_test, predict)
    #F1 scores
    F1Macro = 2*((precisionMacro*recallMacro)/(precisionMacro+recallMacro+K.epsilon()))
    F1Micro = 2*((precisionMicro*recallMicro)/(precisionMicro+recallMicro+K.epsilon()))
    print("Test accuracy: ", accuracy)
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print("Macro F1: ", F1Macro)
    print("Micro F1: ", F1Micro)
    print(confMat)   
    Metrics = [accuracy,recallMicro,recallMacro,precisionMicro,precisionMacro,F1Micro,F1Macro,confMat]
    return Metrics

def Test(datas,size):
    print("Loading Test Data")
    (x_train, y_train), (x_test, y_test), num_classes = datasets(datas)
    print("Loading CNN model")
    model_CNN = tf.keras.models.load_model("./"+datas+'_'+size+'_CNN'+".h5")
    print("Making predictions on test data for CNN")
    Predict(model_CNN,datas)
    print("Loading Vision Transformer model")
    model_ViT = tf.keras.models.load_model("./"+datas+'_'+size+'_vision_transformer'+".h5",custom_objects={'Patches': Patches, 'PatchEncoder':PatchEncoder})
    print("Making predictions on test data for Vision Transformer")
    Predict(model_ViT,datas)
    
def main():
    Size = sys.argv[1]
    Dataset = sys.argv[2]
   
    Test(Dataset,Size)
    


if __name__ == "__main__":
    main()