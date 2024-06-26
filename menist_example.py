import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from Deeplearning_models import functional_model,MyCustommodel
from My_utils import display_some_example

# tensorflow.keras.Sequential
seq_model = tf.keras.Sequential(
[
    Input(shape=(28,28,1)),
     
    Conv2D(32,(3,3),activation='relu'),
    Conv2D(64,(3,3),activation='relu'),
    MaxPool2D(),
    BatchNormalization(),

    Conv2D(128,(3,3),activation='relu'),
    MaxPool2D(),
    BatchNormalization(),

    GlobalAvgPool2D(),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')

])

if __name__ == '__main__':
    (x_train,Y_train) , (x_test,Y_test) = tf.keras.datasets.mnist.load_data()

    print("X_train.shape",x_train.shape)
    print("Y_train.shape",Y_train.shape)
    print("X_test.shape",x_test.shape)
    print("Y_test.shape",Y_test.shape)

    if False:
        display_some_example(X_train,Y_train)
    
    # normlization just the gradiant move faster 
    x_train= x_train.astype('float32') / 255
    x_test= x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train,axis=-1)
    x_test = np.expand_dims(x_test,axis=-1)
    # categorical_crossentropy -> expect your output to be one hot encoded
    #sparse_categorical_crossentropy - > expect your output to be not one hot encoded
    if False:
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

    Y_train = tf.keras.utils.to_categorical(Y_train,10)
    Y_test = tf.keras.utils.to_categorical(Y_test,10)


    if False:
        model = functional_model()

    model = MyCustommodel()

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')



    # at the end of each epochs we will use validation to see the output and see if the metrics is getting better or not
    model.fit(x_train,Y_train, batch_size=64 , epochs=3 , validation_split=0.2)

    model.evaluate(x_test,Y_test,batch_size=64)
    

    y_pred =model.predict(x_test,batch_size=64)

    # display_some_example(y_pred,Y_test)


    #save the model or saving best model during the trainning 
