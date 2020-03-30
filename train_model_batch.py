from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras import regularizers, optimizers
from keras.layers import Input
from keras.preprocessing import image
from sklearn.metrics import hamming_loss, confusion_matrix
import matplotlib.pyplot as plt
import keras
from keras.models import Model, load_model
from keras.applications import MobileNet,ResNet50
from PIL import ImageFile
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.metrics import average_precision_score
from keras import backend as K

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import pandas as pd
import numpy as np

df=pd.read_csv('dataset/dataset/label/train.csv')
df_test=pd.read_csv('dataset/dataset/label/test.csv')

columns=["Gun",'Knife','Wrench','Pliers','Scissors']

df['name']= df['name'].astype(str)+'.jpg'
df_test['name']= df_test['name'].astype(str)+'.jpg'

df = df.sample(frac=1)
NUM_ITEMS = 5
PIXEL=100 #

SIZE_TRAIN=39100
SIZE_VALIDATION = 10000
SIZE_TEST = 12344

DATASET_PATH = 'dataset/dataset'
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(
                                                dataframe=df[:SIZE_TRAIN],
                                                directory=DATASET_PATH+'/train',
                                                x_col="name",
                                                y_col=columns,
                                                batch_size=128,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="other",
                                                target_size=(PIXEL,PIXEL))

valid_generator=test_datagen.flow_from_dataframe(
                                                    dataframe=df[SIZE_TRAIN:SIZE_TRAIN+SIZE_VALIDATION],
                                                    directory=DATASET_PATH+'/train',
                                                    x_col="name",
                                                    y_col=columns,
                                                    batch_size=128,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="other",
                                                    target_size=(PIXEL,PIXEL))
test_generator=test_datagen.flow_from_dataframe(
                                                    dataframe=df_test,
                                                    directory=DATASET_PATH+'/test',
                                                    x_col="name",
 						    y_col=columns,
                                                    batch_size=50,
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode=None,
                                                    target_size=(PIXEL,PIXEL))
def cal_accuracy_at_least1(predict,y_test):
    accuracy = 0
    temp_array = predict * y_test
    row_with_all_zero = np.sum(~temp_array.any(1))
    return (len(y_test)-row_with_all_zero)/len(y_test)
def cal_accuracy_all(predict,y_test):
    return np.sum(np.all(predict == y_test, axis=1))/len(y_test)
def create_model_1():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(PIXEL,PIXEL,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_ITEMS, activation='sigmoid'))
    return model
def create_model_2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(PIXEL,PIXEL,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_ITEMS, activation='sigmoid'))
    return model
def create_MobileNet(models='MobileNet'):
    if models == 'MobileNet' and PIXEL==100:
        model_weight = None
        base_model=MobileNet(weights=model_weight,include_top=False,input_shape=(PIXEL,PIXEL,3)) #imports the mobilenet model and discards the last 1000 neuron layer.
    elif PIXEL==224:
        model_weight = 'imagenet'
        base_model=ResNet50(weights=model_weight,include_top=False,input_shape=(PIXEL,PIXEL,3)) #imports the resnet50 model and discards the last neuron layer.
    # base_model=MobileNet(weights=model_weight,include_top=False,input_shape=(PIXEL,PIXEL,3)) #imports the mobilenet model and discards the last 1000 neuron layer.
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(NUM_ITEMS,activation='sigmoid')(x)
    return Model(inputs=base_model.input,outputs=preds)


def main():
    ''' Specify models to be used '''
    ''' ####################### TRAIN CODE ##################################'''
    model = create_model_1()
    # model = create_model_2()
    # model = create_MobileNet()

    ''' Specify optimizers and other hyperparameters'''
    # model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        # epochs=10,
                        epochs=2,
                        workers=6,
                        use_multiprocessing=False
    )
    model.save("cnn_model.h5")


    '''
    ####################### TEST CODE #####################################
    model = load_model('cnn_model_2.h5')
    predict = model.predict_generator(test_generator,steps = STEP_SIZE_TEST)
    original_predict = predict.copy() # make a copy

    predict[predict>=0.4] = 1.0
    predict[predict<0.4] = 0.0

    y_test = np.array(df[SIZE_TRAIN+SIZE_VALIDATION:SIZE_TRAIN+SIZE_VALIDATION+SIZE_TEST].drop(['imdb_id', 'genres','poster_path'],axis=1))


    accuracy_at_least1 = cal_accuracy_at_least1(predict,y_test)
    accuracy_all = cal_accuracy_all(predict,y_test)
    hammingloss = hamming_loss(predict,y_test)

    cm = confusion_matrix(predict.argmax(axis=1),y_test.argmax(axis=1))
    recall_ = np.diag(cm) / (np.sum(cm, axis = 1) + K.epsilon())
    precision_ = np.diag(cm) / (np.sum(cm, axis = 0)+ K.epsilon())
    f1_ =  (2*recall_*precision_)/(precision_+recall_+ K.epsilon())

    # print("Confusion matrx: ",cm)
    print("Precision: ",precision_)
    print("Recall: ",recall_)
    print("F1: ",f1_)
    print("Accuracy at least one genre: ",accuracy_at_least1)
    print("Accuracy for all genre: ",accuracy_all)
    print("Hamming_loss: ",hammingloss)

    plot_recall_vs_precision(original_predict,y_test)
    plot_ROC_curve(original_predict,y_test)
    '''
if __name__ == '__main__':
    main()
