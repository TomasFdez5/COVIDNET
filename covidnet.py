import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.gen_math_ops import mul

"""from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
try:
	k.set_session(sess)
except:
	print('No session available')

import keras"""
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse


# -------------------------------------------------------------------------------------------------------


def create_cnn(seg_shape):
    """
    Create convolutional neural networks in order to obtain the representative of a segment of the image (element of the bag).
    Inputs:
        - seg_shape: Shape of the image segment.
    Output:
        - cnn: Convolutional neuronal network created.
    """
    cnn =  tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=seg_shape),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax',name='output_layer')
    ])
    return cnn

def generate_data(gen):
    """
    Given the generator of a dataset, load the data by performing the corresponding segmentation.
    Inputs:
        - n_ver: Number of vertical slices.
        - n_hor: Number of horizontal slices.
        - gen: Generator of a data set.
    Outputs:
        - segment_list: List that stores the vectors corresponding to the segments of all the images.
        - labels: Array with the labels of the loaded images.
    """
    data_count = 0
    labels = []
    segment_list = []
    segment_list.append([])
    
    # Generate all the segments for all the images.
    while data_count != gen.n:
        dat = gen.next()
        img= dat[0].reshape(224,224,3)

        segment_list[0].append(img)

        labels.append(np.reshape(dat[1],2))

        data_count +=1

    segment_list[0] = np.array(segment_list[0])

    return segment_list, np.array(labels)


def report(history,real,pred,file):
    """
    Given the training history, the actual and predicted labels, the results obtained are reported.
    Inputs:
        - history: Training history.
        - real: Real labels.
        - pred: Predicted labels.
        - file: Name of the file in case the user want to save the results.
    """
    # TP, TN, FP, FN
    tp = sum(r==p and r==0 for r,p in zip(real,pred))
    tn = sum(r==p and r==1 for r,p in zip(real,pred))
    fn = sum(r!=p and r==0 for r,p in zip(real,pred))
    fp = sum(r!=p and r==1 for r,p in zip(real,pred))

    # Accuracy
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    # Precision
    prec = tp/(tp+fp)
    # Recall
    sensitivity = tp/(tp+fn)
    # F1-score
    f1score = 2*((prec*sensitivity)/(prec+sensitivity))
    # Specificity
    spe = tn / (tn+fp)

    # Results
    if file != None:
        with open(str(file),"w") as f:
            f.write('\t-- Train Accuracy --\n')
            f.write("\nMaximum: {}".format(max(np.array(history.history['acc']))))
            f.write("\nMinimum: {}".format(min(np.array(history.history['acc']))))
            f.write("\nMean: {}".format(np.mean(np.array(history.history['acc']))))
            f.write("\nTypical deviation: {}".format(np.std(np.array(history.history['acc']))))
            f.write('\n\t-- Train Loss --\n')
            f.write("\nMaximum: {}".format(max(np.array(history.history['loss']))))
            f.write("\nMinimum: {}".format(min(np.array(history.history['loss']))))
            f.write("\nMean: {}".format(np.mean(np.array(history.history['loss']))))
            f.write('\t-- Validation Accuracy --\n')
            f.write("\nMaximum: {}".format(max(np.array(history.history['val_acc']))))
            f.write("\nMinimum: {}".format(min(np.array(history.history['val_acc']))))
            f.write("\nMean: {}".format(np.mean(np.array(history.history['val_acc']))))
            f.write("\nTypical deviation: {}".format(np.std(np.array(history.history['val_acc']))))
            f.write('\n\t-- Validation Loss --\n')
            f.write("\nMaximum: {}".format(max(np.array(history.history['val_loss']))))
            f.write("\nMinimum: {}".format(min(np.array(history.history['val_loss']))))
            f.write("\nMean: {}".format(np.mean(np.array(history.history['val_loss']))))
            f.write("\nTypical deviation: {}".format(np.std(np.array(history.history['val_loss']))))
            f.write("\n\t-- Test evaluation--\n\n Confusion matrix:\n( TP:{}  FP: {} )\n( FN:{}  TN: {} )\n\nAccuracy: {}\nSensitivity: {}\nSpecificity:{}\n\nOther metrics:\nPrecision: {}\nF1-score:{}\n".format(tp,fp,fn,tn,accuracy,sensitivity,spe,prec,f1score))
    else:
        print('\t-- Train Accuracy --\n')
        print("Maximum: ", max(np.array(history.history['acc'])))
        print("Minimum: ", min(np.array(history.history['acc'])))
        print("Mean: ", np.mean(np.array(history.history['acc'])))
        print("Typical deviation: ", np.std(np.array(history.history['acc'])))
        print('\n\t-- Train Loss --\n')
        print("Maximum: ", max(np.array(history.history['loss'])))
        print("Minimum: ", min(np.array(history.history['loss'])))
        print("Mean: ", np.mean(np.array(history.history['loss'])))
        print("Typical deviation: ", np.std(np.array(history.history['loss'])))
        print('\t-- Validation Accuracy --\n')
        print("Maximum: ", max(np.array(history.history['val_acc'])))
        print("Minimum: ", min(np.array(history.history['val_acc'])))
        print("Mean: ", np.mean(np.array(history.history['val_acc'])))
        print("Typical deviation: ", np.std(np.array(history.history['val_acc'])))
        print('\n\t-- Validation Loss --\n')
        print("Maximum: ", max(np.array(history.history['val_loss'])))
        print("Minimum: ", min(np.array(history.history['val_loss'])))
        print("Mean: ", np.mean(np.array(history.history['val_loss'])))
        print("Typical deviation: ", np.std(np.array(history.history['val_loss'])))
        print("\n\t-- Test evaluation--\n\n Confusion matrix:\n( TP:{}  FP: {} )\n( FN:{}  TN: {} )\n\nAccuracy: {}\nSensitivity: {}\nSpecificity:{}\n\nOther metrics:\nPrecision: {}\nF1-score:{}\n".format(tp,fp,fn,tn,accuracy,sensitivity,spe,prec,f1score))


def history_graph(history):
    """
    Generate the graphs corresponding to the training history.
    Input:
        - history: Training history.
    """

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training','Validation'], loc='upper right')
    plt.savefig('./model_accuracy.jpg')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training','Validation'], loc='upper right')
    plt.savefig('./model_loss.jpg')

# MAIN ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    # PARAMETER CONTROL
    parser = argparse.ArgumentParser(description='Executable to train and test MIL-CNN Covid-19 detection.')

    parser.add_argument('-d', dest='dataset',help='Directory of the dataset', required=True)
    parser.add_argument('-b', dest='batch_size',help='Size of the batch', required=False,default=32)
    parser.add_argument('-c', dest='modelCheckpoint', help='Save the state of the model when its loss improves', required=False)
    parser.add_argument('-e', dest='epochs',help='Number of epochs', required=False,default=40)
    parser.add_argument('-o', dest='file_output',help='Directory name to save results', required=False,default=None)
    parser.add_argument('-g',dest='graph', help='Save train historical graphs',action='store_true')

    args = parser.parse_args()

    # IMAGES GENERATION ----
    train_path = args.dataset+'/train'
    test_path = args.dataset+'/test'

    train_gen = ImageDataGenerator(rescale=1./255.,validation_split=0.2)
    test_gen = ImageDataGenerator(rescale=1./255.)

    if int(args.batch_size) < 0:
        print("[ERROR] Batch size must to be greater than 0. Setting to default value (16)...")
        batchSize = 32
    else:
        batchSize = int(args.batch_size)

    gtrain = train_gen.flow_from_directory(
        train_path,
        target_size=(224,224),
        batch_size=batchSize, 
        class_mode='categorical',
        subset='training' # Used for training subset
    )

    gvalidation = train_gen.flow_from_directory(
        train_path,
        target_size=(224,224),
        batch_size=batchSize, 
        class_mode='categorical',
        subset='validation' # Used for validation subset
    )

    gtest = test_gen.flow_from_directory(
        test_path,
        target_size=(224,224),
        batch_size=1, # Size 1 to apply segmentation one by one.
        class_mode='categorical'
    )

    # DATASET GENERATION ----
    x_test,y_test = generate_data(gtest)

    # MODEL CREATION ----
    model = create_cnn((224,224,3))

    # MODEL TRAINING ----
    if int(args.epochs) < 0:
        print("[ERROR] Number of epochs must to be greater than 0. Setting to default value (15)...")
        nEpochs = 40
    else:
        nEpochs = int(args.epochs)

    lrate = 1e-4
    opt = tf.keras.optimizers.Adam(lr=lrate)
    callbacks = [] # List of objects that can perform actions at various stages of training
    

    # User want to save the state of the model when its loss improves
    if args.modelCheckpoint:
        if args.modelCheckpoint.find('.h5') == -1:
            print("\n[Warning] The extent of the model is not indicated. Setting to: /checkpoints/best_model.h5")
            checkpointdir = "./checkpoints/best_model.h5"
        else:
            checkpointdir = args.modelCheckpoint

        checkpoint = ModelCheckpoint(filepath=checkpointdir,save_weights_only=False,monitor='loss',verbose=1,save_best_only=True, mode='min')
        callbacks.append(checkpoint)

    # Stop training when a monitored metric has stopped improving.
    earlystop = EarlyStopping(monitor='loss',mode='min',patience=4,restore_best_weights=True,verbose=1)
    callbacks.append(earlystop)

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])
    #modelo.summary()
    history = model.fit(
        gtrain,
        epochs=nEpochs,
        batch_size=batchSize,
        steps_per_epoch=gtrain.n//batchSize,
        validation_data = gvalidation,
        validation_steps = gvalidation.n // batchSize,
        callbacks=callbacks
    )

    if args.modelCheckpoint:
        model.load_weights(checkpointdir)

    # VIEW MODEL TRAINING HISTORY ----
    if args.graph:
        history_graph(history)

    # MODEL TEST ----
    print("Nº datos test: ", gtest.n)
    prediccion = model.predict(x_test)
    y_pred = np.argmax(prediccion,axis=1)
    y_test = [0 if np.argmax(i)==0 else 1 for i in y_test]

    # REPORT ----
    report(history,y_test,y_pred,args.file_output)

if __name__ == "__main__":
    main()