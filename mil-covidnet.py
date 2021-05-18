import tensorflow as tf

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
    x_input = tf.keras.layers.Input(shape=seg_shape)
    x = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(x_input)
    x =  tf.keras.layers.MaxPool2D(2,2)(x)
    x = tf.keras.layers.Conv2D(64,(3,3),activation='relu')(x)
    x =  tf.keras.layers.MaxPool2D(2,2)(x)
    x = tf.keras.layers.Conv2D(128,(3,3),activation='relu')(x)
    x =  tf.keras.layers.MaxPool2D(2,2)(x)
    x = tf.keras.layers.Attention(128)([x,x])
    #x_output = tf.keras.layers.GlobalMaxPool2D()(x)
    #x_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    x_output = tf.keras.layers.Flatten()(x)
    
    cnn = tf.keras.Model(x_input,x_output)

    return cnn


def segment(n_ver,n_hor,image):
    """
    Given an image of the dataset, generate segments cutting the original image in n_view vertical slices and n_hor horizontal slices.
    Inputs:
        - n_ver: Number of vertical slices.
        - n_hor: Number of horizontal slices.
        - image: Original image.
    Output:
        - image_segments: Image segment list.
    """
    height,width,_ = image.shape
    image_copy = image # I STORE A COPY OF THE ORIGINAL IMAGE TO SEGMENT CORRECTLY

    image_segments=[]
    for ivert in range(n_hor):
        for ihor in range(n_ver):
            if ivert == 0: 
                y=0
            else:
                y = height*ivert // n_hor
            if ihor == 0: 
                x=0
            else:
                x = width*ihor // n_ver
            
            h = (height//n_hor)
            w = (width//n_ver)
            #print(x,y,h,w)

            image_segments.append(image[y:y+h,x:x+w])
            image = image_copy


    """
    # CODE TO SEE THE PERFORMED SEGMENTS 

    fig=plt.figure(figsize=(5,5))
    cols = n_ver
    fils = n_hor
    for i in range(1, cols*fils +1):
        img = image_segments[i-1]
        fig.add_subplot(fils, cols, i)
        plt.imshow(img)
    plt.show()
    """
    
    return image_segments


def generate_data(n_ver,n_hor,gen):
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
    for i in range(n_ver*n_hor):
        segment_list.append([])
    
    # Generate all the segments for all the images.
    while data_count != gen.n:
        dat = gen.next()
        img= dat[0].reshape(224,224,3)
        segments_obtained = segment(n_ver,n_hor,img)

        for i in range(n_ver*n_hor):
            segment_list[i].append(segments_obtained[i])

        labels.append(np.reshape(dat[1],2))

        data_count +=1

    #for i in range(n_ver*n_hor):
    #    segment_list[i] = np.array(segment_list[i])
    segment_list = [np.array(i) for i in segment_list]

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
    recall = tp/(tp+fn)
    # F1-score
    f1score = 2*((prec*recall)/(prec+recall))
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
            f.write("\nTypical deviation: {}".format(np.std(np.array(history.history['loss']))))
            f.write("\n\t-- Test evaluation--\n\n Confusion matrix:\n( TP:{}  FP: {} )\n( FN:{}  TN: {} )\n\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1-score:{}\nSpecificity:{}\n".format(tp,fp,fn,tn,accuracy,prec,recall,f1score,spe))
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
        print("\n\t-- Test evaluation--\n\n Confusion matrix:\n( TP:{}  FP: {} )\n( FN:{}  TN: {} )\n\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1-score:{}\nSpecificity:{}\n".format(tp,fp,fn,tn,accuracy,prec,recall,f1score,spe))


def history_graph(history):
    """
    Generate the graphs corresponding to the training history.
    Input:
        - history: Training history.
    """

    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Training'], loc='upper right')
    plt.savefig('./model_accuracy.jpg')

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training'], loc='upper right')
    plt.savefig('./model_loss.jpg')

# MAIN ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    # Parameter control
    parser = argparse.ArgumentParser(description='Executable to train and test MIL-CNN Covid-19 detection.')

    parser.add_argument('-b', dest='batch_size',help='Size of the batch', required=False,default=16)
    parser.add_argument('-c', dest='modelCheckpoint', help='Save the state of the model when its loss improves', required=False)
    parser.add_argument('-e', dest='epochs',help='Number of epochs', required=False,default=15)
    parser.add_argument('-o', dest='file_output',help='Directory name to save results', required=False,default=None)
    parser.add_argument('-v','--vertical', dest='number_vert',help='number of vertical cuts', required=False,default=2)
    parser.add_argument('-n','--horizontal', dest='number_hor',help='number of horizontal  cuts', required=False,default=2)
    parser.add_argument('-g',dest='graph', help='Save train historical graphs',action='store_true')

    args = parser.parse_args()

    # IMAGES GENERATION ----
    train_path = './dataset/train'
    test_path = './dataset/test'

    train_gen = ImageDataGenerator(rescale=1./255.)
    test_gen = ImageDataGenerator(rescale=1./255.)

    if int(args.batch_size) < 0:
        print("[ERROR] Batch size must to be greater than 0. Setting to default value (16)...")
        batchSize = 16
    else:
        batchSize = int(args.batch_size)

    gtrain = train_gen.flow_from_directory(
        train_path,
        target_size=(224,224),
        batch_size=1, # Size 1 to apply segmentation one by one.
        class_mode='categorical'
    )

    gtest = test_gen.flow_from_directory(
        test_path,
        target_size=(224,224),
        batch_size=1, # Size 1 to apply segmentation one by one.
        class_mode='categorical'
    )

    # DATASET GENERATION ----
    if int(args.number_vert) < 0:
        print("[ERROR] Number of vertical slices must to be greater than 0. Setting to default value (2)...")
        n_ver = 2
    else:
        n_ver = int(args.number_vert)
    
    if int(args.number_hor) < 0:
        print("[ERROR] Number of horizontal slices must to be greater than 0. Setting to default value (2)...")
        n_hor = 2
    else:
        n_hor = int(args.number_hor)

    n_seg = n_ver*n_hor
    seg_shape=(224//n_hor,224//n_ver,3)

    x_train,y_train = generate_data(n_ver,n_hor,gtrain)
    x_test,y_test = generate_data(n_ver,n_hor,gtest)

    # CREACIÓN DEL MODELO ----
    model_list = []
    for i in range(n_seg):
        model_list.append(create_cnn(seg_shape))

    inputs_list = [m.input for m in model_list]
    outputs_list = [m.output for m in model_list]
    conc_out = tf.keras.layers.concatenate(outputs_list)
    dense = tf.keras.layers.Dense(128,activation='relu')(conc_out)
    out = tf.keras.layers.Dense(2,activation='softmax',name='output_layer')(dense)

    modelo = tf.keras.Model(inputs_list,out)

    # MODEL TRAINING ----
    if int(args.epochs) < 0:
        print("[ERROR] Number of epochs must to be greater than 0. Setting to default value (15)...")
        nEpochs = 15
    else:
        nEpochs = int(args.epochs)

    lrate = 0.001
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

    modelo.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])
    #modelo.summary()
    history = modelo.fit(
        x_train,
        y_train,
        epochs=nEpochs,
        batch_size=batchSize,
        steps_per_epoch=len(x_train[0])//batchSize,
        callbacks=callbacks
    )

    if args.modelCheckpoint:
        modelo.load_weights(checkpointdir)

    # VIEW MODEL TRAINING HISTORY ----
    if args.graph:
        history_graph(history)

    # MODEL TEST ----
    print("Nº datos test: ", gtest.n)
    prediccion = modelo.predict(x_test)
    y_pred = np.argmax(prediccion,axis=1)
    y_test = [0 if np.argmax(i)==0 else 1 for i in y_test]

    # REPORT ----
    report(history,y_test,y_pred,args.file_output)

if __name__ == "__main__":
    main()