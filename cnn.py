from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.preprocessing import image 
from keras.models import load_model
import numpy as np
import cv2
import imutils 
from imutils.video import VideoStream

image_name=[]

def buildModel():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(38))
    model.add(Activation('softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def train(model):
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'categorical')
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    
    model.fit_generator(training_set,
                             steps_per_epoch = (5349),
                             epochs = 40,
                             validation_data = test_set,
                             validation_steps = (1089))
    model.save('my_model.h5')
    
def loadImageNameAndModelWeight():
    file = open("name_list.txt", "r") 
    for line in file: 
        image_name.append(line)
    del(line)
    return load_model('my_model.h5')

def webcamTest(model):
    cap = VideoStream(src=0).start()
    while (True):
        original = cap.read()
        img = imutils.resize(original, width=400)
        img = cv2.resize(original, (64, 64))  
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
           
        name_num=list(model.predict_proba(img)[0])
        max_value=max(name_num)
        max_index=name_num.index(max_value)
        if max_value>0.80 :
            name=image_name[max_index]
    
        cv2.putText(original, "Label: {0}".format(name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Classification", original)
        cv2.waitKey(5)
        if (cv2.waitKey(5) & 0xFF == ord('q')):
            break;
    
    cap.release()
    cv2.destroyAllWindows()

def  main():
    model= buildModel()
    #train(model)
    model=loadImageNameAndModelWeight()
    webcamTest(model)

if __name__ =='__main__' :
    main()