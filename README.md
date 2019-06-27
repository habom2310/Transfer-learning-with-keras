Link Colab: https://colab.research.google.com/drive/1--X4u7fylUy_dG51iYTU6KoLNb6E9KsF

# Setting up Google Colab

Connect to Google drive to save / load dataset, models ...

```
from google.colab import drive
drive.mount('/content/drive')
```
Create folder, download data, extract.
```
import os
os.chdir("drive/My Drive/")
!mkdir transfer_learning_with_keras
os.chdir("transfer_learning_with_keras")
```
```
#download data and extract
!wget http://download.tensorflow.org/example_images/flower_photos.tgz
!mkdir tf_file
!tar -xvf flower_photos.tgz -C tf_file
```
# Preparing model

Use mobile pretrained model trained on the imagenet dataset. For other pretrained model, see https://keras.io/applications/

```
#import needed packages
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,SeparableConv2D,BatchNormalization, Activation, Dense
from keras.applications.mobilenet import MobileNet
from keras.optimizers import Adam
```
Dataset consisting of 5 classes: roses, sunflowers, tulips, daisy and dandelion. When loading pretrained model, set include_top = Falseso basic model will not Contain FC layers, weights = 'imagenet'to load pretrained model trained on the IMAGEnet dataset.

After loading, ready for use, add some extra layers. The last FC layer's shape is the number of classes, activation funtion is softmax.
```
# dataset has 5 classes
num_class = 5

# Base model without Fully connected Layers
base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224,224,3))
x=base_model.output
# Add some new Fully connected layers to 
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x = Dropout(0.25)(x)
x=Dense(512,activation='relu')(x) 
x = Dropout(0.25)(x)
preds=Dense(num_class, activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
```
Check the model.
```
model.summary()
```

```
...
conv_pw_13 (Conv2D)          (None, 7, 7, 1024)        1048576   
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
global_average_pooling2d_3 ( (None, 1024)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_7 (Dense)              (None, 512)               524800    
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 5)                 2565      
=================================================================
Total params: 4,805,829
Trainable params: 4,783,941
Non-trainable params: 21,888
``` 

```
for i,layer in enumerate(model.layers):
  print("{}: {}".format(i,layer))
```
```
...
82: <keras.layers.normalization.BatchNormalization object at 0x7f5910480dd8>
83: <keras.layers.advanced_activations.ReLU object at 0x7f5910423a90>
84: <keras.layers.convolutional.Conv2D object at 0x7f59103eb7b8>
85: <keras.layers.normalization.BatchNormalization object at 0x7f5910380c18>
86: <keras.layers.advanced_activations.ReLU object at 0x7f59103b3908>
87: <keras.layers.pooling.GlobalAveragePooling2D object at 0x7f5911abbdd8>
88: <keras.layers.core.Dense object at 0x7f5911abbd30>
89: <keras.layers.core.Dropout object at 0x7f5910298748>
90: <keras.layers.core.Dense object at 0x7f590de4d630>
91: <keras.layers.core.Dropout object at 0x7f590de1d278>
92: <keras.layers.core.Dense object at 0x7f590deeaf60>
```
Before layer 87 are base layers of Mobilenet model. Only set trainable = True for layers after layer 87.
```
for layer in model.layers[:87]:
    layer.trainable=False
for layer in model.layers[87:]:
    layer.trainable=True
```
# Preparing data

Use ImageDataGeneratorof keras. Split train-val ratio is 75-25
```
train_datagen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                 validation_split=0.25)

train_generator=train_datagen.flow_from_directory('tf_file/flower_photos/',
                                                 target_size=(224,224),
                                                 batch_size=64,
                                                 class_mode='categorical',
                                                 subset='training')


validation_generator = train_datagen.flow_from_directory(
                                                'tf_file/flower_photos/', # same directory as training data
                                                target_size=(224,224),
                                                batch_size=64,
                                                class_mode='categorical',
                                                subset='validation') # set as validation data
```

```
Found 2755 images belonging to 5 classes.
Found 915 images belonging to 5 classes.
```
# Training
Set hyper parameter.
```
epochs = 50
learning_rate = 0.0005
decay_rate = learning_rate / epochs
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
```
Callback set for saving model and tensorboard
```
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

!mkdir ckpt
!mkdir logs

filepath="ckpt/best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only = False, save_best_only=True, mode='min')
logdir="logs/mobilenet"
tfboard = TensorBoard(log_dir=logdir)

callbacks_list = [checkpoint, tfboard]
```
Train
```
step_size_train = train_generator.n/train_generator.batch_size
step_size_val = validation_generator.samples // validation_generator.batch_size
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator, 
                   validation_steps =step_size_val,
                   callbacks = callbacks_list,
                   epochs=10)
```
```
...

Epoch 00006: val_acc did not improve from 0.80747
Epoch 7/10
87/86 [==============================] - 15s 170ms/step - loss: 0.1841 - acc: 0.9432 - val_loss: 0.6885 - val_acc: 0.8279

Epoch 00007: val_acc improved from 0.80747 to 0.82786, saving model to ckpt/best.hdf5
Epoch 8/10
87/86 [==============================] - 15s 172ms/step - loss: 0.2004 - acc: 0.9317 - val_loss: 0.7222 - val_acc: 0.8075

Epoch 00008: val_acc did not improve from 0.82786
Epoch 9/10
87/86 [==============================] - 15s 171ms/step - loss: 0.1713 - acc: 0.9379 - val_loss: 0.8382 - val_acc: 0.7916

Epoch 00009: val_acc did not improve from 0.82786
Epoch 10/10
87/86 [==============================] - 15s 170ms/step - loss: 0.1288 - acc: 0.9535 - val_loss: 1.0021 - val_acc: 0.7678

Epoch 00010: val_acc did not improve from 0.82786
```
The best model will be save based on the val_acc. In this sample, the best one has 82.78% of accuracy in val set

# Inference
Load the best model for inference.
```
inf_model = keras.models.load_model("ckpt/best.hdf5")
```
Preprocess input image.
```
def preprocess_image(img):
        if (img.shape[0] != 224 or img.shape[1] != 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = (img/127.5)
        img = img - 1
        img = np.expand_dims(img, axis=0)
        return img
```

```
classes = train_generator.class_indices
classes = list(classes.keys())
```
We take an image from dataset to test. It is recommended to have an independent test set.
```
import glob
files = glob.glob("tf_file/flower_photos/dandelion/*.jpg") # lấy ảnh trong folder dandelion để test
```
inference
```
img = cv2.imread(files[0])
pred = inf_model.predict(preprocess_image(img))
result = classes[np.argmax(pred)]
print(result)                     
```
```
'dandelion'
```
# Improvements
Here, overfit is happening when val_loss is not decrease while train_loss is still decrease. It can be seen easily in Tensorboard.

To fix this problem, there are some solutions:
- Instead of freezing all the base model (from the layer 87) we can freeze from e.g layer 70 or 75. It depends on actual datasets and you need to choose the best one.
- Add augmentation steps in `ImageDataGenerator`, see more details in https://keras.io/preprocessing/image/.
- Change hyper parameters.

## Bonus 
Freeze keras model (.hdf5) to make tensorflow model (.pb)
```
K.set_learning_phase(0)

print(model.outputs)
print(model.inputs)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
                              
tf.train.write_graph(frozen_graph, "model", "model.pb", as_text=False)
```
