#Code for Paper  "The reconstruction of flows from spatiotemporal data by autoencoders"

# Facundo Fainstein (1,2), Josefina Catoni (1), Coen Elemans (3) and Gabriel B. Mindlin (1,2,4)* 

# (1) Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Física, Ciudad Universitaria, 1428 Buenos Aires, Argentina.

# (2) CONICET - Universidad de Buenos Aires, Instituto de Física Interdisciplinaria y Aplicada (INFINA), Ciudad Universitaria, 1428 Buenos Aires, Argentina.

# (3) Department of Biology, University of Southern Denmark, 5230 Odense M, Denmark.

# (4) Universidad Rey Juan Carlos, Departamento de Matemática Aplicada, Madrid, Spain. 

# *Gabriel B. Mindlin (corresponding author)
# Email: gabo@df.uba.ar


#Code for: train autoencoders with varying number of units in the latent layer
# Used for the synthetic movie
#%%
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers
import time

from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras import backend as K
import tensorflow as tf

def set_gpu_option(which_gpu, fraction_memory):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    config.gpu_options.visible_device_list = which_gpu
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
    return

set_gpu_option("0", 0.9)
#%%Import synthetic movie
#Movie folder
root_dir = '.../'

x_data = np.load(root_dir+'synthetic_movie_lorenz63.npy').astype('float32')
xmean_data=np.mean(x_data,axis=0) 

#Substract the mean 
x_data -= xmean_data

#Define data 
X = x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2])

#Split in train and test
X_train, X_test = X[0:30000], X[30000:]
print("Train data set length:", len(X_train))
print("Test data set length:", len(X_test))

#Define latent space dimensions
dim_LSs = [1,2,4,5]

#define number of training sessions for each architecture
N_trains = 80
numero_pixeles = X.shape[1]

#Define initialization hyperparameters
mean_init = 0.0
stdde_init = 0.4

#define batch size, learning rate and number of epochs
batch_size_net = 512
learning_rate_net = 0.001
num_epochs = 600

#%% Train networks
for dim_LS in dim_LSs:     
    
    #Define path to saving directory
    saving_dir = '.../dimLS'+str(dim_LS)+"/"
    
    #Measure time
    start = time.time()
    
    for ntrain in range(N_trains):
        
        #Create net
        input_img = keras.Input(shape=(numero_pixeles,))
        encoded = layers.Dense(64, activation='relu',
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(input_img)
        encoded = layers.Dense(32, activation='relu', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(encoded)
        encoded = layers.Dense(16, activation='relu', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(encoded)
        encoded = layers.Dense(dim_LS, activation='linear', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(encoded)
        decoded = layers.Dense(16, activation='relu', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(encoded)
        decoded = layers.Dense(32, activation='relu', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(decoded)
        decoded = layers.Dense(64, activation='relu', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(decoded)
        decoded = layers.Dense(numero_pixeles, activation='linear', 
                               kernel_initializer=initializers.RandomNormal(mean=mean_init, 
                                                                            stddev=stdde_init))(decoded)#Create the autoencoder
        autoencoder = keras.Model(input_img, decoded)
    
        #Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_net)
        autoencoder.compile(loss='mse', metrics=['mean_absolute_error'], optimizer=optimizer)
        
        filepath_best=saving_dir+"{}-checkpoint-best.hdf5".format(ntrain)
        checkpoint_best = ModelCheckpoint(filepath_best, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint_best]

        # training the model and saving metrics in history
        history = autoencoder.fit(X_train, X_train,
                  batch_size=batch_size_net, 
                  epochs=num_epochs,
                  validation_data=(X_test, X_test),
                  callbacks=callbacks_list,
                  verbose=2, 
                  shuffle=True);
    
        train_mse=np.array(history.history['loss'])
        val_mse=np.array(history.history['val_loss'])
        np.save(saving_dir+"{}-train_mse.npy".format(ntrain),train_mse)
        np.save(saving_dir+"{}-val_mse.npy".format(ntrain),val_mse)
    
    
    end = time.time()
    print("The time of execution is :",
          np.round((end-start) / 60, 2), "min")
    