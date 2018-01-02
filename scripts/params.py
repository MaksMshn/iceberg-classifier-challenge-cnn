#model
nb_filters = 16
nb_dense = 128
#train
batch_size = 64
epochs = 50
#weights
weights_init = '../weights/weights_init.hdf5'
weights_file = '../weights/weights_current.hdf5'
#test
batch_size_test = batch_size
validate_before_test = True
