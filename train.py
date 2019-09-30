from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras import backend as K
from models.model import get_net
import tensorflow
from training import load_data, VolumeDataGenerator

if __name__ == "__main__":

    batch_size = 6
    epochs = 1000
    logs_folder = "data/tf_logs/"

    training_path = "data/training/training-set"
    validation_path = "data/validation/validation-set"

    x_train, y_train = load_data(training_path, nb_examples=20)
    x_validation, y_validation = load_data(validation_path, nb_examples = 20)

    print("Loaded Data")

    datagen = VolumeDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        depth_flip=False,
        min_max_normalization=True,
        scale_range=0.1,
        scale_constant_range=0.5
    )

    train_generator = datagen.flow(x_train, y_train, batch_size)
    validation_generator = datagen.flow(x_validation, y_validation, batch_size)

    now = datetime.now()
    logdir = logs_folder + now.strftime("%B-%d-%Y-%I:%M%p") + "/"

    tboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)

    current_checkpoint = ModelCheckpoint(filepath='data/modelweights/current_weights_checkpoint.hdf5', verbose=1)
    period_checkpoint = ModelCheckpoint('data/modelweights/weights{epoch:08d}.hdf5', period=5)
    best_weight_checkpoint = ModelCheckpoint(filepath='data/modelweights/best_weights_checkpoint.hdf5', verbose=1, save_best_only=True)

    conf = tensorflow.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
    K.set_session(tensorflow.Session(config=conf))

    model = get_net()

    model.fit_generator(train_generator,
                        steps_per_epoch=1200,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=300,
                        use_multiprocessing=False,
                        workers=1,
                        callbacks=[tboard, current_checkpoint, best_weight_checkpoint, period_checkpoint],
                        verbose=1)

    model_name = 'model_' + now.strftime("%B-%d-%Y-%I:%M%p")