
# TRAILMAP

DESCRIPTION

## Getting Started

Download this repository with:

```
git clone 
```

or just download the repository directly from github.com

### Prerequisites
Depending on your brain file size the program will take 8-16gb of memory.

You must have python3 installed 

If you would like only to run the network on your brains you only need tensorflow and skimage.

You can use virtualenv or anaconda to manage these dependencies.

You can also install these packages:

```
pip3 install tensorflow
pip3 install opencv-python
```

If you would like to do transfer learning on your own cell types you need to also install these packages
```
pip3 install tensorflow
pip3 install opencv-python
pip3 install numpy
pip3 install PIL
```

### Inference

Note: Your brain volume must be in the form of 2D image slices in a folder where the order
is determined by the alphabetical numerical order of the image file names.
See TRAILMAP/data/testing/example-chunk for an example

To run the network cd into the TRAILMAP directory and run 
```
python3 segment_brain_batch input_folder1 input_folder2 input_folder3 
```

input_folderX is the absolute path to the folder of the brains you would like to process 

The program will create a folder named seg-"INPUT_FOLDER_NAME" in the same directory as the input_folder

To run the example brain do
```
python3 segment_brain_batch PATH-TO-TRAILMAP/data/testing/example-chunk
```


### Training

If you would like to do transfer learning with your own examples, you must label your own data. The network
takes in cubes of length 64 pixels so the training examples must be of this size. The strategy used in our
implementation was to hand label cubes of length ~128px (in the folder training-original) and crop out many 
cubes of length 64px (in the folder training-set).
Note: You may use your own strategy of populating the training-set folder if you wish.

We recommend you to use our strategy. This requires you to crop out some chunks of your own data of
lengths ~128px and put them in the folder training-original/labels and training-original/volumes. The program
will determine a volume's matching label by sorting files in volumes and labels folder alphabetically and assuming
the label and volume at the same index are pairs.

To label the actual chunk you must follow the following legend

You only need to label every 30-60 slices with
* **1** - background
* **2** - axon
* **3** - artifacts (objects that look like axons and should be avoided)
* **4** - the edge of the axon (should be programmatically inserted)

All other slices should be labeled with 
* **0** - unlabeled 


Examples are show here:
TRAINMAP/data/training/training-original/labels/example-label.tif  TRAINMAP/data/training/training-original/volumes/example-volume.tif

After you have placed your labeled examples in training-original folder you can populate
the training-set folder by running
```
python3 generate_training_set num_examples
```
num_examples is an option parameter that determines the numbers of crops to make
in total using a round robin strategy from the training-original folder.
If not specified, the default value set to 100 * NUM_TRAINING_ORIGINAL_EXAMPLES

After generate_training_set has populated the training-set folder, you may start the transfer learning. This will require you to tune the parameters in training.py

There are some default parameters for training, but you will very likely need to tune this depending on how different your own training set is to our data and if you need to do any augmentation.

A VolumeDataGenerator class is provided that handles basic operations (the train.py contains this class and more specific information in the comments). This follows the same paradigm as Tensorflow's ImageDataGenerator.

After you have populated the training-set folder and tuned parameters, start training with:
```
python3 train.py
```
This will load in the current model and start training the model on your own data. Checkpoints are saved to data/modelweights at the end of each epoch.


## Authors

* **Albert Pun**
* **Drew Friedmann**

## License

This project is licensed under the MIT License

## Acknowledgments

* Research sponsored by Liqun Luo's Lab

