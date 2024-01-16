import matplotlib.pyplot as plt
import os
import pickle
import random
import string
import sys
import tensorflow as tf
import tensorflow.keras as tfk
import notebooks.mototaxi_utils as moto_utils


def getDataSets(img_dir, hyperparams, image_size):
    batch_size = hyperparams['batch_size']
    train_dataset = tfk.preprocessing.image_dataset_from_directory(img_dir,
                                                                   shuffle=True,
                                                                   batch_size=batch_size,
                                                                   image_size=image_size,
                                                                   validation_split=0.3,
                                                                   subset='training',
                                                                   seed=40
                                                                   )
    valtest_dataset = tfk.preprocessing.image_dataset_from_directory(img_dir,
                                                                   shuffle=True,
                                                                   batch_size=batch_size,
                                                                   image_size=image_size,
                                                                   validation_split=0.3,
                                                                   subset='validation',
                                                                   seed=40)

    valtest_len = tf.data.experimental.cardinality(valtest_dataset) #42 batches of size 8 -> ~336 pics
    val_dataset = valtest_dataset.take((2*valtest_len) // 3)  #take the first 2/3 of batches
    test_dataset = valtest_dataset.skip((2*valtest_len) // 3) #take the last 1/3 of the batches

    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def create_model1(image_size):
    # 1. Create an auxiliary model, which does not include the last layer, and keep it frozen.
    model_without_top_layer = tfk.applications.MobileNetV2(input_shape=image_size + (3,),
                                                           include_top=False,
                                                           weights='imagenet')
    model_without_top_layer.trainable = False

    # 2. Handle the input images: augment them and preprocess them
    inputs = tfk.Input(shape=image_size + (3,))
    x = tfk.Sequential([
        tfk.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tfk.layers.experimental.preprocessing.RandomRotation(0.05),
    ])(inputs)
    x = tfk.applications.mobilenet_v2.preprocess_input(x)

    # 3. Pass the input through the auxiliary model
    x = model_without_top_layer(x, training=False)

    # 4. Pass the input through the new classifier layer
    new_classifier = tf.keras.Sequential([
        tfk.layers.GlobalAveragePooling2D(),
        tfk.layers.Dropout(0.2),
        tfk.layers.Dense(units=1)
    ])
    outputs = new_classifier(x)

    model1 = tfk.Model(inputs, outputs)
    return model1


def transfer_learning_case1(img_dir, hyperparams):
    print("Training: Case1 (name={:}, batch_size={:}, num_epochs={:}, learning rate: initial={:})".format(
        hyperparams['name'],
        hyperparams['batch_size'],
        hyperparams['num_epochs'],
        hyperparams['lr_initial']
    ))
    image_size = (224, 224)
    num_epochs = hyperparams['num_epochs']
    lr_initial = hyperparams['lr_initial']

    datasets = getDataSets(img_dir, hyperparams, image_size)
    train_dataset = datasets['train']
    val_dataset = datasets['val']

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    model1 = create_model1(image_size)

    model1.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr_initial),
                   loss=tfk.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy']
                   )

    history = model1.fit(train_dataset,
                         validation_data=val_dataset,
                         epochs=num_epochs)

    return model1, history

def transfer_learning_case2(img_dir, hyperparams, starting_model):
    print("Training: Case2 (name={:}, batch_size={:}, num_epochs={:}, learning rate: initial={:})".format(
        hyperparams['name'],
        hyperparams['batch_size'],
        hyperparams['num_epochs'],
        hyperparams['lr_initial']
    ))
    image_size = (224, 224)
    num_epochs = hyperparams['num_epochs']
    lr_initial = hyperparams['lr_initial']

    datasets = getDataSets(img_dir, hyperparams, image_size)
    train_dataset = datasets['train']
    val_dataset = datasets['val']

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Create model2
    model2 = starting_model
    # Unfreeze the 'model_without_top_layer' part of model1.
    aux_model = model2.layers[4]  # model2 has 4 layers. The 3rd is the mobilenetv2
    aux_model.trainable = True  # aux_model has 154 layers
    # From layer 127th to 154th will be allowed to train
    layer_cutoff = 126  # out of 154 layers will be frozen

    for i, layer in enumerate(aux_model.layers):
        if i < layer_cutoff:
            layer.trainable = False

    model2.compile(optimizer=tfk.optimizers.Adam(learning_rate=lr_initial),
                   loss=tfk.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy']
                   )

    history = model2.fit(train_dataset,
                         validation_data=val_dataset,
                         epochs=20+num_epochs,
                         initial_epoch=20)

    return model2, history


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Arg 'case1' or 'case2' not provided")

    case = sys.argv[1].lower()
    if case not in ['case1', 'case2']:
        sys.exit("Arg 'case1' or 'case2' not provided")

    work_dir = '../models/'
    img_dir = os.path.join(os.path.expanduser('~'), 'Downloads/dldata/mototaxi_training_images/')

    random_suffix = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    model_filename = '01_15_24_{:s}_{:s}.keras'.format(case, random_suffix)

    hyperparams = {}
    hyperparams['name'] = model_filename
    hyperparams_filename = '01_15_24_{:s}_{:s}.hparams'.format(case, random_suffix)

    if case == 'case1':
        hyperparams['batch_size'] = 16
        hyperparams['num_epochs'] = 20
        hyperparams['lr_initial'] = 0.001
        model, history = transfer_learning_case1(img_dir, hyperparams)

    if case == 'case2':
        hyperparams['batch_size'] = 32
        hyperparams['num_epochs'] = 10
        hyperparams['lr_initial'] = 0.0001
        base_model_filename = '01_15_24_case1_kxblpsda.keras'
        base_model = tfk.models.load_model(os.path.join(work_dir, base_model_filename))
        model, history = transfer_learning_case2(img_dir, hyperparams, base_model)

    hyperparams['history'] = history.history
    model.save(os.path.join(work_dir, model_filename))
    with open(os.path.join(work_dir, hyperparams_filename), 'wb') as f:
        pickle.dump(hyperparams, f)

    os.system('mpg123 -q ~/Downloads/beep-05.mp3')
