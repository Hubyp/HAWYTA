const tf = require('@tensorflow/tfjs-node');

const createModel = () => {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [224, 224, 3], 
        filters: 32, 
        kernelSize: 3, 
        activation: 'relu', 
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2], 
        strides: 2, 
    }));

    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: 2,
    }));

    model.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: 2,
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 512,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
};

const saveModel = async (model) => {
    try {
        await model.save('file://path_to_save_model'); 
        console.log('Modelul a fost salvat cu succes!');
    } catch (error) {
        console.error('Eroare la salvarea modelului:', error);
    }
};

const model = createModel();
saveModel(model);
