
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

const xs = tf.tensor1d([1, 2, 3, 4]);
const ys = tf.tensor1d([1, 3, 5, 7]);


async function saveModel() {
    await model.fit(xs, ys, { epochs: 5000 }).then(() => {
        const output = model.predict(tf.tensor2d([5], [1, 1]))
        console.log(Math.round(Array.from(output.dataSync())[0]))
    });
    const data = await model.save('file://./model')
    return data
}

saveModel()