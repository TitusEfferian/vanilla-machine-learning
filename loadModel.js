const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

async function loadModel() {
    const data = await tf.loadModel('file://./model/model.json')
    return data
}

loadModel().then((result)=>{
    const data = result.predict(tf.tensor2d([7],[1,1]))
    console.log(Math.round(Array.from(data.dataSync())[0]))
})