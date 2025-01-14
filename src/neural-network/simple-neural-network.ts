import { sigmoid, derivativeOfSigmoid } from './sigmoid';
import meanSquaredErrorLoss from './mse';

function generateNormalRandomValue(mean = 0, standardDeviation = 1) {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const normalRandom =
    Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return normalRandom * standardDeviation + mean;
}

class SimpleNeuralNetwork {
  private weight1: number;
  private weight2: number;
  private weight3: number;
  private weight4: number;
  private weight5: number;
  private weight6: number;

  private bias1: number;
  private bias2: number;
  private bias3: number;

  constructor() {
    this.weight1 = generateNormalRandomValue();
    this.weight2 = generateNormalRandomValue();
    this.weight3 = generateNormalRandomValue();
    this.weight4 = generateNormalRandomValue();
    this.weight5 = generateNormalRandomValue();
    this.weight6 = generateNormalRandomValue();

    this.bias1 = generateNormalRandomValue();
    this.bias2 = generateNormalRandomValue();
    this.bias3 = generateNormalRandomValue();
  }

  feedforward(input: number[]): number {
    const hidden1 = sigmoid(
      this.weight1 * input[0] + this.weight2 * input[1] + this.bias1,
    );
    const hidden2 = sigmoid(
      this.weight3 * input[0] + this.weight4 * input[1] + this.bias2,
    );
    const output = sigmoid(
      this.weight5 * hidden1 + this.weight6 * hidden2 + this.bias3,
    );
    return output;
  }

  train(data: number[][], target: number[]) {
    const learnRate = 0.1;
    const epochs = 1000;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < data.length; i++) {
        const x = data[i];
        const y = target[i];

        const sumForHidden1 =
          this.weight1 * x[0] + this.weight2 * x[1] + this.bias1;
        const hidden1 = sigmoid(sumForHidden1);

        const sumForHidden2 =
          this.weight3 * x[0] + this.weight4 * x[1] + this.bias2;
        const hidden2 = sigmoid(sumForHidden2);

        const sumForOutput =
          this.weight5 * hidden1 + this.weight6 * hidden2 + this.bias3;
        const output = sigmoid(sumForOutput);

        const dL_dtarget = 2 * (output - y);

        // Neuron Output 1
        const dout_d5 = hidden1 * derivativeOfSigmoid(sumForOutput);
        const dout_d6 = hidden2 * derivativeOfSigmoid(sumForOutput);
        const dout_db3 = derivativeOfSigmoid(sumForOutput);

        const dout_dh1 = this.weight5 * derivativeOfSigmoid(sumForOutput);
        const dout_dh2 = this.weight6 * derivativeOfSigmoid(sumForOutput);

        // Neuron Hidden 1
        const dh1_dw1 = x[0] * derivativeOfSigmoid(sumForHidden1);
        const dh1_dw2 = x[1] * derivativeOfSigmoid(sumForHidden1);
        const dh1_db1 = derivativeOfSigmoid(sumForHidden1);

        // Neuron Hidden 2
        const dh2_dw3 = x[0] * derivativeOfSigmoid(sumForHidden2);
        const dh2_dw4 = x[1] * derivativeOfSigmoid(sumForHidden2);
        const dh2_db2 = derivativeOfSigmoid(sumForHidden2);

        // Update Weights and Biases
        // Neuron Hidden 1
        this.weight1 -= learnRate * dL_dtarget * dout_dh1 * dh1_dw1;
        this.weight2 -= learnRate * dL_dtarget * dout_dh1 * dh1_dw2;
        this.bias1 -= learnRate * dL_dtarget * dout_dh1 * dh1_db1;

        // Neuron Hidden 2
        this.weight3 -= learnRate * dL_dtarget * dout_dh2 * dh2_dw3;
        this.weight4 -= learnRate * dL_dtarget * dout_dh2 * dh2_dw4;
        this.bias2 -= learnRate * dL_dtarget * dout_dh2 * dh2_db2;

        // Neuron Output 1
        this.weight5 -= learnRate * dL_dtarget * dout_d5;
        this.weight6 -= learnRate * dL_dtarget * dout_d6;
        this.bias3 -= learnRate * dL_dtarget * dout_db3;
      }

      if (epoch % 10 === 0) {
        const predictedValues = data.map(d => this.feedforward(d));
        const loss = meanSquaredErrorLoss(target, predictedValues);
        console.log(`Epoch: ${epoch}, Loss: ${loss}`);
      }
    }
  }
}

export default SimpleNeuralNetwork;
