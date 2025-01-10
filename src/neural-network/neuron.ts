import { sigmoid } from './sigmoid';

class Neuron {
  public activationFunction: 'sigmoid';

  constructor(
    public weights: number[],
    public bias: number,
  ) {
    this.activationFunction = 'sigmoid';
  }

  public feedForward(inputs: number[]): number {
    const weightsInputsDotProduct = this.weights.reduce(
      (prev, weight, index) => prev + weight * inputs[index],
      0,
    );
    return sigmoid(weightsInputsDotProduct + this.bias);
  }
}

export default Neuron;
