export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function derivativeOfSigmoid(x: number): number {
  return sigmoid(x) * (1 - sigmoid(x));
}
