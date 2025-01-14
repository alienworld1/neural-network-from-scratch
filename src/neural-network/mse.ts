export default function meanSquaredErrorLoss(predicted: number[], actual: number[]): number {
  return predicted.reduce((sum, predictedValue, index) => {
    const actualValue = actual[index];
    return sum + (predictedValue - actualValue) ** 2;
  }, 0) / predicted.length;
}