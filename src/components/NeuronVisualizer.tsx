import { useState } from 'react';
import Neuron from '../neural-network/neuron';

export default function NeuronVisualizer({ neuron }: { neuron: Neuron }) {
  const [inputs, setInputs] = useState(Array(neuron.weights.length).fill(0));

  return (
    <div className="border border-gray-600 rounded-md flex w-fit p-2 gap-2">
      <div className="p-2 flex flex-col border-2 border-black rounded-md gap-2">
        {neuron.weights.map((_, index) => (
          <input
            key={index}
            type="number"
            className="w-12 text-lg border border-gray-600 rounded-md"
            onChange={event => {
              const newInputs = [...inputs];
              newInputs[index] = parseFloat(event.target.value);
              if (isNaN(newInputs[index])) {
                newInputs[index] = 0;
              }
              setInputs(newInputs);
            }}
            value={inputs[index]}
          />
        ))}
      </div>
      <ul className="flex flex-col bg-red-500 p-2 rounded-md justify-center">
        {neuron.weights.map((weight, index) => (
          <li key={index} className="text-white">
            {weight}
          </li>
        ))}
      </ul>
      <div className="flex justify-center items-center font-semibold text-white bg-emerald-700 p-2 rounded-md">
        {neuron.bias}
      </div>
      <div className="flex justify-center items-center font-semibold bg-yellow-400 p-2 rounded-md">
        {neuron.feedForward(inputs)}
      </div>
    </div>
  );
}
