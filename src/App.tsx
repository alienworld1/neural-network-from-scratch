import Neuron from './neural-network/neuron';
import NeuronVisualizer from './components/NeuronVisualizer';

export default function App() {
  const neuron = new Neuron([0, 1], 4);

  return (
    <div>
      <NeuronVisualizer neuron={neuron} />
    </div>
  );
}
