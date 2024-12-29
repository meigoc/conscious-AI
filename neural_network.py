from neuron import Neuron
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.hidden_layer1 = [Neuron(np.random.rand(2), np.random.rand(1)[0]) for _ in range(5000)]
        self.hidden_layer2 = [Neuron(np.random.rand(5000), np.random.rand(1)[0]) for _ in range(5000)]
        self.hidden_layer3 = [Neuron(np.random.rand(5000), np.random.rand(1)[0]) for _ in range(5000)]
        self.output_layer = [Neuron(np.random.rand(5000), np.random.rand(1)[0]) for _ in range(1)]

    def feedforward(self, inputs):
        hidden_outputs1 = [neuron.feedforward(inputs) for neuron in self.hidden_layer1]
        hidden_outputs2 = [neuron.feedforward(hidden_outputs1) for neuron in self.hidden_layer2]
        hidden_outputs3 = [neuron.feedforward(hidden_outputs2) for neuron in self.hidden_layer3]
        final_output = [neuron.feedforward(hidden_outputs3) for neuron in self.output_layer]
        return final_output

    def train(self, data, targets, learning_rate, epochs):
        m = len(data)
        v_dw = [np.zeros_like(neuron.weights) for neuron in self.hidden_layer1 + self.hidden_layer2 + self.hidden_layer3 + self.output_layer]
        v_db = [0 for _ in v_dw]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        t = 0
        
        for _ in range(epochs):
            for inputs, target in zip(data, targets):
                t += 1
                # Прямое распространение
                hidden_outputs1 = [neuron.feedforward(inputs) for neuron in self.hidden_layer1]
                hidden_outputs2 = [neuron.feedforward(hidden_outputs1) for neuron in self.hidden_layer2]
                hidden_outputs3 = [neuron.feedforward(hidden_outputs2) for neuron in self.hidden_layer3]
                final_output = [neuron.feedforward(hidden_outputs3) for neuron in self.output_layer][0]

                # Обратное распространение ошибок для выходного слоя
                output_error = target - final_output

                for neuron in self.output_layer:
                    for i in range(len(neuron.weights)):
                        grad_w = output_error * hidden_outputs3[i]
                        v_dw[i] = beta1 * v_dw[i] + (1 - beta1) * grad_w
                        v_db[i] = beta1 * v_db[i] + (1 - beta1) * output_error
                        neuron.weights[i] += learning_rate * v_dw[i] / (1 - beta1 ** t)
                    neuron.bias += learning_rate * v_db[i] / (1 - beta1 ** t)

                # Обратное распространение ошибок для третьего скрытого слоя
                hidden_errors3 = [0] * len(self.hidden_layer3)
                for i in range(len(self.hidden_layer3)):
                    hidden_errors3[i] = output_error * self.output_layer[0].weights[i]

                for neuron, error in zip(self.hidden_layer3, hidden_errors3):
                    for i in range(len(neuron.weights)):
                        grad_w = error * hidden_outputs2[i]
                        v_dw[i] = beta1 * v_dw[i] + (1 - beta1) * grad_w
                        v_db[i] = beta1 * v_db[i] + (1 - beta1) * error
                        neuron.weights[i] += learning_rate * v_dw[i] / (1 - beta1 ** t)
                    neuron.bias += learning_rate * v_db[i] / (1 - beta1 ** t)

                # Обратное распространение ошибок для второго скрытого слоя
                hidden_errors2 = [0] * len(self.hidden_layer2)
                for i in range(len(self.hidden_layer2)):
                    hidden_errors2[i] = sum(hidden_errors3[j] * self.hidden_layer3[j].weights[i] for j in range(len(self.hidden_layer3)))

                for neuron, error in zip(self.hidden_layer2, hidden_errors2):
                    for i in range(len(neuron.weights)):
                        grad_w = error * hidden_outputs1[i]
                        v_dw[i] = beta1 * v_dw[i] + (1 - beta1) * grad_w
                        v_db[i] = beta1 * v_db[i] + (1 - beta1) * error
                        neuron.weights[i] += learning_rate * v_dw[i] / (1 - beta1 ** t)
                    neuron.bias += learning_rate * v_db[i] / (1 - beta1 ** t)

                # Обратное распространение ошибок для первого скрытого слоя
                hidden_errors1 = [0] * len(self.hidden_layer1)
                for i in range(len(self.hidden_layer1)):
                    hidden_errors1[i] = sum(hidden_errors2[j] * self.hidden_layer2[j].weights[i] for j in range(len(self.hidden_layer2)))

                for neuron, error in zip(self.hidden_layer1, hidden_errors1):
                    for i in range(len(neuron.weights)):
                        grad_w = error * inputs[i]
                        v_dw[i] = beta1 * v_dw[i] + (1 - beta1) * grad_w
                        v_db[i] = beta1 * v_db[i] + (1 - beta1) * error
                        neuron.weights[i] += learning_rate * v_dw[i] / (1 - beta1 ** t)
                    neuron.bias += learning_rate * v_db[i] / (1 - beta1 ** t)

# Генерация более реалистичных тренировочных данных для умножения
def generate_training_data(samples):
    data = []
    targets = []
    for _ in range(samples):
        x = np.random.rand(2) * 10  # Диапазон чисел от 0 до 10
        y = x[0] * x[1]
        data.append(x)
        targets.append(y)
    return np.array(data), np.array(targets)

if __name__ == "__main__":
    # Создание и тренировка нейронной сети
    nn = NeuralNetwork()
    training_data, training_targets = generate_training_data(100000)
    nn.train(training_data, training_targets, learning_rate=0.001, epochs=1000)

    # Тестирование сети на новых данных
    test_data, test_targets = generate_training_data(10)
    for inputs, target in zip(test_data, test_targets):
        output = nn.feedforward(inputs)
        print(f"Input: {inputs}, Predicted Output: {output[0]}, Actual Output: {target}")
