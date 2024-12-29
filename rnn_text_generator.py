# rnn_text_generator.py

from neuron import Neuron
import numpy as np
import random
import string

# Генерация случайного текста
def random_text(length):
    return ''.join(random.choices(string.ascii_lowercase + ' ', k=length))

# Создание словаря символов
def create_char_dict(text):
    chars = sorted(list(set(text)))
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for idx, char for idx, char in enumerate(chars)}
    return char_to_index, index_to_char

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.wx = np.random.randn(hidden_size, input_size) * 0.01  # Вес входа
        self.wh = np.random.randn(hidden_size, hidden_size) * 0.01  # Вес скрытого слоя
        self.wy = np.random.randn(output_size, hidden_size) * 0.01  # Вес выхода
        self.bh = np.zeros((hidden_size, 1))  # Смещение скрытого слоя
        self.by = np.zeros((output_size, 1))  # Смещение выхода

    def forward(self, inputs, h_prev):
        h_next = np.tanh(np.dot(self.wx, inputs) + np.dot(self.wh, h_prev) + self.bh)
        y = np.dot(self.wy, h_next) + self.by
        return y, h_next

# Генерация случайного текста
text_data = random_text(1000)
char_to_index, index_to_char = create_char_dict(text_data)

# Пример использования RNN
rnn = SimpleRNN(input_size=len(char_to_index), hidden_size=50, output_size=len(char_to_index))

def one_hot_encode(index, size):
    vector = np.zeros((size, 1))
    vector[index] = 1
    return vector

def sample(rnn, seed, char_to_index, index_to_char, length=100):
    h_prev = np.zeros((rnn.hidden_size, 1))
    x = one_hot_encode(char_to_index[seed[0]], len(char_to_index))
    result = seed

    for t in range(length):
        y, h_prev = rnn.forward(x, h_prev)
        p = np.exp(y) / np.sum(np.exp(y))
        idx = np.random.choice(range(len(char_to_index)), p=p.ravel())
        char = index_to_char[idx]
        result += char
        x = one_hot_encode(idx, len(char_to_index))

    return result

# Пример генерации текста на основе обученной сети
seed = 'дерево'
generated_text = sample(rnn, seed, char_to_index, index_to_char, length=100)
print(generated_text)
