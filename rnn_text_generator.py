import numpy as np
import random

# Расширенный набор предложений для обучения
training_sentences = [
    "Дерево растет в лесу.",
    "Дерево имеет зеленые листья.",
    "Дерево дает тень.",
    "Осенью листья падают с дерева.",
    "Дерево - дом для птиц.",
    "Дерево может быть очень высоким.",
    "Ветви дерева покачиваются на ветру.",
    "Дерево цветет весной.",
    "Дерево можно использовать для строительства.",
    "Плоды дерева съедобны.",
    "Кора дерева защищает его от вредителей.",
    "Корни дерева уходят глубоко в землю.",
    "Дерево очищает воздух.",
    "Дерево предоставляет пищу и убежище многим животным.",
    "Дерево может жить сотни лет.",
    "Дерево имеет множество видов и форм.",
    "Дерево - важный элемент экосистемы.",
    "Дерево используется для производства бумаги.",
    "Дерево растет медленно, но уверенно.",
    "Дерево является символом жизни и роста.",
    "В лесу много разных деревьев.",
    "Дерево является символом силы.",
    "Зимой дерево покрыто снегом.",
    "Летом дерево дает прохладу.",
    "Дерево является источником кислорода.",
    "Птицы поют на ветвях дерева.",
    "Деревья создают уютную тень.",
    "Деревья помогают сохранять баланс экосистемы.",
    "Листья дерева шуршат на ветру.",
    "Дерево живет долгое время.",
    "Дерево приносит плоды каждое лето.",
    "Дерево является домом для многих существ.",
    "Дерево является символом мудрости.",
    "Дерево использовалось для создания древних инструментов.",
    "Деревья играют важную роль в культуре и мифологии.",
    "Дерево может выжить в суровых условиях.",
    "Листья дерева могут менять цвет осенью.",
    "Деревья помогают предотвратить эрозию почвы.",
    "Дерево является источником вдохновения для художников.",
    "Дерево имеет множество лечебных свойств.",
    "Дерево может быть символом вечности.",
    "Деревья могут расти в различных климатических условиях.",
    "Дерево символизирует гармонию с природой.",
    "Дерево может приносить плоды каждый год.",
    "Дерево является частью наших традиций и праздников.",
    "Дерево является символом стабильности.",
    "Деревья играют важную роль в круговороте углерода.",
    "Деревья предоставляют древесину для строительства.",
    "Дерево - это живая лаборатория природы.",
    "Дерево - это красивая часть ландшафта.",
    "Дерево помогает сохранять биоразнообразие."
]

# Создание словаря символов
def create_char_dict(sentences):
    text = ' '.join(sentences)
    chars = sorted(list(set(text)))
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_index, index_to_char

# Генерация случайного текста
def random_text(length):
    text = ' '.join(training_sentences)
    return ''.join(random.choices(text, k=length))

# Создание словаря символов
char_to_index, index_to_char = create_char_dict(training_sentences)

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
        y = np.exp(y) / np.sum(np.exp(y))  # Нормализация
        return y, h_next

def one_hot_encode(index, size):
    vector = np.zeros((size, 1))
    vector[index][0] = 1
    return vector

def sample(rnn, seed, char_to_index, index_to_char, length=100):
    h_prev = np.zeros((rnn.hidden_size, 1))
    x = one_hot_encode(char_to_index[seed[0]], len(char_to_index))
    result = seed

    for t in range(length):
        y, h_prev = rnn.forward(x, h_prev)
        idx = np.random.choice(range(len(char_to_index)), p=y.ravel())
        char = index_to_char[idx]
        result += char
        x = one_hot_encode(idx, len(char_to_index))

    return result

def train_rnn(rnn, data, char_to_index, epochs=300, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for sentence in data:
            h_prev = np.zeros((rnn.hidden_size, 1))
            for i in range(len(sentence) - 1):
                x = one_hot_encode(char_to_index[sentence[i]], len(char_to_index))
                y_true = one_hot_encode(char_to_index[sentence[i + 1]], len(char_to_index))

                # Прямое распространение
                y_pred, h_prev = rnn.forward(x, h_prev)

                # Вычисление ошибки
                loss = -np.sum(y_true * np.log(y_pred + 1e-8))  # Добавляем небольшое значение, чтобы избежать log(0)
                total_loss += loss

                # Обратное распространение
                dy = y_pred - y_true
                dwy = np.dot(dy, h_prev.T)
                dby = dy
                dwx = np.dot(np.dot(rnn.wx.T, dy), x.T)
                dwh = np.dot(np.dot(rnn.wh.T, dy), h_prev.T)
                dbh = np.sum(dy, axis=1, keepdims=True)

                # Обновление весов и смещений
                rnn.wx -= learning_rate * dwx
                rnn.wh -= learning_rate * dwh
                rnn.bh -= learning_rate * dbh
                rnn.wy -= learning_rate * dwy
                rnn.by -= learning_rate * dby

        if epoch % 10 == 0:
            print(f"Epoch {epoch} complete with total loss: {total_loss:.4f}")

# Подготовка данных для обучения
training_data = [sentence for sentence in training_sentences]

# Инициализация RNN
rnn = SimpleRNN(input_size=len(char_to_index), hidden_size=128, output_size=len(char_to_index))

# Тренировка RNN
train_rnn(rnn, training_data, char_to_index, epochs=300)

# Пример генерации текста на основе обученной сети
seed = 'Дерево '
generated_text = sample(rnn, seed, char_to_index, index_to_char, length=100)
print(generated_text)
