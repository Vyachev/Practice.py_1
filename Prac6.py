#Классификатор на основе перцептрона

#Импорт нужных библиотек
import matplotlib.pyplot as plt
import neurolab as nl

#Ввод целевых значений
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

#Объявление сети (2 входа и 1 нейрон)
net = nl.net.newp([[0, 1],[0, 1]], 1)

#Тренировка сети
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1)

#Визуализация с помощью графика
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.show()



#Однослойные нейронные сети

#Импорт нужных библиотек
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

#Загрузка данных из файла
input_data = np.loadtxt('/Users/1/Desktop/6_пр/data_6.txt')

#Разделение данных на 2 столбца данных и 2 метки
data = input_data[:, 0:2]
labels = input_data[:, 2:]

#График ввода данных
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data') 
plt.show()

#Определение минимального и максимального значения для каждого измерения
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

#Определение кол-ва нейронов в выходном слое
nn_output_layer = labels.shape[1]

#Определение однослойной нейронной сети
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer)

#Тренировка сети
error = neural_net.train(data, labels, epochs = 200, show = 20, lr = 0.01)

#Визуализация процесса тренировки на графике
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()