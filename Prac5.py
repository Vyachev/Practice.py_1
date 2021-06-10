#подключение нужных библиотек
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#функция, считывающая данные из входного файла и преобразование данных во временные ряды
def read_data(input_file):
  input_data = np.loadtxt(input_file, delimiter = None)
  dates = pd.date_range('1950-01', periods = input_data.shape[0], freq = 'M')
  output = pd.Series(input_data[:, index], index = dates)
  return output

#Путь к входному файлу и преобразование столбца в формат временных рядов
if __name__=='__main__':
  input_file = "/Users\1\Desktop\5 пр"
  timeseries = read_data(input_file)

#Визуализация данных на графиках
plt.figure()
timeseries.plot()
plt.show()

#Разбитие данных только с 1980 по 1990 год на графике
timeseries['1980':'1990'].plot()
<matplotlib.axes._subplots.AxesSubplot at 0xa0e4b00>
plt.show()


#Функция mean() для нахождения среднего значения
timeseries.mean()

#Функция max() для нахождения максимального значения
timeseries.max()

#Функцияю min(), для нахождения минимального значения
timeseries.min()

#Повторная выборка данных с использованием метода mean()
timeseries_mm = timeseries.resample("A").mean()
timeseries_mm.plot(style = 'g--')
plt.show()

#Повторная выборка данных с использованием метода median()
timeseries_mm = timeseries.resample("A").median()
timeseries_mm.plot()
plt.show()

#Расчет скользящего среднего значения
timeseries.rolling(window = 12, center = False).mean().plot(style = '-g')
plt.show()