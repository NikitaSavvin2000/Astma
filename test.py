import numpy as np
import matplotlib.pyplot as plt

# Создание набора данных для распределения Гаусса
mean = 0 # Среднее значение
std_dev = 1 # Стандартное отклонение
data = np.random.normal(mean, std_dev, 10000)

# Создание гистограммы
plt.hist(data, bins=50, density=True, alpha=0.5, color='b')

# Создание линии распределения
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = 1/(std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * std_dev**2))
plt.plot(x, y, color='r')

# Добавление названий осей и заголовка
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.title('Распределение Гаусса')

# Отображение графика
plt.show()
