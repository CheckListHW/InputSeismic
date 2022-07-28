from math import sqrt, pi
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from scipy import signal

'''
model - куб акустичсекой жесткости
cube - сейсмический куб
dt - Шаг дискретизации (ms);
points-  длина вейвлета в дискретах (samples)
sign_freq - частота вейвлета (Hz)
nlv - доля шума относительно амплитуды сигнала Риккера
bfreq - Частота дискретизации
sampling_scale - Масштаб дискрета
input_file - существующий шаблон файла SGY
output_file - выходной SGY-файл
cube - записываемый массив
segy - это общепринятый формат сейсмических данных, в котором много параметров,
но главные для нас здесь - координаты, глубина (время) и значение амплитуды в каждой точке.
'''


def save_sgy_model(input_file, output_file, cube):
    # Функция записывает файл по существующему шаблону (из которого берется заполнение заголовков)
    copyfile(input_file, output_file)
    with segyio.open(output_file, "r+") as src:
        for i, k in zip(src.ilines, range(len(src.ilines))):
            src.iline[i] = cube[k].reshape((cube.shape[1] * cube.shape[2]))


def add_noise(cube, dt, points, sign_freq, nlv):
    # функция добавляет случайный шум к входному кубу
    bfreq = 1. / dt
    sampling_scale = bfreq / (sqrt(2.0) * pi * sign_freq)  #
    seismic_noise = cube + (np.random.uniform(-1.0, 1.0, size=cube.shape)
                            * signal.ricker(points, sampling_scale).max() * nlv)
    return seismic_noise


def make_seismic(model, dt, points, sign_freq):
    # Функция выполняет сверточное моделирование на основании входного куба АИ
    bfreq, m, s, f = 1. / dt, model, model.shape, lambda a, b: (a - b) / (a + b)
    sampling_scale, rpp, seis = bfreq / (sqrt(2.0) * pi * sign_freq), np.zeros(s), np.zeros(s)

    for i, j in [(i, j) for i in range(s[0]) for j in range(s[1])]:
        rpp[i, j, :] = [f(m[i, j, k + 1], m[i, j, k]) for k in range(s[2] - 1)] + [rpp[i, j, -1]]
        seis[i, j, :] += np.convolve(rpp[i, j, :], signal.ricker(points, sampling_scale), mode='same')
    return seis


def plot_section_rainbow(ai_cube):
    # Разрез
    plt.figure(figsize=(5, 3), dpi=100)
    plt.imshow(ai_cube[25, :, :].transpose(), cmap='rainbow')  # итоговая синтетическая сейсмика
    plt.xlabel('xline')
    plt.ylabel('sample')
    plt.grid(ls=':', alpha=.5)
    plt.colorbar(shrink=0.4)
    plt.show()


def plot_section_seismic(seismic):
    # Разрез
    plt.figure(figsize=(5, 3), dpi=100)
    plt.imshow(seismic[-25, :, :].transpose(), vmin=-0.2, vmax=0.2,
               cmap='seismic')  # итоговая синтетическая сейсмика
    plt.xlabel('xline')
    plt.ylabel('sample')
    plt.grid(ls=':', alpha=.5)
    plt.colorbar(shrink=0.4)
    plt.show()

    # На горизонтальном срезе видим один из каналов. Разрез или карта (срез)

    plt.figure(figsize=(5, 3), dpi=100)
    plt.imshow(seismic[:, :, 60].transpose(), vmin=-0.2, vmax=0.2,
               cmap='seismic')  # итоговая синтетическая сейсмика
    plt.xlabel('inline')
    plt.ylabel('sample')
    plt.grid(ls=':', alpha=.5)
    plt.colorbar(shrink=0.4)
    plt.show()


def plot_final_seismic_map(seismic, noise_seis):
    plt.figure(figsize=(10, 5), dpi=200)
    plt.subplot(211)
    plt.imshow(seismic[-25, :, :].transpose(), vmin=-0.2, vmax=0.2,
               cmap='seismic')  # итоговая синтетическая сейсмика
    plt.xlabel('inline')
    plt.ylabel('sample')
    plt.grid(ls=':', alpha=.5)
    plt.colorbar(shrink=0.4)
    plt.subplot(212)
    plt.imshow(noise_seis[-25, :, :].transpose(), vmin=-0.2, vmax=0.2,
               cmap='seismic')  # итоговая синтетическая сейсмика
    plt.xlabel('inline')
    plt.ylabel('sample')
    plt.grid(ls=':', alpha=.5)
    plt.colorbar(shrink=0.4)
    plt.show()


def noise_with_plot():
    """
        Загружаем куб акустического импеданса
        (который можно получить умножив куб скорости на куб плотности).
        Обращаем внимание, что перевод формата в segy был произведён в Petrel.
        Основная сложность применения этого кода для OilCase - переход в формат segy в Пайтоне.
    """

    input_file, output_file, output_file_noise, output_file_xlsx = \
        'AI_3channels.sgy', 'f70Hz_noise10.sgy', 'f70Hz_noise5.sgy', 'Clust30Hz_noise10.xlsx'

    ai_cube = segyio.tools.cube(input_file)  # загрузка куба

    """
        Разрез по кросслайну 25 Это пока просто импеданс, палетка обыкновенная геологическая.
        Здесь смотрим на адекватность исходных данных В данном случае - sample - это глубина
    """

    plot_section_rainbow(ai_cube)

    """
        Меняем только частоту (Гц) - последний параметр, остальные не трогаем. 
        Шаг дискретизации 0,002 сек стандартный. Длина вейвлета 45 дискретов параметр технический, 
        его я могу поварьировать на новой модели, но скорее всего не изменится
    """

    seismic = make_seismic(ai_cube, 0.002, 45, 70)

    """
        plot_section_seismic(seismic)
        Пример с уровнем шума 5%
        Предлагаю варьировать шум от 5% для новой сеймики до 30% для старой сейсмики
    """

    # Далее уже цветовая шкала сейсмики. Да, в случае без шума выглядит нереалистично, но полностью соответствует модели
    noise_seismic = add_noise(seismic, 0.002, 15, 30, 0.05)

    """
        Думаю, что в нашей ситуации код ниже не нужен. Оставляю, будет проще разобраться с форматом segy
        Если есть необходмость выгрузки результата в стандартном сейсмическом формате
        И загрузки, например в Petrel или другое ПО
    """

    # Без шума:
    save_sgy_model(input_file, output_file, seismic)
    # C шумом:
    save_sgy_model(input_file, output_file_noise, noise_seismic)

    dat = [noise_seismic[i, j, :] for i in range(100) for j in range(100)]

    clust_data = pd.DataFrame(dat)
    clust_data.head(1)
    clust_data.to_excel(output_file_xlsx)

    plt.figure(figsize=(3, 10), dpi=100)
    plt.gca().invert_yaxis()
    plt.plot(dat[-1], np.arange(0, 100, 1), c='green')
    plt.show()


def make_noise(ai_cube: np.ndarray):
    seismic = make_seismic(ai_cube, 0.002, 45, 70)
    noise_seismic = add_noise(seismic, 0.002, 15, 30, 0.05)
    return noise_seismic


if __name__ == '__main__':
    m_n = make_noise(segyio.tools.cube('AI_3channels.sgy'))
    s0, s1, s2 = range(m_n.shape[0]), range(m_n.shape[1]), range(m_n.shape[2])
    m_n_1d = [(m_n[i, j, k], i, j, k) for k in s2 for i in s1 for j in s0]
    df = pd.DataFrame(m_n_1d, columns=['val', 'i', 'j', 'z'])
    df.to_csv('final.csv', decimal='.', sep=';', index=False)
