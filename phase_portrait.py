import pandas as pd
import numpy as np
# интерполяция
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline as CubicSpline
# Рисование графиков
import matplotlib.pyplot as plt
import matplotlib as mpl


def slice_data(df, slice_period, reset_index=True):
    # 2 сортировки, чтобы последняя дата оставалась
    if reset_index:
        return df.copy().sort_values(by='date', ascending=False)[::slice_period].sort_values(by='date').reset_index(
            drop=True)
    else:
        return df.copy().sort_values(by='date', ascending=False)[::slice_period].sort_values(by='date')


def get_arange(length, step):
    return np.arange(1, length + step, step=step)


def get_interpotate_range(array, step):
    return np.arange(array.min(), array.max() + step, step=step)


def deriv_spline(spline, n):
    # Находим производные сплайна порядка 0-3
    pp = []
    for i in range(n):
        pp.append(spline.derivative(i))
    return pp


def calc_spline_values(xx, pp):
    # Вычисляем значения для производных сплайна
    ppf = []
    for i in range(len(pp)):
        ppf.append(pp[i](xx))
    return ppf


def interpolate(ppf, step):
    # Задаем интервалы для интерполяции
    t = np.arange(1, ppf[0].shape[0] + 1)
    pp1f_y = get_arange(ppf[0].shape[0], step=step)
    # Производим интерполяцию
    interp1 = lambda x: interp1d(t, x, kind='cubic', fill_value="extrapolate")(pp1f_y)  # нужен ли параметр fill_value
    deriv = []
    for i in range(len(ppf)):
        deriv.append(interp1(ppf[i]))
    return deriv


def get_phase_portrait(x, y, step_spline=0.05, step_deriv=.01, deriv_num=4):
    # Высчитываем данные портрета и xx
    xx = get_arange(x.shape[0], step=step_spline)
    spline = CubicSpline(x, y, bc_type='not-a-knot')  # clamped
    # yy = spline(xx)
    pp = deriv_spline(spline, deriv_num)
    ppf = calc_spline_values(xx, pp)

    deriv = interpolate(ppf, step_deriv)
    return xx, deriv


def get_ticks(arr, num=5):
    # Поиск нужного кол-во делений на шкалах, возвращает массив с данными
    assert num > 2

    mask = np.linspace(0, arr.shape[0] - 1, num=num).round().astype(int)
    if type(arr) == pd.core.frame.DataFrame or type(arr) == pd.core.series.Series:
        return [arr.iloc[i] for i in mask]
    else:
        return [arr[i] for i in mask]


def plot_phase_portrait(x, tt, deriv, dates, start_date='', end_date='', cbar_ticks_num=5, graph_name='', ylabel='Rate',
                        xlabel='Value', cmap=mpl.cm.plasma):
    '''

    :param deriv: Производные сплайна.
    :param tt: Интерполированные значения x.
    :param dates: Даты.
    :param start_date: Начальная дата для выделения цветом.
    :param end_date: Конечная дата для выделения цветом.
    :param cbar_ticks_num: Количество подписей на цветовой шкале.
    :param graph_name: Название графика.
    :param ylabel: Название оси Y.
    :param xlabel: Название оси X.
    :param cmap: Цветовая схема для наложения на график.
    :return:
    '''
    plt.rcParams.update({'xtick.labelsize': 15})
    plt.rcParams.update({'ytick.labelsize': 15})

    fig, ax = plt.subplots()
    if not start_date and not end_date:
        start_date = dates.min()
        end_date = dates.max()

    ## Находим маску для выбранного периода
    dates_colored_mask = get_dates_mask(dates, start_date, end_date)
    colored_mask = get_mask_tt(x, tt, dates_mask=dates_colored_mask)
    tt_colored = tt[colored_mask]

    ## Рисуем фоновый график
    ax.plot(deriv[0], deriv[1], ':g', alpha=0.7)

    ### Красим выбранный период графика
    norm = mpl.colors.Normalize(vmin=tt_colored.min(), vmax=tt_colored.max())
    ax.scatter(deriv[0][colored_mask], deriv[1][colored_mask], c=tt, cmap=cmap, s=0.1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        orientation='vertical', ticks=get_ticks(tt_colored, cbar_ticks_num))
    cbar.ax.set_yticklabels(map(lambda x: x.strftime('%d.%m.%y'), get_ticks(dates[dates_colored_mask], cbar_ticks_num)))
    ###

    if graph_name:
        plt.title(graph_name)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Выделение линии Y=0
    plt.axhline(y=0, color='k', linestyle='-')

    # настройка размера фигуры
    fig.set_figheight(8)
    fig.set_figwidth(9)
    plt.show()


def plot_phase_portrait_old(xx, dates, deriv, tt, name, ticks_num=5, save_fig_name='', y_limits=(), x_limits=()):
    cmap = plt.cm.plasma
    norm = plt.Normalize(xx.min(), xx.max())

    c_bar = plt.cm.plasma(np.linspace(0, 1, tt.shape[0]))

    fig, ax = plt.subplots()
    ax.scatter(deriv[0], deriv[1], c=c_bar, s=0.05)
    cax = ax.plot(deriv[0][-1], deriv[1][-1], '*g', markersize=6)
    ax.annotate(dates.iloc[-1].strftime('%d.%m.%y'), xy=(deriv[0][-1], deriv[1][-1]), color='black', ha="right")
    ax.set_ylabel('Rate')
    ax.set_xlabel('Value')

    ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='x')

    plt.title(name)

    fig.set_figheight(8)
    fig.set_figwidth(8)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ticks=get_ticks(xx, ticks_num),
                        orientation='vertical')
    cbar.ax.set_yticklabels(map(lambda x: x.strftime('%d.%m.%y'), get_ticks(dates, ticks_num)))
    if len(y_limits) == 2:
        plt.ylim(y_limits)
    if len(x_limits) == 2:
        plt.xlim(x_limits)
    plt.grid()
    if save_fig_name:
        plt.savefig(f"{save_fig_name}.png", dpi=200)


def get_filename(name, min_date, max_date, freq):
    return f"{name}_{min_date}_to_{max_date}_f_{freq}"


def calc_phase_portrait(df, slice_period=1, normalize=''):
    if normalize:
        normalization_value = df['value'][(df['date'] > pd.to_datetime(normalize, format='%d.%m.%Y'))].head(1).iloc[0]
        df['value'] = df['value'] / normalization_value
    else:
        df['value'] = df['value'] / 1000
    df = slice_data(df, slice_period)

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    x = df.index.to_series() + 1
    y = df['value']
    x, y, xx, deriv, tt = calc_phase_portrait_raw(x, y)

    return x, y, xx, deriv, tt, slice_period, df['date']


def calc_phase_portrait_raw(x, y):
    xx, deriv = get_phase_portrait(x, y, step_spline=.05, step_deriv=.01, deriv_num=2)
    tt = get_arange(x.shape[0], step=.0005)

    return x, y, xx, deriv, tt


def generate_phase_portrait(df, slices, normalize=''):
    data = []
    for slice_period in slices:
        data.append(calc_phase_portrait(df.copy(), slice_period, normalize))
    return data


def generate_png(path, name, df, generated_data, x_lim, y_lim, upscale=1):
    assert upscale > 0

    for data in generated_data:
        x, y, xx, deriv, tt, slice_period = data

        filename = get_filename(name, df['date'].min().strftime('%d_%m_%y'), df['date'].max().strftime('%d_%m_%y'),
                                slice_period)
        deriv = list(map(lambda x: x * upscale, deriv))

        plot_phase_portrait_old(xx, df['date'], deriv, tt, f"{name} t={slice_period}", ticks_num=5,
                                save_fig_name=path + filename, y_limits=y_lim, x_limits=x_lim)


def get_mask_tt(x, tt, dates=None, start_date=None, end_date=None, dates_mask=None):
    if dates_mask is None:
        if not dates or not start_date or not end_date:
            raise Exception("You should pass dates_mask or dates, start_date and end_date")
        dates_mask = get_dates_mask(dates, start_date, end_date)
    filtered_x = x[dates_mask]
    return (tt >= filtered_x.min()) & (tt <= filtered_x.max())


def get_dates_mask(dates, start_date, end_date):
    return (dates >= pd.to_datetime(start_date, format='%d.%m.%Y')) & (
            dates <= pd.to_datetime(end_date, format='%d.%m.%Y'))
