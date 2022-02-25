import pandas as pd
import numpy as np
# интерполяция
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline as CubicSpline
# Рисование графиков
import matplotlib.pyplot as plt
import matplotlib as mpl


def slice_data(df, slice_period=1, reset_index=True):
    # 2 сортировки, чтобы последняя дата оставалась
    if reset_index:
        return df.sort_values(by='date', ascending=False)[::slice_period].sort_values(by='date').reset_index()
    else:
        return df.sort_values(by='date', ascending=False)[::slice_period].sort_values(by='date')


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


def interpolate(ppf, step, params):
    if params is None:
        params = {}
    kind = params.get('interpolate.kind', 'cubic')
    fill_value = params.get('interpolate.fill_value', 'extrapolate')
    # Задаем интервалы для интерполяции
    t = np.arange(1, ppf[0].shape[0] + 1)
    pp1f_y = get_arange(ppf[0].shape[0], step=step)
    # Производим интерполяцию
    interp1 = lambda x: interp1d(t, x, kind=kind, fill_value=fill_value)(pp1f_y)  # нужен ли параметр fill_value
    deriv = []
    for i in range(len(ppf)):
        deriv.append(interp1(ppf[i]))
    return deriv


def get_phase_portrait(x, y, step_spline=0.05, step_deriv=.01, deriv_num=4, params=None):
    if params is None:
        params = {}
    bc_type = params.get('get_phase_portrait.bc_type', 'not-a-knot')
    # Высчитываем данные портрета и xx
    xx = get_arange(x.shape[0], step=step_spline)
    spline = CubicSpline(x, y, bc_type=bc_type)  # clamped
    # yy = spline(xx)
    pp = deriv_spline(spline, deriv_num)
    ppf = calc_spline_values(xx, pp)

    deriv = interpolate(ppf, step_deriv, params)
    return xx, deriv


def get_ticks(arr, num=5):
    # Поиск нужного кол-во делений на шкалах, возвращает массив с данными
    assert num > 2

    mask = np.linspace(0, arr.shape[0] - 1, num=num).round().astype(int)
    if type(arr) == pd.DataFrame or type(arr) == pd.Series:
        return [arr.iloc[i] for i in mask]
    else:
        return [arr[i] for i in mask]


def plot_phase_portrait(x, tt, deriv, dates, start_date='', end_date='', cbar_ticks_num=5, graph_name='', ylabel='Rate',
                        xlabel='Value', cmap=mpl.cm.plasma, fig=None, ax=None, plot_bar=True,
                        background_graph_params=None, rcParams=None, crop=False):
    '''

    :param rcParams: Параметры для настройки графика в целом.
    :param background_graph_params: Параметры для фонового графика.
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
    if not background_graph_params:
        background_graph_params = {}
    if not rcParams:
        rcParams = {'xtick.labelsize': 15, 'ytick.labelsize': 15}
    plt.rcParams.update(rcParams)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    if not start_date:
        start_date = dates.min()
    if not end_date:
        end_date = dates.max()

    ## Находим маску для выбранного периода
    dates_colored_mask = get_dates_mask(dates, start_date, end_date)
    colored_mask = get_mask_tt(x, tt, dates_mask=dates_colored_mask)
    tt_colored = tt[colored_mask]

    ## Рисуем фоновый график
    if not background_graph_params.pop('disable', False):
        ax.plot(deriv[0], deriv[1], alpha=0.7, **background_graph_params)

    ### Красим выбранный период графика
    norm = mpl.colors.Normalize(vmin=tt_colored.min(), vmax=tt_colored.max())
    ax.scatter(deriv[0][colored_mask], deriv[1][colored_mask], c=tt_colored, cmap=cmap, s=0.1)
    if plot_bar:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                            orientation='vertical', ticks=get_ticks(tt_colored, cbar_ticks_num))
        cbar.ax.set_yticklabels(
            map(lambda x: x.strftime('%d.%m.%y'), get_ticks(dates[dates_colored_mask], cbar_ticks_num)))
    ###
    if crop:
        if type(crop) == tuple:
            ax.set_xlim(crop)
        ax.set_xlim((deriv[0][colored_mask].min()-500, deriv[0][colored_mask].max()+500))
    if graph_name:
        ax.set_title(graph_name)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Выделение линии Y=0
    ax.axhline(y=0, color='k', linestyle='-')

    # настройка размера фигуры
    # fig.set_figheight(8)
    # fig.set_figwidth(9)
    return fig, ax


def calc_phase_portrait(df, slice_period=1, reset_index=False, params=None):
    if params is None:
        params = {}
    df = slice_data(df, slice_period, reset_index)

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    x = df.index.to_series() + 1
    y = df['value']
    x, y, xx, deriv, tt = calc_phase_portrait_raw(x, y, params=params)

    return x, y, xx, deriv, tt, slice_period, df['date']


def calc_phase_portrait_raw(x, y, params=None):
    if params is None:
        params = {}
    step_spline = params.get('calc_phase_portrait_raw.step_spline', .05)
    step_deriv = params.get('calc_phase_portrait_raw.step_deriv', .01)
    deriv_num = params.get('calc_phase_portrait_raw.deriv_num', 2)
    tt_step = params.get('calc_phase_portrait_raw.tt_step', .0005)

    xx, deriv = get_phase_portrait(x, y, step_spline=step_spline, step_deriv=step_deriv, deriv_num=deriv_num,
                                   params=params)
    tt = get_arange(x.shape[0], step=tt_step)

    return x, y, xx, deriv, tt


def generate_phase_portrait(df, slices, params=None):
    if params is None:
        params = {}
    data = []
    for slice_period in slices:
        data.append(calc_phase_portrait(df.copy(), slice_period, params=params))
    return data


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
