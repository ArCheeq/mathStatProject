import math
import traceback

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')


# Фукнция для построения доверительного интервала

def norm_conf_int(alpha, mean_hat, std_hat, margin=5):

    plt.figure(figsize=(10, 5))
    xs = np.linspace(mean_hat - margin, mean_hat + margin)
    pdf = stats.norm(mean_hat, std_hat).pdf(xs)

    plt.plot(xs, pdf)
    plt.ylabel('$f(x)$', fontsize=18)
    plt.xlabel('$   x$', fontsize=18)

    left, right = stats.norm.interval(1 - alpha, loc=mean_hat, scale=std_hat)

    for i in [left, right]:
        y_max = plt.ylim()[1]
        plt.axvline(i, color="blue", linestyle="dashed", lw=2)

        if i == left:
            xq = np.linspace(mean_hat - margin, left)
        else:
            xq = np.linspace(right, mean_hat + margin)

        text_margin = 0.05
        plt.text(i + text_margin, 0.8*y_max, round(i), color="blue", fontsize=18)
        yq = stats.norm(mean_hat, std_hat).pdf(xq)
        plt.fill_between(xq, 0, yq, color="blue", alpha=0.3)

    return left, right, plt


df = pd.read_csv('US_Accidents_Dec21.csv')

df['Start_Time'].tail()

df.isna().sum().sort_values(ascending=False)

df['City_State'] = df['City'] +' (' + df['State'] + ')'
top_cities = df.groupby('City_State').count().sort_values('Severity', ascending=False)['ID'][0:10].reset_index()

print(top_cities)

# Гистограмма по топ 10 городам с авариями

plt.figure(figsize=(10, 10))
plt.bar(top_cities['City_State'], top_cities['ID'])
ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
plt.ylabel('Accidents Count', size=15, color='g')
plt.xlabel('City', size=15, color='g')
plt.xticks(rotation='vertical')
plt.grid()
plt.show()

# Гистограмма аварий по годам

year_data = df.groupby(pd.DatetimeIndex(df['Start_Time']).year).count()['ID'].reset_index()
year_data = year_data.sort_values('Start_Time', ascending=False).reset_index()
year_data = year_data.drop(columns='index')

print(year_data)

year_data['lagged'] = year_data['ID'].shift(-1)

# Столбец для процентного изменения

for year in year_data:
    year_data['percent_change'] = year_data['ID'] / year_data['lagged'] * 100 - 100

year_data['Start_Time'] = year_data['Start_Time'].astype(str)

accidents_count = year_data[['ID', 'percent_change']]
accidents_count['count'] = range(0, 6)
accidents_count = accidents_count.values.tolist()

ax = plt.gca()
ax.spines[['top', 'right']].set_visible(False)
plt.bar(year_data['Start_Time'], year_data['ID'])
plt.xlabel('Accidents Count (M)')
plt.ylabel('Years')

for value, percent, index in accidents_count:
    plt.text(round(index), round(value), str(round(percent, 2)) + '%', ha='center', va='bottom')

plt.text(4, 1400000, '(%age increase)')

plt.show()

print(year_data)

# Вывод основных статистических характеристик

mu_hat = year_data['ID'].mean()
print(f"Мат ожидание: {round(mu_hat)}")
print(f"Разброс велечин: {year_data['ID'].max() - year_data['ID'].min()}")
n = year_data['ID'].count()
sd_hat = year_data['ID'].std(ddof=1)/np.sqrt(n)
print(f"Среднее квадратическое отклонение: {round(sd_hat, 4)}")
print(f"Объем выборки: {n}")

# Вычисление доверительного интервала
alpha = 0.05
left, right, plt = norm_conf_int(alpha, mu_hat, sd_hat, margin=5)

print(f"Доверительный интервал [{round(left)}; {round(right)}] ширины {round(right - left)}")
plt.show()

# Проверка гипотезы о экспоненциальном законе распределения случайной величины c помощью критерия Колмогорова

res = stats.kstest(year_data['ID'], 'expon', args=(0, year_data['ID'].mean()))
print(res)

# Дисперсионный анализ зависимости количества дтп от времени суток

df['Start_Time'] = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')

data_morning = df[(6 < df.Start_Time.dt.hour) & (df.Start_Time.dt.hour < 10)][["City_State", "Start_Time"]].City_State.value_counts()[0:10]
data_morning = pd.DataFrame(data_morning)
data_morning.index = ['Miami (FL)', 'Los Angeles (CA)', 'Orlando (FL)', 'Dallas (TX)', 'Houston (TX)', 'Charlotte (NC)', 'Sacramento (CA)', 'San Diego (CA)', 'Raleigh (NC)', 'Minneapolis (MN)']
print(data_morning)

data_dinner = df[(10 < df.Start_Time.dt.hour) & (df.Start_Time.dt.hour < 16)][["City_State", "Start_Time"]].City_State.value_counts()[0:10]
data_dinner = pd.DataFrame(data_dinner)
data_dinner.index = ['Miami (FL)', 'Los Angeles (CA)', 'Orlando (FL)', 'Dallas (TX)', 'Houston (TX)', 'Charlotte (NC)', 'Sacramento (CA)', 'San Diego (CA)', 'Raleigh (NC)', 'Minneapolis (MN)']
print(data_dinner)

data_evening = df[(16 < df.Start_Time.dt.hour) & (df.Start_Time.dt.hour < 22)][["City_State", "Start_Time"]].City_State.value_counts()[0:10]
data_evening = pd.DataFrame(data_evening)
data_evening.index = ['Miami (FL)', 'Los Angeles (CA)', 'Orlando (FL)', 'Dallas (TX)', 'Houston (TX)', 'Charlotte (NC)', 'Sacramento (CA)', 'San Diego (CA)', 'Raleigh (NC)', 'Minneapolis (MN)']
print(data_evening)

f_oneway = stats.f_oneway(data_morning.City_State, data_dinner.City_State, data_evening.City_State)

print(f_oneway)

sns.displot(df['Start_Time'].dt.hour, bins=24)
plt.show()

# Создание HeatMap

import folium
from folium.plugins import HeatMap

lat_long_data = list(zip(list(df.Start_Lat), list(df.Start_Lng)))

map = folium.Map()

HeatMap(lat_long_data[:int(len(lat_long_data)*0.01)]).add_to(map)

map.save('map.html')
