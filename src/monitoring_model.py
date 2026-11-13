import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Загрузка данных из CSV-файла
df = pd.read_csv('../data/sensors_telemetry.csv')

# Проверка загруженных данных
print(df.head())

# 2. Расчет теоретического теплопотребления по формуле теплового баланса
# Формула: Q = расход * плотность воды * теплоёмкость воды * ΔT (разница температур)
# Примечание: Плотность воды ~ 1 кг/л, удельная теплоёмкость воды ~ 4.2 kJ/(кг·°C)

# Добавляем столбец рассчитанного теплопотребления
df['theoretical_Q'] = df['flow_rate'] * 4.2 * (df['supply_temp'] - df['return_temp'])

# Сравним расчетное теплопотребление с фактическим значением счетчика
df['difference'] = df['meter_reading'] - df['theoretical_Q']

# 3. Обнаружение аномалий и вывод предупреждений
for index, row in df.iterrows():
    # Перетоп определяется превышением разницы показаний счетчика и расчета больше 10%
    if abs(row['difference']) / max(abs(row['meter_reading']), 1e-9) > 0.1:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Перетоп обнаружен в моменте {index}, разница составляет {row['difference']:.2f}")
    
    # Опасно низкая температура в помещении (<18°C)
    if row['room_temp'] < 18:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Температура в помещении упала ниже нормы ({row['room_temp']:.1f} °C)")

# 4. Создание простой модели линейной регрессии для прогноза теплопотребления
X = df[['outside_temp']]  # Входной признак - наружная температура
y = df['meter_reading']   # Целевой показатель - фактический расход тепла

model = LinearRegression()
model.fit(X, y)

# Прогнозируем теплопотребление по модели
df['predicted_heat_consumption'] = model.predict(X)

# Оцениваем качество модели через коэффициент детерминации (R²)
r_squared = model.score(X, y)
print(f"Коэффициент детерминации (R²): {r_squared:.2f}")

# 5. Построение графика зависимости теплопотребления от наружной температуры
plt.figure(figsize=(10, 6))
plt.scatter(df['outside_temp'], df['meter_reading'], label='Фактические значения', color='blue')
plt.plot(df['outside_temp'], df['predicted_heat_consumption'], label='Прогноз модели', color='red')
plt.title('Зависимость теплопотребления от наружной температуры')
plt.xlabel('Температура снаружи (°C)')
plt.ylabel('Теплопотребление (kW/h)')
plt.legend()

# Создаем папку reports если отсутствует
if not os.path.exists('reports'):
    os.makedirs('reports')

# Сохраняем отчетный график в файл
plt.savefig('reports/monitoring_report.png')
print("Отчетный график успешно сохранен")