# optimizer

#### Программа анализирует поведение пользователей, обучает модель логистической регрессии для прогнозирования их действий и оценивает финансовую эффективность показов. Программа очищает и объединяет данные, рассчитывает ключевые показатели активности, прогнозирует вероятность ответа и оценивает рентабельность взаимодействия:

1. загрузка и предварительная обработка данных;
2. анализ триггеров и действий;
3. обучение модели машинного обучения;
4. оценка модели (матрица, точность, отзыв, f1, ROC, AUC-ROC);
5. финансовый анализ.

Запуск программы:

1. Перейти в директорию проекта: `cd /dir_name`
  
3. Создать виртуальную среду: `python3 -m venv venv`
   
4. Активировать виртуальную среду: `source venv/bin/activate`

5. Установка необходимых библиотек:
      - `pip3 install pandas`
      - `pip3 install numpy`
      - `pip3 install scikit-learn`
      - `pip3 install seaborn`
      - `pip3 install matplotlib`
        
6. Запуск кода: `python3 optimizer.py`

[Cкачать CSV-файлы здесь.](https://disk.yandex.ru/d/KetMP60FvKsK9Q)

Программа является одним из решений следующей технической задачи. Имеются счетчики посещения разных сайтов в интернете. Когда пользователь заходит на сайт или в приложение, его действие логируется в файле **triggers.csv**. Также у компании есть несколько выкупленных баннеров, на которых показываются предложения зарегистрироваться в новом видеосервисе компании. В случае если предложение заинтересовало пользователя, он по нему кликает, что логируется в файле **actions.csv**.

Event Seq с действиями пользователей:
- **triggers.csv** - посещения;
- **actions.csv** - показы.

#### actions.csv:
-- **guid** - идентификатор пользователя;

**date** - дата взаимодействия с пользователем;

**result** - 0 если пользователь нет заинтересовался предложением, 1 иначе.

triggers.csv:
      - **guid** - идентификатор пользователя;
    - **guid** - идентификатор пользователя;
    - **date** - дата посещения;
    **trigger** - идентификатор ресурса;

**type** - тип сбора метрики.



---

#### The program analyzes user behavior, trains a logistic regression model to predict their actions, and evaluates the financial effectiveness of displays. It cleans and merges data, calculates key activity metrics, predicts response probability, and assesses interaction profitability:

1. data loading and preprocessing;
2. trigger and action analysis;
3. machine learning model training;
4. model evaluation(matrix, precision, recall, f1, ROC, AUC-ROC);
5. financial analysis.

Start the program:

1. Go to the project directory: `cd /dir_name`

2. Create a virtual environment: `python3 -m venv venv`

3. Activate the virtual environment: `source venv/bin/activate`

4. Install the necessary libraries:
      - `pip3 install pandas`
      - `pip3 install numpy`
      - `pip3 install scikit-learn`
      - `pip3 install seaborn`
      - `pip3 install matplotlib`

5. Run the code: `python3 optimizer.py`

[Download CSV here.](https://disk.yandex.ru/d/KetMP60FvKsK9Q)
