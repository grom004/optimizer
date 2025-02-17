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



Программа является одним из решений следующей технической задачи. Имеются счетчики посещения разных сайтов в интернете. Когда пользователь заходит на сайт или в приложение, его действие логируется в файле **triggers.csv**. Также у компании есть несколько выкупленных баннеров, на которых показываются предложения зарегистрироваться в новом видеосервисе компании. В случае если предложение заинтересовало пользователя, он по нему кликает, что логируется в файле **actions.csv**.

Event Seq с действиями пользователей:
- **triggers.csv** - посещения;
- **actions.csv** - показы.

#### actions.csv:
  - **guid** - идентификатор пользователя;
  - **date** - дата взаимодействия с пользователем;
  - **result** - 0 если пользователь нет заинтересовался предложением, 1 иначе.

#### triggers.csv:

- **guid** - идентификатор пользователя;
- **date** - дата посещения;
- **trigger** - идентификатор ресурса;
- **type** - тип сбора метрики.

[Cкачать CSV-файлы здесь.](https://disk.yandex.ru/d/KetMP60FvKsK9Q)


Ограничения в задаче:
1. В рамках задачи нельзя взаимодействовать с пользователем чаще чем один раз в две недели;
2. Стоимость одного показа баннера: 1$;
3. Вознаграждение за результативный показ: 5$;
4. Нужно получить максимальный оборот, при положительном балансе. 



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


The program is one of the solutions to the following technical problem. There are counters for visiting different sites on the Internet. When a user visits a site or an application, his action is logged in the triggers.csv file. Also, the company has several redeemed banners that show offers to register in the company's new video service. If a user is interested in the offer, he clicks on it, which is logged in the actions.csv file.

Event Seq with user actions:
- **triggers.csv** - visits;
- **actions.csv** - shows.

#### actions.csv:
  - **guid** - user id;
  - **date** - date of interaction with the user;
  - **result** - 0 if the user was not interested in the offer, 1 otherwise.

#### triggers.csv:

- **guid** - user id;
- **date** - date of visit;
- **trigger** - resource identifier;
- **type** - type of metric collection.

[Download CSV here.](https://disk.yandex.ru/d/KetMP60FvKsK9Q)

The limits of the task are:
1. You cannot communicate with the user more than once every two weeks;
2. Cost of one banner display: 1$;
3. Reward for a successful display: 5$;
4. We need to get the maximum turnover, with a positive balance. 
