# Parallel Computing Coursework

Репозиторий с лабораторными работами и курсовым проектом по дисциплине
"Параллельные вычисления".

## Содержание

- `lab 1/` - CPU multi-processing, исходный код и результаты бенчмарков.
- `lab 2/` - GPU multi-processing, скрипты и таблицы замеров.
- `lab 3/` - distributed computing/MPI.
- `lab 4/` - multiprocessing multi-agent simulation, результаты и отчетные графики.
- `lab 5/` - асинхронная модель gateway, стратегии, тесты и отчетные материалы.
- `lab 6/` - Spark MLlib, данные Telco Customer Churn, метрики и визуализации.
- `REPORTS/` - итоговые DOCX-отчеты по лабораторным работам и курсовой.
- `course/` - материалы курсовой работы.

## Курсовая работа

Тема курсовой: параллельная реализация простого генетического алгоритма
(Simple Genetic Algorithm, SGA) для оптимизации математических функций.

Основные файлы:

- `course/sga_parallel.py` - ядро SGA, последовательная и параллельная оценка
  fitness через `multiprocessing.Pool`, CLI для одиночного запуска.
- `course/run_experiments.py` - воспроизводимый пакет экспериментов, запись CSV
  и генерация графиков.
- `course/regenerate_plots.py` - повторная генерация PNG-графиков из уже
  сохраненных CSV.
- `course/build_report.py` - сборка DOCX-отчета из результатов экспериментов.
- `course/restyle_report_like_example.py` - приведение DOCX-отчета к стилю
  образца.
- `course/results/` - сохраненные таблицы и графики экспериментов.
- `REPORTS/Курсовая.docx` - финальный отчет по курсовой работе.

Сохраненные результаты курсовой:

- `course/results/experiment_runs.csv` - все запуски экспериментов.
- `course/results/convergence_history.csv` - история сходимости по поколениям.
- `course/results/speedup_summary.csv` - среднее время, ускорение и эффективность
  для 1, 2, 4 и 8 процессов.
- `course/results/plots/convergence.png` - график сходимости.
- `course/results/plots/speedup.png` - график ускорения.
- `course/results/plots/function_quality.png` - сравнение качества на Sphere,
  Rastrigin и Rosenbrock.

## Запуск курсовой

Зависимости для курсовой части:

```powershell
python -m pip install pillow python-docx
```

Одиночный запуск SGA:

```powershell
cd course
python sga_parallel.py --objective rastrigin --dimensions 20 --population-size 100 --generations 300 --processes 4 --eval-repeats 35 --output-json results/single_run.json
```

Полное воспроизведение экспериментов и графиков:

```powershell
cd course
python run_experiments.py --replicates 5 --eval-repeats 35
```


