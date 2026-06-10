# Отчёт по лабораторной работе 5

## Тема

Асинхронный шлюз агрегации данных с динамической балансировкой.

## Выбранные внешние API

В работе используются локальные эмуляторы HTTP API на `aiohttp.web`. Такой вариант выбран вместо публичных API, потому что он воспроизводим: задержки, таймауты и вероятность ошибок контролируются кодом тестовой подсистемы.

- `stable`: All APIs respond within 50-180 ms and do not return errors. Timeout: 1.2 s, requests per API: 3.
- `slow`: One API responds after 3-5 seconds, while the others are fast. Timeout: 1.2 s, requests per API: 3.
- `unstable`: Some APIs return HTTP 500/503 with probability 30-50%. Timeout: 1.2 s, requests per API: 3.

## Архитектура

- `gateway/api.py` — HTTP-шлюз с endpoint `POST /aggregate`.
- `gateway/strategies.py` — стратегии `fixed`, `timeout_race` и `adaptive`.
- `gateway/stats.py` — скользящее окно статистики и адаптивная корректировка concurrency.
- `gateway/emulators.py` — локальные API-эмуляторы.
- `gateway/benchmark.py` — CLI-бенчмарк, сохранение JSON/CSV и построение графиков.
- `tests/test_adaptive.py` — модульные тесты логики адаптации.

## Адаптивная стратегия

Для каждого URL ведётся скользящее окно последних запросов. В окне считаются доля успешных ответов и среднее время ответа. Если доля ошибок выше 30% или среднее время ответа превышает 2000 мс, concurrency для этого API уменьшается на 1, но не ниже 1. Если success rate не ниже 90%, а среднее время ниже 500 мс, concurrency осторожно увеличивается на 1 до внутреннего верхнего лимита.

## Результаты

Тестовый прогон выполнен с `repeats=10`. Для каждого вызова агрегировалось по 3 запроса к каждому из трёх API.

| Scenario | Strategy | Avg time, ms | Avg successful | Avg failed | Avg timeouts |
|---|---:|---:|---:|---:|---:|
| stable | fixed | 402.27 | 9.00 | 0.00 | 0.00 |
| stable | timeout_race | 161.03 | 9.00 | 0.00 | 0.00 |
| stable | adaptive | 255.81 | 9.00 | 0.00 | 0.00 |
| slow | fixed | 1809.04 | 6.00 | 3.00 | 3.00 |
| slow | timeout_race | 1203.50 | 6.00 | 3.00 | 3.00 |
| slow | adaptive | 3162.79 | 6.00 | 3.00 | 3.00 |
| unstable | fixed | 825.44 | 6.10 | 2.90 | 0.00 |
| unstable | timeout_race | 392.76 | 6.30 | 2.70 | 0.00 |
| unstable | adaptive | 838.83 | 6.90 | 2.10 | 0.00 |

Графики сохранены рядом с этим отчётом:

- `avg_time_by_strategy.png` — среднее время выполнения по стратегиям.
- `adaptive_concurrency.png` — изменение concurrency для адаптивной стратегии.
- `success_failure_by_strategy.png` — успешные и неуспешные запросы.

## Поведение adaptive

- `slow:127.0.0.1:60823/data`: concurrency 3 -> 6.
- `slow:127.0.0.1:60824/data`: concurrency 3 -> 6.
- `slow:127.0.0.1:60825/data`: concurrency 3 -> 1.
- `stable:127.0.0.1:60623/data`: concurrency 3 -> 6.
- `stable:127.0.0.1:60624/data`: concurrency 3 -> 6.
- `stable:127.0.0.1:60625/data`: concurrency 3 -> 6.
- `unstable:127.0.0.1:63881/data`: concurrency 3 -> 6.
- `unstable:127.0.0.1:63882/data`: concurrency 3 -> 1.
- `unstable:127.0.0.1:63883/data`: concurrency 3 -> 1.

## Анализ

На стабильных API агрессивные стратегии обычно выигрывают по времени, потому что все ответы быстрые и нет смысла искусственно ограничивать запросы. На медленном сценарии `fixed` страдает от очередей: медленные запросы занимают слоты, а следующие ждут освобождения semaphore. `timeout_race` быстрее отсекает медленные API, так как запускает все запросы сразу. `adaptive` занимает промежуточную позицию: после накопления статистики он снижает нагрузку на проблемные URL и защищает общий пул запросов.

В нестабильном сценарии адаптивная стратегия уменьшает concurrency у API с частыми 500/503, поэтому сервис тратит меньше одновременных слотов на заведомо проблемные источники. Ограничение подхода — адаптации нужно накопить несколько измерений, поэтому первые запросы выполняются с начальной настройкой.

## Инструкция по запуску

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py serve --host 127.0.0.1 --port 8000
```

Проверка endpoint:

```bash
curl -X POST http://127.0.0.1:8000/aggregate -H "Content-Type: application/json" -d "{\"urls\":[\"https://jsonplaceholder.typicode.com/todos/1\"],\"strategy\":\"adaptive\"}"
```

Запуск встроенной тестовой системы:

```bash
python main.py test --scenario all --repeats 10 --output report/
```

Модульные тесты:

```bash
pytest
```

## Выводы

Асинхронность позволяет выполнять независимые HTTP-запросы без последовательного ожидания каждого API. Фиксированная стратегия предсказуема, но плохо реагирует на медленные источники. `timeout_race` хорош для минимизации времени ожидания, но может создавать всплеск нагрузки. `adaptive` добавляет обратную связь: он использует реальные измерения времени ответа и ошибок, чтобы постепенно перераспределять доступную параллельность.
