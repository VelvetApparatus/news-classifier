# Руководство пользователя и эксплуатации сервиса

## Описание

Сервис предназначен для автоматической тематической классификации новостного потока на русском языке.

Система принимает новости (через Kafka или API), определяет их тему с помощью модели RuBERT + KMeans и сохраняет результат в базу данных.

---

## Архитектура системы

### Основные компоненты

- Producer — сбор новостей (API / парсер)
- Kafka — транспорт сообщений
- Consumer — запись данных в БД
- Batch Worker — инференс модели
- PostgreSQL — хранение данных
- Inference API — ручной доступ к модели

---

## Поток обработки данных

1. Новость поступает в Kafka
2. Consumer читает сообщение и сохраняет в PostgreSQL
3. Batch worker периодически:
   - выбирает необработанные записи
   - выполняет инференс
   - обновляет результат
4. API позволяет получить результат или выполнить инференс вручную

---

## Режимы работы

### Offline режим

- обучение модели RuBERT
- кластеризация (KMeans / HDBSCAN)
- генерация:
  - `clusterer.pkl`
  - `topics.json`
  - `bert_config.json`

### Online режим

- обработка входящих новостей
- назначение кластера
- сохранение результатов
- обработка unknown тем

---

## Инференс-модуль

### Назначение

Сервис принимает текст новости и возвращает:

- topic_id
- label
- top_words
- confidence
- latency

---

### Вход

```json
{
  "text": "Россия провела переговоры с США"
}
```

---

### Выход

```json
{
  "topic_id": 1,
  "label": "Политика",
  "top_words": ["россия", "сша", "переговоры"],
  "confidence": 0.82,
  "latency_ms": 45
}
```

---

### Логика инференса

1. Текст → embedding (RuBERT)
2. embedding → кластер (KMeans)
3. вычисление расстояния до центроида
4. преобразование в confidence
5. если confidence < threshold → topic = "unknown"

---

## API

### POST /predict

Выполнить инференс вручную

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Россия провела переговоры с США"}'
```

---

### GET /healthz

Проверка состояния системы

Проверяет:
- Kafka
- PostgreSQL
- модель
- metadata

---

### GET /metrics

Метрики (Prometheus)

---

## Структура данных

### Таблица: news

| поле | тип | описание |
|------|-----|----------|
| id | uuid | идентификатор |
| external_id | string | id для идемпотентности |
| text | text | текст новости |
| topic_id | int | кластер |
| confidence | float | уверенность |
| status | string | processed / unknown |
| created_at | timestamp | время |

---

### Таблица: topic_metadata

| поле | тип | описание |
|------|-----|----------|
| cluster_id | int | id кластера |
| label | string | название темы |
| top_words | jsonb | ключевые слова |

---

## Обработка ошибок

| Сценарий | Поведение |
|---------|----------|
| confidence < threshold | topic = unknown |
| ошибка модели | статус ERROR |
| Kafka недоступна | retry |
| БД недоступна | логирование ошибки |

---

## Надежность системы

- идемпотентность через external_id
- batch обработка
- retry механизм
- периодическая загрузка metadata
- разделение ingestion и inference

---

## Мониторинг

### Логирование

Логируются:
- входящие сообщения
- инференс
- ошибки

---

### Метрики

- news_published_total
- incoming_news_stored_total
- batch_processing_duration_seconds
- news_processed_total
- news_marked_unknown_total

---

### Метрики нагрузки

- latency
- throughput
- размер батча
- доля unknown

---

## Контейнеризация

### Docker

Каждый компонент имеет отдельный контейнер:
- API
- Kafka consumer
- batch worker
- producer

---

### Запуск

```bash
docker-compose up --build
```

---

## Переменные окружения

| Переменная | Назначение |
|-----------|-----------|
| KAFKA_BOOTSTRAP_SERVERS | адрес Kafka |
| KAFKA_TOPIC | топик |
| POSTGRES_DSN | подключение к БД |
| BATCH_SIZE | размер батча |
| UNKNOWN_THRESHOLD | порог confidence |
| MODEL_ARTIFACTS_PATH | путь к модели |

---

## Минимальные требования

- Docker
- docker-compose
- 4 GB RAM
- Python 3.11 (опционально)

---

## Пример работы системы

1. Новость поступает в Kafka  
2. Сохраняется в БД  
3. Обрабатывается worker  
4. Получает тему  

Результат:

```json
{
  "topic_id": 2,
  "confidence": 0.76,
  "status": "processed"
}
```
