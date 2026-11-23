# energy-to-quality

Небольшой воспроизводимый бенчмарк/CLI для измерения **энергии до достижения целевого качества** при обучении нейросети на CPU.

Главная метрика: **`Joules_to_Q_min`** — сколько джоулей потрачено до первого достижения порога качества `Q_min` (например, test accuracy >= 0.95). Энергия измеряется через Intel RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`), не через программный прокси.

> Статус: публичный прототип для воспроизведения и проверки. Результаты привязаны к конкретному железу, задаче и протоколу; это не универсальный физический минимум.

**Для быстрой презентации/проверки:** см. [ONE_PAGER.md](ONE_PAGER.md).

## Что внутри

- `scripts/run_champion.py` — воспроизводимый запуск подтверждённых конфигураций (MNIST 12k / full MNIST).
- `scripts/run_proof.py` — честное сравнение baseline vs champion: чередование, повторы, idle-power, парная статистика, checkpoint JSON.
- `results/` — сохранённые JSON-артефакты прогонов.
- `RESULTS.md` — сводка подтверждённых результатов и ограничений.
- `METHODOLOGY.md` — протокол измерения и правила честного сравнения.

## Быстрый старт

Требования: Linux, Intel CPU с RAPL package-0, Python 3.10+, доступ на чтение `energy_uj`.

```bash
# через uv (рекомендуется)
uv run --with numpy --with scikit-learn --with numba -- python scripts/run_champion.py --task mnist12k --q-min 0.94 --repeats 5 --out results/champion_12k.json

uv run --with numpy --with scikit-learn --with numba -- python scripts/run_champion.py --task mnist-full --q-min 0.95 --repeats 5 --out results/champion_full.json

# честное proof-сравнение const_lr vs one-cycle на full MNIST
uv run --with numpy --with scikit-learn --with numba -- python scripts/run_proof.py --repeats 10 --out results/proof.json
```

## Подтверждённые конфигурации (этот репозиторий)

| Задача | Q_min | Конфиг | Результат | Статус |
|---|---:|---|---|---|
| MNIST 12k (2 split) | 0.94 | `784-64-10`, batch 64, 40ep, max_lr 0.08, one-cycle, Numba, early stop | ~29.6–33.6 J медиана, 5/5 на split 42 и 123 | подтверждён |
| Full MNIST 60k/10k (fixed split) | 0.95 | `784-128-10`, batch 128, 20ep, max_lr 0.08, one-cycle, Numba, early stop | 5/5 по seed на одном split; энергия зависит от фона | подтверждён как конфиг |
| Full MNIST, baseline vs one-cycle | 0.95 | тот же split/Q/model, чередование, 10/20 повторов | one-cycle лучше по медиане/среднему и в большинстве пар; IQR перекрываются в v36/v37b | сильное практическое свидетельство |

Подробности и оговорки: [RESULTS.md](RESULTS.md), [METHODOLOGY.md](METHODOLOGY.md).

## Ограничения (важно)

- **RAPL package-0** измеряет энергию всего CPU-пакета, не только обучения; фон системы влияет на абсолютные джоули.
- Результаты **не переносятся автоматически** на другие CPU/GPU/TPU, датасеты, модели, драйверы, режимы питания.
- `5/5` или `10/10` всегда нужно читать вместе с тем, что повторялось: разные seed обучения на одном split или разные split.
- Это практический инженерный рецепт/протокол, **не** доказательство физического минимума и **не** “революция в обучении ИИ”.

## Лицензия

MIT — см. [LICENSE](LICENSE).
