[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Интерактивные демонстрации (блокноты Jupyter)

Мы подготовили блокноты Jupyter, где вы можете в интерактивном режиме испытать «использование мозга для конкретных задач» и «надежность» SRA прямо в своем браузере.Вы можете запустить их в Google Colab за считанные секунды, поэтому попробуйте!

| # | Демо | Описание | Colab |
|---|------|----------|-------|
| 🟢 1 | [SRA Quickstart](../../notebooks/01_sra_quickstart_ru.ipynb) | Базовая структура SRA и визуализация маршрутизации | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_ru.ipynb) |
| 🔵 2 | [Обучение и маршрутизация](../../notebooks/02_learning_and_routing_demo_ru.ipynb) | Обучение одной задаче и специализация маршрутизации | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_ru.ipynb) |
| 🔴 3 | [Многозадачная маршрутизация](../../notebooks/03_multitask_routing_demo_ru.ipynb) ✨ | Многозадачное обучение и переключение синапсов | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_ru.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_ru.ipynb) | Разделение восприятия и действия в RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_ru.ipynb) |
| 🧠 5 | [Эксперимент с повреждением](../../notebooks/05_lesion_experiment_demo_ru.ipynb) ✨ | Доказательство функциональной модульности разрушением синапсов | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_ru.ipynb) |
| 🔌 6 | [Эксперимент Hot-Swap](../../notebooks/06_hotswap_experiment_demo_ru.ipynb) | Динамическая горячая замена синапсов и пределы обучения | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_ru.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_ru.ipynb) | Интеграция моделей через Gumbel-Softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_ru.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_ru.ipynb) | Построение и обучение Tiny LLM с SRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_ru.ipynb) |
| 📚 9 | [Мультидоменный LLM](../../notebooks/09_sra_llm_demo_multidomain_ru.ipynb) | Одновременное мультидоменное обучение (Code/Math/Text) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_ru.ipynb) |
| 💻 10 | [Плагин Hot-Swap](../../notebooks/10_hotswap_plugins_demo_ru.ipynb) | Горячая замена zero-shot (нулевое забывание) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_ru.ipynb) |
| 🗑️ 11 | [Удаление синапсов](../../notebooks/11_synapse_deletion_demo_ru.ipynb) | Динамическое удаление синапсов (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_ru.ipynb) |
| 🧬 12 | [Появление виртуального нейрона](../../notebooks/12_virtual_neuron_experiment_ru.ipynb) | 5 доменов x 5 задач демонстрируют формирование автономных виртуальных нейронов | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_ru.ipynb) |
| 🧠 13 | [Горячая замена виртуальных нейронов](../../notebooks/13_virtual_neuron_hotswap_ru.ipynb) | Безопасное отучение на уровне детализации виртуального нейрона (сборки клеток) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_ru.ipynb) |
| 🔬 14 | [Сравнение единиц удаления](../../notebooks/14_compare_deletion_units_ru.ipynb) | Удаление синаптической единицы против нейронной единицы и запутывание знаний | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_ru.ipynb) |
| 📐 15 | [Гипотеза емкости](../../notebooks/15_capacity_hypothesis_experiment_ru.ipynb) | Емкость синапса в сравнении с порогом безопасного отучения | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_ru.ipynb) |
| 💤 16 | [Предотвращение ленивой маршрутизации](../../notebooks/16_lazy_routing_prevention_experiment_ru.ipynb) | Диагностика и устранение ленивости маршрутизатора | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_ru.ipynb) |
| 🔁 17 | [Резервная маршрутизация](../../notebooks/17_routing_fallback_experiment_ru.ipynb) | Переназначить трафик, когда синапс становится недоступным | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_ru.ipynb) |
| 🧩 18 | [Пользовательские синапсы](../../notebooks/18_custom_synapses_ru.ipynb) | Необучаемые синапсы Vector DB и Calculator | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_ru.ipynb) |
| 🎯 19 | [Жесткая маршрутизация с нулевым выстрелом](../../notebooks/19_zero_shot_hard_routing_ru.ipynb) | Принудительная маршрутизация через разрешенную маску без переобучения | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_ru.ipynb) |
| 🛠️ 20 | [Точная настройка маршрутизации](../../notebooks/20_routing_finetuning_ru.ipynb) | Автономное обучение маршрутизации посредством тонкой настройки небольшого набора данных | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_ru.ipynb) |
| 🧯 21 | [Точная настройка проверки забывания](../../notebooks/21_finetuning_forgetting_check_ru.ipynb) | Проверка катастрофического забывания после тонкой настройки маршрутизации | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_ru.ipynb) |
| 🧪 22 | [Нейро-символическое сосуществование](../../notebooks/22_multi_synapse_hotswap_eval_ru.ipynb) | LLM плюс Vector DB плюс калькулятор на основе правил на одной архитектуре | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_ru.ipynb) |
| 🦙 23 | [Интеграция SRA LLM (TinyLlama)](../../notebooks/23_sra_llm_integration_ru.ipynb) | Интеграция встроенного SRA-маршрутизатора с TinyLlama (PoC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/23_sra_llm_integration_ru.ipynb) |
| 🏎️ 24 | [Тест архитектуры маршрутизатора](../../notebooks/24_router_architecture_benchmark_ru.ipynb) | Тестирование одноэтапных/многоступенчатых маршрутизаторов/маршрутизаторов с последним токеном | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_ru.ipynb) |
| 🧰 25 | [Архитектура материнской платы](../../notebooks/25_integrated_heterogeneous_routing_ru.ipynb) | Маршрутизатор последнего токена и семантический резервный вариант для гетерогенных синапсов | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_ru.ipynb) |
| 💬 26 | [Демонстрация чат-бота SRA](../../notebooks/26_chatbot_demo_ru.ipynb) | Рабочий интерфейс чата, объединяющий синапсы LLM/Vector DB/калькулятора | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_ru.ipynb) |


Synaptic Routing Architecture (SRA) — это новая динамичная, разреженная (sparse) и модульная архитектура нейронной сети, вдохновленная биологическим мозгом (синапсами).

## 🎯 Мотивация
SRA решает проблемы огромных монолитных моделей с помощью **разреженного подхода: "динамический вызов и комбинирование только необходимых модулей (синапсов) в зависимости от входных данных"**. Это позволяет обучать несколько задач без "катастрофического забывания".

## 💡 Основная идея

Обычные модели ИИ (такие как Трансформеры) пытаются обрабатывать всё с помощью одного гигантского "мозга". Однако при таком подходе вычислительная нагрузка становится слишком большой каждый раз, когда модель делают умнее или больше. Поэтому SRA применяет систему, в которой **подготовлено множество "маленьких мозгов экспертов (которые в SRA называются 'синапсами')", и в зависимости от решаемой задачи вызываются только нужные эксперты**.

Ключевым моментом здесь является механизм, который решает, "какого эксперта вызвать". В SRA есть "маршрутизатор (проводник)", который мгновенно выбирает наиболее подходящего эксперта, глядя на входные данные. По мере того как каждый эксперт становится умнее (учится), этот маршрутизатор одновременно учится, "кого правильно выбирать", развиваясь так, чтобы автоматически делать оптимальные назначения.

## 🧠 Обзор архитектуры
1. **Синапс:** Независимые вычислительные единицы.
2. **Маршрутизатор (Router):** Выбирает `Top-k` лучших синапсов.
3. **Синаптическое пространство:** Самоорганизация синапсов по "функциональному сходству".
4. **Локальное правило обучения:** Использование локальных правил для балансировки нагрузки.


---

### 🔌 6. Эксперимент с динамической горячей заменой синапсов и пределы обучения маршрутизатора
**Файл:** [`06_hotswap_experiment_demo_ru.ipynb`](./06_hotswap_experiment_demo_ru.ipynb)

Демонстрирует истинную мощь SRA: «динамическое добавление и удаление синапсов в качестве плагинов (Hot-Swap)».
Мы проводим эксперимент, в котором синапс, специфичный для испанского языка, сливается с работающей моделью перевода с французского/немецкого.
В этом блокноте вы узнаете о **критической важности совместного использования и замораживания пространства знаний базовой модели (слои встраивания/внимания и т. д.)** для установки горячей замены. В то же время вы столкнетесь с **главным препятствием SRA (проблема исчезающего градиента)**: стандартная жесткая маршрутизация (Top-k) не может задним числом изучить (дифференцировать) маршрутизацию добавленных синапсов. Это ограничение служит критическим предзнаменованием для последующего раздела «Gumbel-Softmax (Super Router)».

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_ru.ipynb)

---

### 👑 7. Интеграция моделей с помощью Super Router и Gumbel-Softmax
**Файл:** [`07_super_router_gumbel_demo_ru.ipynb`](./07_super_router_gumbel_demo_ru.ipynb)

Мы создаем «Супермаршрутизатор», который объединяет несколько специализированных моделей (модель FR/DE и модель ES) и динамически маршрутизирует обработку на основе ввода.
Это демонстрирует проблему «ленивой маршрутизации» простой мягкой маршрутизации и показывает, как использование Gumbel-Softmax обеспечивает **идеальную жесткую маршрутизацию**, на 100% сокращая ненужные вычисления модели.

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_ru.ipynb)


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🧪 Эксперименты и анализ
- [Анализ многозадачного обучения и маршрутизации](./routing_analysis_algorithmic.md)
- [Анализ маршрутизации в кросс-доменном языковом моделировании](./routing_analysis_language.md)
- [Анализ маршрутизации в многоязычном переводе (Англ / Фр / Яп) и Zero-Shot обобщение](../dev/multilingual_translation_routing_analysis.md)
  - Захватывающий отчет, показывающий, как SRA автоматически назначает различные модули перевода в зависимости от грамматической структуры (SVO или SOV). Что еще более удивительно, при переводе неизученной языковой пары он бессознательно использует английский язык в качестве "языка-посредника"!
- [Полное разделение восприятия и стратегии в Decision Transformer (Обучение с подкреплением)](../dev/decision_transformer_routing_analysis.md)
  - Мы дали SRA возможность поиграть в игру. Он самостоятельно открыл невероятную модульную структуру: он использует абсолютно один и тот же модуль "зрения" для восприятия среды во всех задачах, но переключается на совершенно разные модули "мозга" в зависимости от того, нужно ли ему найти сокровище или убежать.
- [Проверка практического многоязычного перевода с использованием SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Отчет, демонстрирующий, что за счет расширения SRA до архитектуры Encoder-Decoder и обучения в течение 30 000 шагов на реальном корпусе (opus100) модель может переводить практические выражения, такие как "Merci beaucoup." и "Good morning.", с BLEU=1.0. Внедрение Cross-Attention привело к скачку от Decoder-only (BLEU=0) до общего среднего BLEU 0,27 и достигло почти практической точности BLEU=0,56 в направлении FR→EN.

---

### 📖 8. Демо SRA LLM (Шекспир)
**File:** [`08_sra_llm_demo_shakespeare_ru.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_ru.ipynb)

Это учебное пособие, в котором используются мелкомасштабные данные Шекспира для обучения SRA как генеративной модели (LLM), специфичной для декодера. После обучения используется тепловая карта для визуализации того, через какой синапс прошел каждый токен сгенерированного текста.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_ru.ipynb)

---

### 🌐 9. Демонстрация многодоменного LLM SRA (код, математика, текст)
**File:** [`09_sra_llm_demo_multidomain_ru.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_ru.ipynb)

Испытайте специальность SRA по «одновременному изучению нескольких предметных областей (код, математика, текст)» в небольшом LLM. Вы можете проверить, как модель автоматически разделяет (специализирует) синапсы на основе данных.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_ru.ipynb)

---

### 💻 10. Практичная горячая замена плагина (Zero-Shot)
**File:** [`10_hotswap_plugins_demo_ru.ipynb`](../../notebooks/10_hotswap_plugins_demo_ru.ipynb)

Мы продемонстрируем рабочий процесс, в котором несколько команд разработчиков независимо изучают плагины для «кода» и «математики» и «физически объединяют (горячую замену)» их в базовую модель производственной среды постфактум. Доказано, что даже после слияния потери всех доменов точно такие же, как и при независимом обучении (Zero Forgetting).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_ru.ipynb)

---

### 🗑️ 11. Динамическое синаптическое удаление
**File:** [`11_synapse_deletion_demo_ru.ipynb`](../../notebooks/11_synapse_deletion_demo_ru.ipynb)

Мы демонстрируем функцию SRA, «удаление синапса». Вы можете испытать как «удаление плагинов (pop_synapses)», которое физически удаляет синапсы, добавленные позже, с конца, так и «очистку определенного домена (clear_synapses)», которая безопасно очищает и отключает синапсы, которые не являются общими.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_ru.ipynb)

