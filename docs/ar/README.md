[English](../../README.md) | [日本語](../../README_JP.md)

# بنية التوجيه التشابكي (SRA)

## 🎮 عروض توضيحية تفاعلية (دفاتر Jupyter)

لقد قمنا بإعداد دفاتر Jupyter Notebooks حيث يمكنك تجربة "استخدام العقل الخاص بمهمة محددة" و"المتانة" الخاصة بـ SRA بشكل تفاعلي في متصفحك مباشرةً.يمكنك تشغيلها في ثوانٍ على Google Colab، لذا يرجى تجربتها!

| # | العرض التوضيحي | الوصف | Colab |
|---|------|------|-------|
| 🟢 1 | [البداية السريعة SRA](../../notebooks/01_sra_quickstart_ar.ipynb) | بنية SRA الأساسية وتصور التوجيه | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_ar.ipynb) |
| 🔵 2 | [التعلم والتوجيه](../../notebooks/02_learning_and_routing_demo_ar.ipynb) | تعلم مهمة واحدة وتخصص التوجيه | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_ar.ipynb) |
| 🔴 3 | [التوجيه متعدد المهام](../../notebooks/03_multitask_routing_demo_ar.ipynb) ✨ | التعلم متعدد المهام وتبديل المشابك | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_ar.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_ar.ipynb) | فصل الإدراك والعمل في RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_ar.ipynb) |
| 🧠 5 | [تجربة الآفة](../../notebooks/05_lesion_experiment_demo_ar.ipynb) ✨ | إثبات النمذجة الوظيفية بتدمير المشابك | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_ar.ipynb) |
| 🔌 6 | [تجربة التبديل السريع](../../notebooks/06_hotswap_experiment_demo_ar.ipynb) | التبديل السريع الديناميكي وحدود تعلم الموجه | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_ar.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_ar.ipynb) | تكامل النموذج عبر Gumbel-Softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_ar.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_ar.ipynb) | بناء وتدريب Tiny LLM مع SRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_ar.ipynb) |
| 📚 9 | [LLM متعدد المجالات](../../notebooks/09_sra_llm_demo_multidomain_ar.ipynb) | التعلم المتزامن متعدد المجالات (Code/Math/Text) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_ar.ipynb) |
| 💻 10 | [التبديل السريع للمكونات](../../notebooks/10_hotswap_plugins_demo_ar.ipynb) | تبديل سريع بدون تعلم مسبق (صفر نسيان) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_ar.ipynb) |
| 🗑️ 11 | [حذف المشابك](../../notebooks/11_synapse_deletion_demo_ar.ipynb) | حذف المشابك الديناميكي (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_ar.ipynb) |
| 🧬 12 | [ظهور الخلايا العصبية الافتراضية](../../notebooks/12_virtual_neuron_experiment_ar.ipynb) | 5 مجالات × 5 مهام تكشف عن تكوين الخلايا العصبية الافتراضية المستقلة | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_ar.ipynb) |
| 🧠 13 | [المبادلة السريعة للخلايا العصبية الافتراضية](../../notebooks/13_virtual_neuron_hotswap_ar.ipynb) | التعلّم الآمن في تفاصيل الخلايا العصبية الافتراضية (تجميع الخلايا). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_ar.ipynb) |
| 🔬 14 | [مقارنة وحدة الحذف](../../notebooks/14_compare_deletion_units_ar.ipynb) | وحدة المشبك مقابل حذف وحدة الخلايا العصبية وتشابك المعرفة | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_ar.ipynb) |
| 📐 15 | [فرضية القدرة](../../notebooks/15_capacity_hypothesis_experiment_ar.ipynb) | قدرة المشبك مقابل عتبة التعلم الآمن | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_ar.ipynb) |
| 💤 16 | [منع التوجيه البطيء](../../notebooks/16_lazy_routing_prevention_experiment_ar.ipynb) | تشخيص وتخفيف كسل جهاز التوجيه | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_ar.ipynb) |
| 🔁 17 | [التوجيه الاحتياطي](../../notebooks/17_routing_fallback_experiment_ar.ipynb) | إعادة تعيين حركة المرور عندما يصبح المشبك غير متاح | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_ar.ipynb) |
| 🧩 18 | [نقاط الاشتباك العصبي المخصصة](../../notebooks/18_custom_synapses_ar.ipynb) | قاعدة بيانات المتجهات والمشابك الحاسبة غير القابلة للتدريب | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_ar.ipynb) |
| 🎯 19 | [توجيه صعب بدون إطلاق النار](../../notebooks/19_zero_shot_hard_routing_ar.ipynb) | فرض التوجيه عبرAlive_mask دون إعادة التدريب | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_ar.ipynb) |
| 🛠️ 20 | [التوجيه الدقيق](../../notebooks/20_routing_finetuning_ar.ipynb) | تعلم التوجيه الذاتي من خلال الضبط الدقيق لمجموعة البيانات الصغيرة | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_ar.ipynb) |
| 🧯 21 | [ضبط دقيق للنسيان](../../notebooks/21_finetuning_forgetting_check_ar.ipynb) | فحص النسيان الكارثي بعد ضبط التوجيه | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_ar.ipynb) |
| 🧪 22 | [التعايش الرمزي العصبي](../../notebooks/22_multi_synapse_hotswap_eval_ar.ipynb) | LLM plus Vector DB plus آلة حاسبة قائمة على القواعد في بنية واحدة | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_ar.ipynb) |
| 🦙 23 | [تكامل SRA LLM (TinyLlama)](../../notebooks/23_sra_llm_integration_ar.ipynb) | تكامل جهاز التوجيه SRA الأصلي مع TinyLlama (PoC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/23_sra_llm_integration_ar.ipynb) |
| 🏎️ 24 | [معيار هندسة جهاز التوجيه](../../notebooks/24_router_architecture_benchmark_ar.ipynb) | أجهزة توجيه معيارية أحادية المرحلة/متعددة المراحل/الرمز الأخير | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_ar.ipynb) |
| 🧰 25 | [هندسة اللوحة الأم](../../notebooks/25_integrated_heterogeneous_routing_ar.ipynb) | جهاز توجيه الرمز الأخير والتراجع الدلالي على المشابك العصبية غير المتجانسة | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_ar.ipynb) |
| 💬 26 | [SRA Chatbot التجريبي](../../notebooks/26_chatbot_demo_ar.ipynb) | واجهة مستخدم للدردشة العاملة تجمع بين نقاط الاشتباك العصبي LLM/Vector DB/الآلة الحاسبة | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_ar.ipynb) |


بنية التوجيه التشابكي (SRA) هي بنية شبكة عصبية ديناميكية ومتناثرة (sparse) ومعيارية جديدة مستوحاة من الدماغ البيولوجي (المشابك العصبية).

## 🎯 الدوافع
تحل SRA مشاكل النماذج الضخمة باستخدام **نهج متناثر: "استدعاء ودمج الوحدات الصغيرة (المشابك) الضرورية فقط ديناميكياً اعتماداً على المدخلات"**. هذا يسمح بتعلم مهام متعددة دون تداخل.

## 💡 الفكرة الأساسية

تحاول نماذج الذكاء الاصطناعي النموذجية (مثل Transformers) معالجة كل شيء باستخدام "دماغ" عملاق واحد. ومع ذلك، مع هذا النهج، يصبح عبء الحوسبة ثقيلًا جدًا في كل مرة يتم فيها جعل النموذج أكثر ذكاءً أو أكبر. لذلك، يعتمد SRA نظامًا حيث **يتم إعداد العديد من "أدمغة الخبراء الصغيرة (والتي يطلق عليها SRA اسم 'synapses')"، ويتم استدعاء الخبراء الضروريين فقط بناءً على المشكلة المطروحة**.

المفتاح هنا هو الآلية التي تقرر "أي خبير يجب استدعاؤه". يحتوي SRA على "موجه (دليل)"، يختار على الفور الخبير الذي يبدو أكثر قدرة من خلال النظر إلى بيانات الإدخال. بينما يصبح كل خبير أكثر ذكاءً (يتعلم)، يتعلم هذا الموجه في نفس الوقت "من هو الشخص الصحيح للاختيار"، وينمو ليتمكن من إجراء التخصيصات المثلى تلقائيًا.

## 🧠 نظرة عامة على البنية
1. **المشبك (Synapse):** وحدات حسابية مستقلة.
2. **الموجه (Router):** يختار أفضل `Top-k` مشابك ديناميكياً.
3. **مساحة المشبك:** تنظيم ذاتي حسب "التشابه الوظيفي".
4. **قاعدة التعلم المحلي:** تستخدم قواعد محلية لتحقيق التوازن.


---

### 🔌 6. تجربة التبديل السريع الديناميكي للتشابك وحدود تعلم الموجه
**الملف:** [`06_hotswap_experiment_demo_ar.ipynb`](./06_hotswap_experiment_demo_ar.ipynb)

يوضح القوة الحقيقية لـ SRA: "الإضافة والإزالة الديناميكية للتشابكات كمكونات إضافية (Hot-Swap)".
نجري تجربة حيث يتم دمج تشابك خاص باللغة الإسبانية في نموذج ترجمة فرنسي/ألماني قيد التشغيل.
في دفتر الملاحظات هذا، ستتعلم **الأهمية الحاسمة لمشاركة وتجميد مساحة معرفة النموذج الأساسي (طبقات التضمين/الانتباه، إلخ)** لإنشاء تبديل سريع. في الوقت نفسه، ستواجه **أكبر عائق لـ SRA (مشكلة التدرج المتلاشي)**: التوجيه الصعب القياسي (Top-k) لا يمكنه تعلم (تفاضل) توجيه التشابكات المضافة بأثر رجعي. يعمل هذا القيد كتمهيد بالغ الأهمية لقسم "Gumbel-Softmax (Super Router)" اللاحق.

[![فتح في Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_ar.ipynb)

---

### 👑 7. تكامل النموذج عبر Super Router و Gumbel-Softmax
**الملف:** [`07_super_router_gumbel_demo_ar.ipynb`](./07_super_router_gumbel_demo_ar.ipynb)

نبني "موجه فائق" يجمع نماذج متخصصة متعددة (نموذج FR/DE ونموذج ES) ويوجه المعالجة ديناميكيًا بناءً على الإدخال.
يوضح هذا مشكلة "التوجيه الكسول" للتوجيه المرن البسيط ويوضح كيف يحقق استخدام Gumbel-Softmax **توجيهًا صلبًا مثاليًا**، مما يقلل حساب النموذج غير الضروري بنسبة 100٪.

[![فتح في Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_ar.ipynb)


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🧪 التجارب والتحليل
- [التعلم متعدد المهام وتحليل التوجيه](./routing_analysis_algorithmic.md)
- [تحليل التوجيه في نمذجة اللغة عبر المجالات](./routing_analysis_language.md)
- [تحليل التوجيه في الترجمة متعددة اللغات (الإنجليزية / الفرنسية / اليابانية) والتعميم الخالي من الأمثلة (Zero-Shot)](../dev/multilingual_translation_routing_analysis.md)
  - تقرير رائع يوضح كيف يقوم SRA بتعيين وحدات ترجمة مختلفة تلقائيًا بناءً على البنية النحوية (SVO مقابل SOV). والأكثر إثارة للدهشة، عند طلب ترجمة زوج لغوي لم يتم تعلمه، فإنه يستخدم اللغة الإنجليزية بوعي باطني كـ "لغة وسيطة" لحل المشكلة!
- [الفصل الكامل بين الإدراك والسياسة في Decision Transformer (التعلم المعزز)](../dev/decision_transformer_routing_analysis.md)
  - جعلنا SRA يلعب لعبة. لقد اكتشف هيكلًا معياريًا مذهلاً بمفرده: فهو يستخدم وحدة "الرؤية" نفسها تمامًا لإدراك البيئة في جميع المهام، ولكنه يتحول إلى وحدات "دماغ" مختلفة تمامًا اعتمادًا على ما إذا كان بحاجة إلى العثور على كنز أو الهروب.
- [التحقق من الترجمة العملية متعددة اللغات باستخدام SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - تقرير يوضح أنه من خلال توسيع SRA إلى بنية Encoder-Decoder والتدريب لمدة 30,000 خطوة على مجموعة نصوص حقيقية (opus100)، فإنه يمكنه ترجمة التعبيرات العملية مثل "Merci beaucoup." و "Good morning." بدقة BLEU=1.0. أدى إدخال الاهتمام المتبادل (Cross-Attention) إلى قفزة من Decoder-only (BLEU=0) إلى متوسط عام BLEU يبلغ 0.27، وحقق دقة عملية تقريبية تبلغ BLEU=0.56 في الاتجاه من الفرنسية إلى الإنجليزية.

---

### 📖 8. عرض SRA LLM (شكسبير)
**File:** [`08_sra_llm_demo_shakespeare_ar.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_ar.ipynb)

هذا برنامج تعليمي يستخدم بيانات شكسبير على نطاق صغير لتدريب SRA كنموذج توليدي خاص بوحدة فك التشفير (LLM). بعد التعلم، يتم استخدام الخريطة الحرارية لتصور المشبك الذي تم تمريره من خلال كل رمز مميز للنص الذي تم إنشاؤه.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_ar.ipynb)

---

### 🌐 9. عرض SRA متعدد المجالات LLM (الرمز، الرياضيات، النص)
**File:** [`09_sra_llm_demo_multidomain_ar.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_ar.ipynb)

اختبر تخصص SRA في "التعلم المتزامن لمجالات متعددة (الرمز والرياضيات والنص)" في برنامج LLM صغير الحجم. يمكنك التحقق من كيفية قيام النموذج تلقائيًا بتقسيم (تخصيص) المشابك العصبية بناءً على البيانات.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_ar.ipynb)

---

### 💻 10. التبديل السريع للمكونات الإضافية العملية (اللقطة الصفرية)
**File:** [`10_hotswap_plugins_demo_ar.ipynb`](../../notebooks/10_hotswap_plugins_demo_ar.ipynb)

سنعرض سير العمل الذي تتعلم فيه فرق التطوير المتعددة بشكل مستقل المكونات الإضافية لـ "الكود" و"الرياضيات" و"دمجها فعليًا (التبديل السريع)" في النموذج الأساسي لبيئة الإنتاج بعد حدوثها. لقد ثبت أنه حتى بعد الدمج، تكون خسائر جميع المجالات هي نفسها تمامًا كما كانت أثناء التعلم المستقل (صفر نسيان).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_ar.ipynb)

---

### 🗑️ 11. الحذف التشابكي الديناميكي
**File:** [`11_synapse_deletion_demo_ar.ipynb`](../../notebooks/11_synapse_deletion_demo_ar.ipynb)

نعرض وظيفة SRA، "حذف المشبك العصبي". يمكنك تجربة كل من ``إزالة المكونات الإضافية (pop_synapses)''، والتي تحذف فعليًا المشابك العصبية التي تمت إضافتها لاحقًا من النهاية، و``تطهير مجال معين (clear_synapses)''، الذي يقوم بمسح وتعطيل المشابك العصبية التي لم تتم مشاركتها بشكل آمن.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_ar.ipynb)

