[English](../../README.md) | [日本語](../../README_JP.md)

# सिनैप्टिक राउटिंग आर्किटेक्चर (SRA)

## 🎮 इंटरएक्टिव डेमो (ज्यूपिटर नोटबुक)

हमने ज्यूपिटर नोटबुक तैयार की है जहां आप सीधे अपने ब्राउज़र में एसआरए के "कार्य-विशिष्ट मस्तिष्क उपयोग" और "मजबूती" का अनुभव कर सकते हैं।आप उन्हें Google Colab पर कुछ ही सेकंड में चला सकते हैं, इसलिए कृपया उन्हें आज़माएँ!

| # | डेमो | विवरण | Colab |
|---|------|-------|-------|
| 🟢 1 | [SRA क्विकस्टार्ट](../../notebooks/01_sra_quickstart_hi.ipynb) | मूल SRA संरचना और रूटिंग विज़ुअलाइज़ेशन | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_hi.ipynb) |
| 🔵 2 | [सीखना और रूटिंग](../../notebooks/02_learning_and_routing_demo_hi.ipynb) | एकल-कार्य शिक्षा और रूटिंग विशेषज्ञता | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_hi.ipynb) |
| 🔴 3 | [मल्टीटास्क रूटिंग](../../notebooks/03_multitask_routing_demo_hi.ipynb) ✨ | मल्टीटास्क शिक्षा और सिनैप्स स्विचिंग | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_hi.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_hi.ipynb) | RL में धारणा और क्रिया का पृथक्करण | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_hi.ipynb) |
| 🧠 5 | [लीजन प्रयोग](../../notebooks/05_lesion_experiment_demo_hi.ipynb) ✨ | सिनैप्स को नष्ट करके कार्यात्मक मॉड्यूलरिटी का प्रमाण | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_hi.ipynb) |
| 🔌 6 | [हॉट-स्वैप प्रयोग](../../notebooks/06_hotswap_experiment_demo_hi.ipynb) | गतिशील सिनैप्टिक हॉट-स्वैप और राउटर शिक्षा सीमाएँ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_hi.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_hi.ipynb) | Gumbel-Softmax के माध्यम से मॉडल एकीकरण | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_hi.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_hi.ipynb) | SRA के साथ Tiny LLM बनाएं और प्रशिक्षित करें | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_hi.ipynb) |
| 📚 9 | [मल्टीडोमेन LLM](../../notebooks/09_sra_llm_demo_multidomain_hi.ipynb) | बहु-डोमेन (Code/Math/Text) एक साथ सीखना | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_hi.ipynb) |
| 💻 10 | [प्लगइन हॉट-स्वैप](../../notebooks/10_hotswap_plugins_demo_hi.ipynb) | ज़ीरो-शॉट हॉट-स्वैप (शून्य विनाशकारी भूलना) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_hi.ipynb) |
| 🗑️ 11 | [सिनैप्स हटाना](../../notebooks/11_synapse_deletion_demo_hi.ipynb) | गतिशील सिनैप्स हटाना (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_hi.ipynb) |
| 🧬 12 | [आभासी न्यूरॉन उद्भव](../../notebooks/12_virtual_neuron_experiment_hi.ipynb) | 5 डोमेन x 5 कार्य स्वायत्त आभासी न्यूरॉन गठन को प्रकट करते हैं | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_hi.ipynb) |
| 🧠 13 | [वर्चुअल न्यूरॉन हॉट-स्वैप](../../notebooks/13_virtual_neuron_hotswap_hi.ipynb) | वर्चुअल न्यूरॉन (सेल असेंबली) ग्रैन्युलैरिटी पर सुरक्षित अनसीखना | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_hi.ipynb) |
| 🔬 14 | [विलोपन इकाई तुलना](../../notebooks/14_compare_deletion_units_hi.ipynb) | सिनैप्स-यूनिट बनाम न्यूरॉन-यूनिट विलोपन और ज्ञान उलझाव | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_hi.ipynb) |
| 📐 15 | [क्षमता परिकल्पना](../../notebooks/15_capacity_hypothesis_experiment_hi.ipynb) | सिनैप्स क्षमता बनाम सुरक्षित-अनसीखने की सीमा | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_hi.ipynb) |
| 💤 16 | [आलसी रूटिंग निवारण](../../notebooks/16_lazy_routing_prevention_experiment_hi.ipynb) | राउटर आलस्य का निदान करें और उसे कम करें | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_hi.ipynb) |
| 🔁 17 | [रूटिंग फ़ॉलबैक](../../notebooks/17_routing_fallback_experiment_hi.ipynb) | जब सिनैप्स अनुपलब्ध हो जाए तो ट्रैफ़िक पुनः असाइन करें | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_hi.ipynb) |
| 🧩 18 | [कस्टम सिनैप्स](../../notebooks/18_custom_synapses_hi.ipynb) | गैर-प्रशिक्षित वेक्टर डीबी और कैलकुलेटर सिनैप्स | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_hi.ipynb) |
| 🎯 19 | [जीरो-शॉट हार्ड रूटिंग](../../notebooks/19_zero_shot_hard_routing_hi.ipynb) | पुनः प्रशिक्षण के बिना अनुमति_मास्क के माध्यम से रूटिंग को बाध्य करें | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_hi.ipynb) |
| 🛠️ 20 | [रूटिंग फ़ाइन-ट्यूनिंग](../../notebooks/20_routing_finetuning_hi.ipynb) | छोटे-डेटासेट फ़ाइन-ट्यूनिंग के माध्यम से स्वायत्त रूटिंग सीखना | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_hi.ipynb) |
| 🧯 21 | [भूलने की जाँच को ठीक करना](../../notebooks/21_finetuning_forgetting_check_hi.ipynb) | रूटिंग फ़ाइन-ट्यूनिंग के बाद भयावह भूल की जाँच करें | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_hi.ipynb) |
| 🧪 22 | [तंत्रिका-प्रतीकात्मक सह-अस्तित्व](../../notebooks/22_multi_synapse_hotswap_eval_hi.ipynb) | एलएलएम प्लस वेक्टर डीबी प्लस एक आर्किटेक्चर पर नियम-आधारित कैलकुलेटर | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_hi.ipynb) |
| 🦙 23 | [एसआरए एलएलएम एकीकरण (टिनीलामा)](../../notebooks/nb23_sra_llm_integration_hi.ipynb) | TinyLlama (PoC) के साथ नेटिव SRA-राउटर एकीकरण | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/nb23_sra_llm_integration_hi.ipynb) |
| 🏎️ 24 | [राउटर आर्किटेक्चर बेंचमार्क](../../notebooks/24_router_architecture_benchmark_hi.ipynb) | बेंचमार्क सिंगल-स्टेज/मल्टी-स्टेज/लास्ट-टोकन राउटर | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_hi.ipynb) |
| 🧰 25 | [मदरबोर्ड आर्किटेक्चर](../../notebooks/25_integrated_heterogeneous_routing_hi.ipynb) | अंतिम-टोकन राउटर और विषम सिनैप्स पर सिमेंटिक फ़ॉलबैक | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_hi.ipynb) |
| 💬 26 | [एसआरए चैटबॉट डेमो](../../notebooks/26_chatbot_demo_hi.ipynb) | एलएलएम/वेक्टर डीबी/कैलकुलेटर सिनैप्स को मिलाकर कार्यशील चैट यूआई | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_hi.ipynb) |


सिनैप्टिक राउटिंग आर्किटेक्चर (SRA) जैविक मस्तिष्क (सिनैप्स) से प्रेरित एक नया गतिशील, विरल (sparse) और मॉड्यूलर न्यूरल नेटवर्क आर्किटेक्चर है।

## 🎯 प्रेरणा
SRA बड़े और स्थिर मॉडलों की समस्याओं को **एक विरल दृष्टिकोण के माध्यम से हल करता है: "इनपुट के आधार पर केवल आवश्यक छोटे मॉड्यूल (सिनैप्स) को गतिशील रूप से कॉल करना और जोड़ना"**। यह बिना किसी हस्तक्षेप के कई कार्यों को सीखने की अनुमति देता है।

## 💡 मूल विचार

सामान्य AI मॉडल (जैसे ट्रांसफॉर्मर) एक ही विशाल "मस्तिष्क" का उपयोग करके सब कुछ संसाधित करने का प्रयास करते हैं। हालांकि, इस दृष्टिकोण के साथ, हर बार मॉडल को अधिक स्मार्ट या बड़ा बनाने पर कम्प्यूटेशनल बोझ बहुत अधिक हो जाता है। इसलिए, SRA एक ऐसी प्रणाली अपनाता है जहाँ **कई "छोटे विशेषज्ञ मस्तिष्क (जिन्हें SRA में 'सिनैप्स' कहा जाता है)" तैयार किए जाते हैं, और समस्या के आधार पर केवल आवश्यक विशेषज्ञों को ही बुलाया जाता है**।

यहाँ मुख्य बात वह तंत्र है जो यह तय करता है कि "किस विशेषज्ञ को बुलाना है।" SRA में एक "राउटर (गाइड)" होता है, जो इनपुट डेटा को देखकर तुरंत सबसे सक्षम विशेषज्ञ का चयन करता है। जैसे-जैसे प्रत्येक विशेषज्ञ होशियार होता है (सीखता है), यह राउटर एक साथ यह भी सीखता है कि "किसे चुनना सही है," जिससे वह स्वचालित रूप से इष्टतम आवंटन करने में सक्षम हो जाता है।

## 🧠 आर्किटेक्चर का अवलोकन
1. **सिनैप्स:** स्वतंत्र गणना इकाइयाँ।
2. **राउटर:** इनपुट के आधार पर `Top-k` सर्वश्रेष्ठ सिनैप्स का चयन करता है।
3. **सिनैप्स स्पेस:** "कार्यात्मक समानता" द्वारा स्व-व्यवस्थित होता है।
4. **स्थानीय शिक्षण नियम:** भार संतुलन के लिए स्थानीय नियमों का उपयोग करता है।


---

### 🔌 6. डायनेमिक सिनैप्टिक हॉट-स्वैप प्रयोग और राउटर लर्निंग सीमाएँ
**फ़ाइल:** [`06_hotswap_experiment_demo_hi.ipynb`](./06_hotswap_experiment_demo_hi.ipynb)

SRA की वास्तविक शक्ति को प्रदर्शित करता है: "प्लगइन (हॉट-स्वैप) के रूप में सिनैप्स को गतिशील रूप से जोड़ना और हटाना"।
हम एक प्रयोग करते हैं जहां एक स्पेनिश-विशिष्ट सिनैप्स को चल रहे फ्रेंच/जर्मन अनुवाद मॉडल में मिला दिया जाता है।
इस नोटबुक में, आप हॉट-स्वैप स्थापित करने के लिए **बेस मॉडल के ज्ञान स्थान (एम्बेडिंग/अटेंशन लेयर्स, आदि) को साझा करने और फ़्रीज़ करने के महत्वपूर्ण महत्व** को जानेंगे। साथ ही, आप SRA की **सबसे बड़ी बाधा (वैनिशिंग ग्रेडिएंट समस्या)** का सामना करेंगे: मानक हार्ड रूटिंग (Top-k) जोड़े गए सिनैप्स की रूटिंग को पूर्वव्यापी रूप से सीख (भेद) नहीं सकती है। यह सीमा अगले "Gumbel-Softmax (Super Router)" अनुभाग के लिए एक महत्वपूर्ण पूर्वाभास के रूप में कार्य करती है।

[![Colab में खोलें](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_hi.ipynb)

---

### 👑 7. सुपर राउटर और गम्बेल-सॉफ्टमैक्स के माध्यम से मॉडल एकीकरण
**फ़ाइल:** [`07_super_router_gumbel_demo_hi.ipynb`](./07_super_router_gumbel_demo_hi.ipynb)

हम एक "सुपर राउटर" बनाते हैं जो कई विशेष मॉडलों (एक एफआर/डीई मॉडल और एक ईएस मॉडल) को बंडल करता है और इनपुट के आधार पर प्रोसेसिंग को गतिशील रूप से रूट करता है।
यह सरल सॉफ्ट राउटिंग की "आलसी राउटिंग" समस्या को प्रदर्शित करता है और दिखाता है कि कैसे गम्बेल-सॉफ्टमैक्स का उपयोग करके **सही हार्ड राउटिंग** प्राप्त होती है, जिससे अनावश्यक मॉडल गणना में 100% की कटौती होती है।

[![Colab में खोलें](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_hi.ipynb)


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🧪 प्रयोग और विश्लेषण
- [मल्टीटास्क लर्निंग और राउटिंग विश्लेषण](./routing_analysis_algorithmic.md)
- [क्रॉस-डोमेन लैंग्वेज मॉडलिंग में राउटिंग विश्लेषण](./routing_analysis_language.md)
- [बहुभाषी अनुवाद (अंग्रेजी / फ्रेंच / जापानी) में रूटिंग विश्लेषण और ज़ीरो-शॉट सामान्यीकरण](../dev/multilingual_translation_routing_analysis.md)
  - एक दिलचस्प रिपोर्ट जो दिखाती है कि कैसे SRA व्याकरणिक संरचना (SVO बनाम SOV) के आधार पर स्वचालित रूप से विभिन्न अनुवाद मॉड्यूल असाइन करता है। इससे भी अधिक आश्चर्यजनक बात यह है कि जब किसी अनसीखी भाषा जोड़ी का अनुवाद करने के लिए कहा जाता है, तो यह अवचेतन रूप से अंग्रेजी का उपयोग "पिवट भाषा" के रूप में करता है!
- [डिसीजन ट्रांसफार्मर (सुदृढीकरण शिक्षा) में धारणा और नीति का पूर्ण पृथक्करण](../dev/decision_transformer_routing_analysis.md)
  - हमने SRA को एक गेम खेलने की क्षमता दी। इसने अपने आप एक अविश्वसनीय मॉड्यूलर संरचना की खोज की: यह सभी कार्यों में पर्यावरण को देखने के लिए बिल्कुल उसी "विज़न" मॉड्यूल का उपयोग करता है, लेकिन खजाना खोजने या भागने की आवश्यकता के आधार पर पूरी तरह से अलग "ब्रेन" मॉड्यूल में बदल जाता है।
- [SRA एनकोडर-डिकोडर का उपयोग करके व्यावहारिक बहुभाषी अनुवाद का सत्यापन](../dev/sra_seq2seq_translation_analysis.md)
  - एक रिपोर्ट जो प्रदर्शित करती है कि SRA को एनकोडर-डिकोडर आर्किटेक्चर में विस्तारित करके और वास्तविक कॉर्पस (opus100) पर 30,000 चरणों के लिए प्रशिक्षण देकर, यह BLEU=1.0 के साथ "Merci beaucoup." और "Good morning." जैसे व्यावहारिक भावों का अनुवाद कर सकता है। क्रॉस-अटेंशन की शुरुआत ने डिकोडर-ओनली (BLEU=0) से 0.27 के समग्र औसत BLEU तक छलांग लगाई, और FR→EN दिशा में BLEU=0.56 की लगभग-व्यावहारिक सटीकता प्राप्त की।

---

### 📖 8. एसआरए एलएलएम डेमो (शेक्सपियर)
**File:** [`08_sra_llm_demo_shakespeare_hi.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_hi.ipynb)

यह एक ट्यूटोरियल है जो एसआरए को डिकोडर-विशिष्ट जेनरेटर मॉडल (एलएलएम) के रूप में प्रशिक्षित करने के लिए छोटे पैमाने के शेक्सपियर डेटा का उपयोग करता है। सीखने के बाद, एक हीट मैप का उपयोग यह देखने के लिए किया जाता है कि उत्पन्न पाठ का प्रत्येक टोकन किस सिनैप्स से होकर गुजरा है।

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_hi.ipynb)

---

### 🌐 9. एसआरए मल्टी-डोमेन एलएलएम डेमो (कोड, गणित, टेक्स्ट)
**File:** [`09_sra_llm_demo_multidomain_hi.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_hi.ipynb)

छोटे पैमाने के एलएलएम में ``एक साथ कई डोमेन (कोड, गणित, पाठ) सीखने'' की एसआरए की विशेषता का अनुभव करें। आप सत्यापित कर सकते हैं कि मॉडल डेटा के आधार पर स्वचालित रूप से सिनैप्स को कैसे विभाजित (विशेषीकृत) करता है।

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_hi.ipynb)

---

### 💻 10. प्रैक्टिकल प्लगइन हॉट-स्वैप (जीरो-शॉट)
**File:** [`10_hotswap_plugins_demo_hi.ipynb`](../../notebooks/10_hotswap_plugins_demo_hi.ipynb)

हम एक वर्कफ़्लो प्रदर्शित करेंगे जिसमें कई विकास टीमें स्वतंत्र रूप से "कोड" और "गणित" के लिए प्लग-इन सीखती हैं और तथ्य के बाद उन्हें उत्पादन वातावरण के बेस मॉडल में "भौतिक रूप से मर्ज (हॉट-स्वैप)" करती हैं। यह सिद्ध हो चुका है कि विलय के बाद भी, सभी डोमेन के नुकसान बिल्कुल वही हैं जो स्वतंत्र सीखने (शून्य भूलने) के दौरान होते हैं।

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_hi.ipynb)

---

### 🗑️ 11. डायनामिक सिनैप्टिक डिलीशन
**File:** [`11_synapse_deletion_demo_hi.ipynb`](../../notebooks/11_synapse_deletion_demo_hi.ipynb)

हम एसआरए के कार्य को प्रदर्शित करते हैं, "सिनैप्स विलोपन।" आप ``प्लग-इन को हटाने (पॉप_सिनैप्स)'' दोनों का अनुभव कर सकते हैं, जो अंत में बाद में जोड़े गए सिनैप्स को भौतिक रूप से हटा देता है, और ``एक विशिष्ट डोमेन को शुद्ध करता है (क्लियर_सिनैप्स)'', जो साझा नहीं किए गए सिनैप्स को सुरक्षित रूप से साफ़ और अक्षम करता है।

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_hi.ipynb)

