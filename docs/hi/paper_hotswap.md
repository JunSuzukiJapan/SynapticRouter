# Hotswappable LLM: सिनैप्टिक रूटिंग आर्किटेक्चर के माध्यम से ज़ीरो-शॉट मॉड्यूल कम्पोज़िशन और सर्जिकल ज्ञान विलोपन

**Jun Suzuki**, स्वतंत्र शोधकर्ता

## Abstract
बड़े भाषा मॉडल (LLMs) सभी ज्ञान को एक एकल मोनोलिथिक पैरामीटर स्पेस में सघन रूप से संग्रहीत करते हैं, जिससे विशिष्ट ज्ञान को जोड़ना या हटाना अत्यंत कठिन हो जाता है और परिचालन लचीलेपन पर गंभीर बाधाएं लगती हैं। इस पेपर में, मैं Synaptic Routing Architecture (SRA) की मॉड्यूलरिटी का लाभ उठाकर **Hot-Swap** — तंत्रिका नेटवर्क के लिए एक द्विदिशात्मक मॉड्यूल विनिमय ऑपरेशन — प्रस्तावित और मान्य करता हूं। Hot-Swap **Plug-In** सक्षम करता है: स्वतंत्र रूप से प्रशिक्षित विशेष सिनैप्स को पूर्व-प्रशिक्षित बेस मॉडल में **बिना किसी पुनर्प्रशिक्षण के** भौतिक रूप से सम्मिलित करना, और **Unplug**: अब आवश्यक न रहे ज्ञान को शल्य चिकित्सा की तरह हटाना। प्रयोगों से प्रदर्शित होता है कि वेक्टर डेटाबेस प्री-फ़िल्टरिंग तकनीकों से प्रेरित हार्ड-मास्किंग तंत्र **Zero Forgetting** प्राप्त करता है, जहां बेस मॉडल का आउटपुट लॉस सम्मिलन से पहले और बाद में दशमलव बिंदु तक बिल्कुल मेल खाता है। इसके अतिरिक्त, Unplug ऑपरेशन के दौरान कोसाइन समानता में शून्य वेक्टर की "ब्लैक होल समस्या" की खोज और समाधान किया, मॉड्यूलर AI का पूर्ण जीवन चक्र स्थापित करते हुए: प्रशिक्षण → Plug-In → Unplug → पुन: उपयोग।

## 1. Introduction

### 1.1 मोनोलिथिक मॉडलों की परिचालन सीमाएं
"Attention Is All You Need" की शुरुआत के बाद से, Transformer आर्किटेक्चर ने प्राकृतिक भाषा प्रसंस्करण सहित कई क्षेत्रों में प्रमुख स्थान स्थापित किया है। हालांकि, सैकड़ों अरबों पैरामीटरों वाले मोनोलिथिक LLMs निम्नलिखित गंभीर परिचालन चुनौतियों का सामना करते हैं:

1. **विनाशकारी भुलक्कड़पन (Catastrophic Forgetting)**: एक सामान्य-उद्देश्य मॉडल को विशिष्ट डोमेन पर फाइन-ट्यून करने से इसकी मूल सामान्य क्षमताएं नष्ट या क्षीण हो जाती हैं।
2. **प्रशिक्षण लागत में वृद्धि**: प्रत्येक ज्ञान जोड़ने पर पूरे मॉडल का पुनर्प्रशिक्षण आवश्यक होता है।
3. **ज्ञान विलोपन की असंभवता**: "Machine Unlearning" मोनोलिथिक मॉडलों में अत्यंत कठिन है जहां पैरामीटर गहराई से जुड़े होते हैं।

### 1.2 योगदान
SRA [Suzuki, 2026] — सूक्ष्म स्वतंत्र मॉड्यूल (सिनैप्स) और हल्के राउटर से बनी विरल आर्किटेक्चर। यह पेपर SRA की मॉड्यूलरिटी द्वारा सक्षम **Hot-Swap** पर केंद्रित है, तीन योगदान प्रस्तुत करते हुए:

1. **Hot-Swap: Plug-In (सम्मिलन)**: बेस मॉडल के खाली स्लॉट में भौतिक टेंसर कॉपी द्वारा तैनाती पूर्ण।
2. **Hot-Swap: Unplug (निष्कासन)**: भौतिक अलगाव (`pop_synapses`) और शून्य-क्लियर शुद्धिकरण (`clear_synapses`) दो हटाने की API।
3. **Zero Forgetting का प्रायोगिक प्रमाण**: हार्ड मास्क तंत्र सम्मिलन और निष्कासन दोनों के पहले और बाद बेस मॉडल लॉस की सटीक समानता सुनिश्चित करता है।

## 2. Background: SRA Architecture

### 2.1 Router
SRA का हृदय — बिना Attention तंत्र के एक **एकल रैखिक परत**। इनपुट $h$ और प्रत्येक सिनैप्स एम्बेडिंग $e_i$ के बीच कोसाइन समानता गणना करता है।

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

### 2.2 Tiny Synapses
प्रत्येक सिनैप्स छोटे Multi-Head Attention और MLP से बना स्वतंत्र मॉड्यूल।

### 2.3 साझा ट्रंक (Shared Trunk)
Hot-Swap की पूर्व शर्त: सभी विशेष सिनैप्स एक ही पूर्व-प्रशिक्षित बेस मॉडल से व्युत्पन्न हों।

## 3. Hot-Swap: Plug-In (मॉड्यूल सम्मिलन)

```python
with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

## 4. Zero Forgetting: वेक्टर DB प्री-फ़िल्टरिंग से प्रेरित हार्ड मास्क

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale
logits = logits.masked_fill(~allowed_mask, float('-inf'))
vals, idx = torch.topk(logits, k, dim=-1)
```

`masked_fill` सुनिश्चित करता है कि राउटर केवल अनुमत सिनैप्स से विशेषज्ञों का चयन करे। **बेस मॉडल का लॉस संयोजन से पहले और बाद दशमलव तक पूर्णतः मेल खाता है (गणितीय रूप से शून्य हस्तक्षेप)**।

## 5. Hot-Swap: Unplug (मॉड्यूल निष्कासन)

### 5.2 दृष्टिकोण 1: भौतिक निष्कासन (pop_synapses)
```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

### 5.3 दृष्टिकोण 2: शून्य-क्लियर शुद्धिकरण (clear_synapses)
```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

## 6. The Cosine Similarity Trap: शून्य वेक्टर ब्लैक होल समस्या

कोसाइन समानता $[-1.0, 1.0]$ की सीमा में है। शून्य-क्लियर सिनैप्स ($0.0$) वैध सिनैप्स के ऋणात्मक समानता स्कोर से गणितीय रूप से अधिक हो जाता है।

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

**समाधान**: $-\infty$ मास्क खाली सिनैप्स के चयन को गणितीय रूप से असंभव बनाता है।

```python
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

## 7. The Complete Lifecycle of Modular AI

```
प्रशिक्षण → Hot-Swap (संयोजन) → सेवा
   ↓                                ↓
स्वतंत्र                      विलोपन (शुद्धि)
समानांतर विकास                      ↓
                            स्लॉट पुन: उपयोग
                                    ↓
                            नया Hot-Swap
```

1. **प्रशिक्षण**: कई टीमें बेस मॉडल साझा करती हैं और स्वतंत्र रूप से अपने विशेष सिनैप्स विकसित करती हैं।
2. **संयोजन**: प्रशिक्षित टेंसर को प्रोडक्शन बेस मॉडल में भौतिक रूप से कॉपी किया जाता है।
3. **सेवा**: हार्ड मास्क प्री-फ़िल्टरिंग द्वारा Zero Forgetting गारंटीकृत।
4. **विलोपन**: अनावश्यक सिनैप्स भौतिक रूप से हटाए या शून्य-क्लियर किए जाते हैं।
5. **पुन: उपयोग**: खाली स्लॉट में नए विशेष सिनैप्स Hot-Swap द्वारा डाले जाते हैं।

## 8. Discussion

### 8.1 प्रतिनिधित्व विचलन
Hot-Swap की **अनिवार्य शर्त**: सभी सिनैप्स एक ही पूर्व-प्रशिक्षित बेस मॉडल (साझा ट्रंक) से व्युत्पन्न हों।

### 8.2 सुपर राउटर
स्वतंत्र मॉडलों को Gumbel-Softmax सुपर राउटर से बांधने का वैकल्पिक दृष्टिकोण $1.0$ vs $0.0$ हार्ड रूटिंग प्राप्त करता है।

### 8.3 सुरक्षा जोखिम
Pickle शोषण, बैकडोर इंजेक्शन, रूटिंग हाइजैकिंग, DoS हमले। `safetensors` प्रारूप अनिवार्य करने और क्रिप्टोग्राफ़िक हस्ताक्षर की सिफारिश।

### 8.4 वर्तमान सीमाएं
$d_\text{model}=128$, $n_\text{layers}=4$ के छोटे मॉडलों पर प्रयोगात्मक चरण। 10B वर्ग के LLMs पर मान्यकरण भविष्य की चुनौती।

## 9. Conclusion

इस पेपर में, SRA की मॉड्यूलरिटी का लाभ उठाकर LLMs को "Hotswappable" बनाने की विधियां प्रस्तावित और मान्य की गईं। Plug-In ऑपरेशन केवल भौतिक टेंसर कॉपी से तैनाती पूर्ण करता है। Unplug ऑपरेशन भौतिक अलगाव और शून्य-क्लियर शुद्धिकरण दो दृष्टिकोण स्थापित करता है। हार्ड मास्क तंत्र Zero Forgetting को गणितीय रूप से गारंटी करता है। "ब्लैक होल समस्या" का समाधान सुरक्षित स्लॉट पुन: उपयोग प्राप्त करता है।

## References

- Suzuki, J. (2026). [All You Need Is Router. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

- **Hot-Swap सिनैप्स संयोजन डेमो**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **सिनैप्स विलोपन डेमो**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[SRA का भविष्य: गतिशील Hot-Swap और विस्तारशीलता](./sra_future_hotswap_ja.md)**
- **[SRA Hot-Swap में सुरक्षा जोखिम](./sra_security_risks_hotswap_ja.md)**
- **[प्रतिनिधित्व विचलन और पदानुक्रमित रूटिंग](./sra_representation_divergence_ja.md)**
- **[SRA पदानुक्रमित राउटर के लिए हार्ड रूटिंग तुलना](./sra_hierarchical_hard_routing_ja.md)**
