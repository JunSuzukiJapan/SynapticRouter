# Hotswappable LLM: تركيب الوحدات بدون عينات والحذف الجراحي للمعرفة عبر بنية التوجيه المشبكي

**Jun Suzuki**، باحث مستقل

## Abstract
تخزن نماذج اللغة الكبيرة (LLMs) جميع المعارف بكثافة ضمن فضاء معاملات أحادي متراص، مما يجعل إضافة أو إزالة معرفة محددة أمراً بالغ الصعوبة ويفرض قيوداً شديدة على المرونة التشغيلية. في هذه الورقة، أستفيد من نمطية بنية التوجيه المشبكي (SRA) لاقتراح والتحقق من **Hot-Swap** — عملية تبادل ثنائية الاتجاه للوحدات في الشبكات العصبية. يتيح Hot-Swap **Plug-In**: إدخال المشابك المتخصصة المدربة بشكل مستقل فعلياً في نموذج أساسي مدرب مسبقاً **بدون أي إعادة تدريب**، و**Unplug**: إزالة المعرفة التي لم تعد مطلوبة جراحياً. تُظهر التجارب أن آلية القناع الصلب المستوحاة من تقنيات الترشيح المسبق لقواعد البيانات المتجهة تحقق **Zero Forgetting**، حيث يتطابق خسارة إخراج النموذج الأساسي تماماً إلى الخانة العشرية قبل وبعد الإدراج. بالإضافة إلى ذلك، أكتشف وأحل "مشكلة الثقب الأسود" للمتجهات الصفرية في تشابه جيب التمام التي تُواجَه أثناء عملية Unplug، مؤسساً دورة الحياة الكاملة للذكاء الاصطناعي النمطي: تدريب → Plug-In → Unplug → إعادة استخدام.

## 1. Introduction

### 1.1 القيود التشغيلية للنماذج المتراصة
منذ تقديم "Attention Is All You Need"، أسست بنية Transformer مكانة مهيمنة عبر العديد من المجالات بما في ذلك معالجة اللغة الطبيعية. ومع ذلك، تواجه النماذج المتراصة LLMs ذات مئات المليارات من المعاملات التحديات التشغيلية الحرجة التالية:

1. **النسيان الكارثي**: الضبط الدقيق لنموذج عام على مجال محدد يدمر أو يُضعف قدراته العامة الأصلية.
2. **تصاعد تكاليف التدريب**: كل إضافة للمعرفة تتطلب إعادة تدريب النموذج بالكامل، مما يجعل التطوير الموازي الكامل بواسطة فرق متعددة غير عملي.
3. **استحالة حذف المعرفة**: "Machine Unlearning" صعب للغاية في النماذج المتراصة حيث المعاملات متشابكة بعمق.

### 1.2 المساهمات
اقترحت سابقاً SRA [Suzuki, 2026]، بنية متفرقة مكونة من وحدات مستقلة صغيرة (مشابك) وموجه خفيف. تركز هذه الورقة على الابتكارات التشغيلية — **Hot-Swap** ثنائي الاتجاه للوحدات — مع الإبلاغ عن ثلاث مساهمات:

1. **Hot-Swap: Plug-In (الإدراج)**: النشر يتطلب فقط نسخ تنسور فعلي في الفتحات الفارغة.
2. **Hot-Swap: Unplug (الإزالة)**: تصميم واجهتي إزالة — الفصل الفعلي (`pop_synapses`) والتطهير بالتصفير (`clear_synapses`) — واكتشاف وحل "مشكلة الثقب الأسود".
3. **إثبات تجريبي لـ Zero Forgetting**: آلية القناع الصلب تضمن تطابق الخسارة تماماً قبل وبعد الإدراج والإزالة.

## 2. Background: SRA Architecture

### 2.1 Router
الموجه هو **طبقة خطية واحدة** تحسب تشابه جيب التمام بين الحالة المخفية $h$ وتضمين كل مشبك $e_i$، وتختار Top-k من المشابك.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

### 2.2 Tiny Synapses
كل مشبك هو وحدة صغيرة مستقلة من Multi-Head Attention وMLP.

### 2.3 الجذع المشترك (Shared Trunk)
الشرط الحاسم لـHot-Swap هو نهج **الجذع المشترك**. جميع المشابك المتخصصة مشتقة من نفس النموذج الأساسي المدرب مسبقاً، مع تدريب مكونات المشابك فقط بشكل مستقل.

## 3. Hot-Swap: Plug-In (إدراج الوحدة)

```python
with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

## 4. Zero Forgetting: القناع الصلب المستوحى من الترشيح المسبق لقواعد البيانات المتجهة

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale
logits = logits.masked_fill(~allowed_mask, float('-inf'))
vals, idx = torch.topk(logits, k, dim=-1)
```

يضمن `masked_fill` أن الموجه يختار الخبراء فقط من المشابك المسموح بها. **خسارة النموذج الأساسي تتطابق تماماً إلى الخانة العشرية قبل وبعد التركيب (تداخل صفري رياضياً)**.

## 5. Hot-Swap: Unplug (إزالة الوحدة)

### 5.2 النهج 1: الإزالة الفعلية (pop_synapses)
```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

### 5.3 النهج 2: التطهير بالتصفير (clear_synapses)
```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

## 6. The Cosine Similarity Trap: مشكلة الثقب الأسود للمتجه الصفري

تشابه جيب التمام يتراوح في $[-1.0, 1.0]$. المشبك المُصفَّر ($0.0$) يحصل على درجة أعلى رياضياً من المشابك الصالحة ذات التشابه السلبي.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

**الحل**: قناع $-\infty$ يجعل اختيار المشابك الفارغة مستحيلاً رياضياً.

```python
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

## 7. The Complete Lifecycle of Modular AI

```
تدريب → Hot-Swap (تركيب) → خدمة
   ↓                           ↓
تطوير مواز                 حذف (تطهير)
مستقل                         ↓
                        إعادة استخدام الفتحات
                               ↓
                        Hot-Swap جديد
```

## 8. Discussion

### 8.1 تباعد التمثيل
الشرط المطلق لـHot-Swap هو أن جميع المشابك مشتقة من نفس النموذج الأساسي (الجذع المشترك).

### 8.2 الموجه الفائق (Super Router)
نهج بديل يغلف نماذج مستقلة كاملة ويوجهها بـGumbel-Softmax، محققاً توجيهاً صلباً مثالياً $1.0$ vs $0.0$.

### 8.3 مخاطر أمنية
تتضمن المخاطر: استغلال Pickle، حقن الأبواب الخلفية، اختطاف التوجيه، وهجمات DoS.

### 8.4 القيود الحالية
البحث في مرحلة تجريبية مع نماذج صغيرة ($d_\text{model}=128$, $n_\text{layers}=4$).

## 9. Conclusion

في هذه الورقة، اقترحت وأثبتت صحة طرق لجعل LLMs "Hotswappable" من خلال استغلال نمطية SRA. عملية Plug-In تكمل النشر بنسخ التنسور الفعلي فقط. آلية القناع الصلب تضمن Zero Forgetting رياضياً. حل "مشكلة الثقب الأسود" يحقق إعادة استخدام آمنة للفتحات.

## References

- Suzuki, J. (2026). [All You Need Is Router. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

- **عرض تركيب مشابك Hot-Swap**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **عرض حذف المشابك**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[مستقبل SRA: Hot-Swap الديناميكي والقابلية للتوسع](./sra_future_hotswap_ja.md)**
- **[مخاطر أمنية في Hot-Swap SRA](./sra_security_risks_hotswap_ja.md)**
- **[تباعد التمثيل والتوجيه الهرمي](./sra_representation_divergence_ja.md)**
- **[مقارنة التوجيه الصلب للموجه الهرمي SRA](./sra_hierarchical_hard_routing_ja.md)**
