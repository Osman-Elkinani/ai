# 🏋️ شرح المشروع من الألف للياء
# AI for Personalized Health & Fitness Recommendations

---

## 📌 1. فكرة المشروع

المشروع عبارة عن **نظام ذكاء اصطناعي** يعطي المستخدم توصيات شخصية للتمارين والتغذية بناءً على حالته الصحية وأهدافه.

**مثال بسيط:** لو شخص وزنه زائد وهدفه ينزل وزن، النظام يقول له:
- "اليوم اعمل جري + نظام غذائي قليل سعرات"
- "بكرا اعمل مشي + بروتين عالي"

النظام يتعلم مع الوقت أي توصيات هي الأفضل عن طريق **التعلم المعزز (Reinforcement Learning)**.

---

## 📌 2. التقنيات المستخدمة (ربطها بالمحاضرات)

| التقنية | المحاضرة | الملف | الوظيفة |
|---------|----------|-------|---------|
| **MDP** (عمليات قرار ماركوف) | محاضرة 7 و 8 | `fitness_env.py` | نمذجة المشكلة كـ MDP |
| **A\* Search** (بحث مُوجَّه) | محاضرة 3 | `search_agent.py` | إيجاد أفضل خطة تمارين |
| **Q-Learning** | محاضرة 9 | `q_learning_agent.py` | تعلم أفضل قرار بدون نموذج |
| **Approximate Q-Learning** | محاضرة 10 | `approx_q_agent.py` | تعلم باستخدام features |
| **Epsilon-Greedy** | محاضرة 10 | كل الـ agents | موازنة الاستكشاف والاستغلال |

---

## 📌 3. هيكل المشروع (Project Structure)

```
Project/
├── run.py                     ← نقطة التشغيل الرئيسية
├── config.py                  ← كل الإعدادات والثوابت
├── requirements.txt           ← المكتبات المطلوبة
├── README.md                  ← توثيق المشروع
│
├── environment/               ← البيئة (MDP)
│   ├── fitness_env.py         ← بيئة المحاكاة
│   └── user_profile.py        ← ملفات المستخدمين
│
├── agents/                    ← عملاء الذكاء الاصطناعي
│   ├── q_learning_agent.py    ← عميل Q-Learning
│   ├── approx_q_agent.py      ← عميل Approximate Q-Learning
│   └── search_agent.py        ← عميل A* Search
│
├── data/                      ← قواعد البيانات
│   ├── workouts.py            ← أنواع التمارين
│   └── nutrition.py           ← خطط التغذية
│
└── web/                       ← لوحة التحكم
    ├── app.py                 ← سيرفر Flask
    ├── templates/
    │   └── dashboard.html     ← صفحة الداشبورد
    └── static/
        ├── css/style.css      ← التصميم
        └── js/dashboard.js    ← التفاعل
```

---

## 📌 4. شرح الـ MDP (محاضرة 7-8)

### ايش هو الـ MDP؟
**Markov Decision Process** = طريقة رياضية لنمذجة مشاكل اتخاذ القرار.

يتكون من 5 عناصر:

### 4.1 الحالات (States) — فضاء الحالات ~43,750 حالة

كل حالة عبارة عن **tuple** من 6 قيم:

```python
state = (fitness, energy, weight, muscle, fatigue, day)
```

| المتغير | القيم | المعنى |
|---------|-------|--------|
| `fitness` | 0-9 | مستوى اللياقة القلبية |
| `energy` | 0-4 | مستوى الطاقة |
| `weight` | 0-4 | حالة الوزن (نحيف → سمين) |
| `muscle` | 0-4 | مستوى العضلات |
| `fatigue` | 0-4 | مستوى الإرهاق |
| `day` | 0-6 | يوم الأسبوع |

**حجم فضاء الحالات** = 10 × 5 × 5 × 5 × 5 × 7 = **43,750 حالة**

### 4.2 الأفعال (Actions) — 28 فعل

كل فعل = **تمرين + نظام غذائي**

**7 تمارين:**
| التمرين | الشدة | السعرات |
|---------|-------|---------|
| 😴 Rest | 0 | 0 |
| 🚶 Walking | 2 | 150 |
| 🏃 Jogging | 5 | 350 |
| 🔥 HIIT | 9 | 500 |
| 🏋️ Strength | 7 | 300 |
| 🏊 Swimming | 6 | 400 |
| 🚴 Cycling | 6 | 380 |

**4 خطط غذائية:**
| الخطة | السعرات | التأثير |
|-------|---------|---------|
| 📉 Caloric Deficit | 1600 | نزول وزن |
| ⚖️ Maintenance | 2200 | ثبات |
| 📈 Caloric Surplus | 2800 | زيادة عضل |
| 🥩 High Protein | 2200 | بناء عضلات |

**المجموع** = 7 × 4 = **28 فعل ممكن**

**طريقة التشفير:**
```python
action = workout_index * 4 + nutrition_index
# مثال: HIIT (3) + High Protein (3) = 3*4+3 = 15
```

### 4.3 دالة الانتقال T(s, a, s') — عشوائية

لما المستخدم يسوي تمرين + تغذية، الحالة تتغير:

```python
# التأثيرات المتوقعة + ضوضاء عشوائية (±20%)
delta_fitness = workout_effect + random.uniform(-0.2, 0.2)
delta_energy = workout_effect + nutrition_effect + noise
# ... الخ

# ثم نطبق التغييرات على الحالة
new_fitness = clamp(fitness + delta_fitness, 0, 9)
```

**لماذا عشوائية؟** لأن في الواقع نتائج التمارين ما تكون مضمونة 100%.

### 4.4 دالة المكافأة R(s, a, s') — تعتمد على الهدف

المكافأة تختلف حسب هدف المستخدم:

**هدف نزول الوزن:**
```
+3.0  لكل نقطة نزول وزن
+1.5  لكل نقطة زيادة لياقة
+0.5  لو الطاقة ≥ متوسطة
-3.0  لو الإرهاق عالي جداً (خطر إصابة)
-2.0  لو الطاقة منخفضة جداً
-0.1  عقوبة زمنية (لتشجيع الكفاءة)
```

**هدف بناء العضلات:**
```
+3.0  لكل نقطة زيادة عضلات
+1.5  مكافأة إضافية لو (strength + high protein)
+0.8  لكل نقطة زيادة لياقة
```

### 4.5 معامل الخصم (Discount Factor) γ = 0.95

يعني نحن نهتم بالمكافآت المستقبلية لكن نفضل المكافآت القريبة.

---

## 📌 5. شرح Q-Learning (محاضرة 9)

### ايش هو Q-Learning؟
خوارزمية **Model-Free** تتعلم قيمة كل (حالة، فعل) بدون ما تعرف كيف البيئة تشتغل.

### الفكرة الأساسية:
نبني جدول Q-table يخزن **Q(s, a)** = المكافأة المتوقعة لو اخذنا الفعل `a` في الحالة `s`.

### قاعدة التحديث (Update Rule):

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
```

**شرح المعادلة:**
- `α = 0.1` → معدل التعلم (خطوة التحديث)
- `r` → المكافأة الفورية
- `γ = 0.95` → معامل الخصم
- `max_a' Q(s', a')` → أفضل قيمة مستقبلية
- الفرق `[r + γ × max - Q(s,a)]` يسمى **TD Error**

### سياسة Epsilon-Greedy (محاضرة 10):

```
بنسبة ε  → اختر فعل عشوائي (استكشاف)
بنسبة 1-ε → اختر أفضل فعل (استغلال)
```

- ε يبدأ = 1.0 (استكشاف كامل)
- ε ينقص كل حلقة: `ε = max(0.05, ε × 0.995)`
- في النهاية ε ≈ 0.05 (استغلال بنسبة 95%)

### دورة التدريب (في الكود):

```python
for episode in range(500):           # 500 حلقة تدريب
    state = env.reset()              # 1. ابدأ من البداية
    for step in range(30):           # 2. لكل يوم (30 يوم)
        action = epsilon_greedy()    #    اختر فعل
        next_state, reward = env.step(action)  # نفذ
        Q[s,a] += α*(r + γ*max(Q[s']) - Q[s,a]) # حدّث
        state = next_state
    decay_epsilon()                  # 3. قلل الاستكشاف
```

### الملف: [q_learning_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/q_learning_agent.py)

---

## 📌 6. شرح Approximate Q-Learning (محاضرة 10)

### المشكلة:
الـ Q-table حجمه = 43,750 × 28 = **1.2 مليون** خانة! كثير جداً.

### الحل:
بدل ما نخزن قيمة لكل (s, a)، نستخدم **دالة تقريبية**:

```
Q(s, a) ≈ w₁·f₁(s,a) + w₂·f₂(s,a) + ... + w₁₂·f₁₂(s,a)
```

### الـ Features (الخصائص) المستخدمة:

| # | الخاصية | الوصف |
|---|---------|-------|
| f1 | fitness / 9 | مستوى اللياقة (مُطبّع) |
| f2 | energy / 4 | مستوى الطاقة (مُطبّع) |
| f3 | (target - fitness) / 10 | الفجوة للهدف في اللياقة |
| f4 | (target - muscle) / 5 | الفجوة للهدف في العضلات |
| f5 | \|weight - target\| / 5 | فجوة الوزن |
| f6 | fatigue / 4 | مستوى الإرهاق |
| f7 | is_cardio? | هل التمرين كارديو؟ |
| f8 | is_strength? | هل تمرين قوة؟ |
| f9 | is_rest? | هل يوم راحة؟ |
| f10 | is_high_protein? | هل التغذية بروتين عالي؟ |
| f11 | is_deficit? | هل عجز سعرات؟ |
| f12 | 1 (bias) | ثابت الانحياز |

### قاعدة تحديث الأوزان:

```
difference = [r + γ × max Q(s')] - Q(s, a)
wᵢ ← wᵢ + α × difference × fᵢ(s, a)
```

### الميزة الكبيرة:
بدل 1.4 مليون قيمة ← **12 وزن فقط!** + يقدر يعمم على حالات ما شافها.

### الملف: [approx_q_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/approx_q_agent.py)

---

## 📌 7. شرح Value Iteration (محاضرة 8)

### ايش هو Value Iteration؟
خوارزمية **Model-Based** تحل الـ MDP بشكل **دقيق** باستخدام **معادلات بلمان**.

### معادلة بلمان (Bellman Equation):
```
V*(s) = max_a [ R(s, a, s') + γ · V*(s') ]
```

**المعنى:** قيمة أي حالة = أفضل مكافأة فورية + قيمة أفضل حالة مستقبلية مخصومة.

### الخوارزمية:
```
1. حط V(s) = 0 لكل الحالات (6,250 حالة)
2. كرر حتى التقارب:
   لكل حالة s:
     V(s) ← max_a [ R(s,a,s') + γ · V(s') ]
3. استخرج السياسة:
   π(s) = argmax_a [ R(s,a,s') + γ · V(s') ]
```

### الفرق عن Q-Learning:
| | Value Iteration | Q-Learning |
|---|---|---|
| يحتاج نموذج؟ | ✅ نعم (model-based) | ❌ لا (model-free) |
| دقة الحل | ✅ أمثل (exact) | ⚠️ تقريبي |
| يتعلم من التجربة؟ | ❌ لا | ✅ نعم |
| عدد الحالات المحسوبة | كل الحالات (6,250) | فقط المزارة |

### الملف: [value_iteration_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/value_iteration_agent.py)

---

## 📌 8. شرح A* Search (محاضرة 3)

### ايش هو A*؟
خوارزمية بحث **مُوجَّه** تجد أقصر/أفضل مسار من البداية للهدف.

### المعادلة:
```
f(n) = g(n) + h(n)
```
- **g(n)**: التكلفة الفعلية من البداية للنقطة n
- **h(n)**: تقدير المسافة المتبقية للهدف (الـ heuristic)
- **f(n)**: الأولوية الكلية (الأقل = الأفضل)

### الـ Heuristic (دالة التقدير):

```python
def heuristic(state):
    distance = 0
    # لهدف نزول الوزن:
    distance += max(0, weight - target_weight) * 2
    distance += max(0, target_fitness - fitness) * 1.5
    distance += fatigue * 0.5
    return distance
```

**هذه الدالة admissible** = لا تبالغ أبداً في التقدير → تضمن أن A* يجد الحل الأمثل.

### خوارزمية A* (Graph Search):

```
1. أضف نقطة البداية للـ frontier (priority queue)
2. أنشئ closed set فارغة
3. طالما الـ frontier مو فاضي:
   a. اسحب النقطة ذات أقل f(n)
   b. لو وصلت للهدف → ارجع الحل
   c. لو زرتها قبل → تخطاها (Graph Search)
   d. أضفها للـ closed set
   e. لكل فعل ممكن (28 فعل):
      - احسب الحالة الجديدة (بشكل حتمي)
      - احسب g و h الجديدة
      - أضف للـ frontier
```

### الفرق عن RL:
| | A* Search | Q-Learning |
|---|-----------|------------|
| يحتاج نموذج؟ | ✅ نعم | ❌ لا |
| يتعامل مع العشوائية؟ | ❌ لا | ✅ نعم |
| يتكيف مع التغيير؟ | ❌ لا | ✅ نعم |
| الحل الأمثل؟ | ✅ مضمون | ❌ تقريبي |

### الملف: [search_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/search_agent.py)

---

## 📌 8. شرح البيئة (fitness_env.py)

### الملف: [fitness_env.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/environment/fitness_env.py)

البيئة تتبع واجهة **OpenAI Gym** بدالتين رئيسيتين:

### `reset()` — إعادة تعيين البيئة
```python
def reset(self):
    self.state = (fitness=2, energy=3, weight=3, muscle=1, fatigue=0, day=0)
    return self.state
```

### `step(action)` — تنفيذ خطوة واحدة (يوم واحد)
```python
def step(self, action):
    # 1. فك تشفير الفعل
    workout_idx = action // 4    # أي تمرين
    nutrition_idx = action % 4   # أي تغذية

    # 2. احسب التغييرات (مع ضوضاء عشوائية)
    delta = workout_effects + nutrition_effects + noise

    # 3. طبق التغييرات على الحالة
    new_state = clamp(old_state + delta)

    # 4. احسب المكافأة
    reward = calculate_reward(old, new, goal)

    # 5. تحقق هل الهدف تحقق
    done = is_goal_reached()

    return new_state, reward, done, truncated, info
```

---

## 📌 9. شرح ملفات المستخدمين (user_profile.py)

### الملف: [user_profile.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/environment/user_profile.py)

3 ملفات شخصية جاهزة:

| الملف | الاسم | الهدف | الحالة الأولية |
|-------|-------|-------|----------------|
| `beginner_weight_loss` | Alex | نزول وزن | fitness=2, weight=3 (overweight) |
| `intermediate_muscle` | Sam | بناء عضل | fitness=5, muscle=2 |
| `advanced_fitness` | Jordan | لياقة عامة | fitness=6, muscle=3 |

---

## 📌 10. شرح الويب (Dashboard)

### الملف: [app.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/web/app.py)

سيرفر **Flask** يوفر API endpoints:

| Endpoint | الوظيفة |
|----------|---------|
| `GET /` | عرض صفحة الداشبورد |
| `POST /api/setup` | إعداد ملف المستخدم |
| `POST /api/train` | تدريب الـ agents |
| `POST /api/search` | تشغيل A* Search |
| `POST /api/simulate` | محاكاة خطوة واحدة |
| `GET /api/recommend` | الحصول على التوصيات |
| `POST /api/reset` | إعادة تعيين البيئة |

### الداشبورد يعرض:
1. **إعدادات المستخدم** — اختيار الملف الشخصي
2. **أزرار التحكم** — تدريب / بحث / محاكاة
3. **منحنى التعلم** — رسم بياني يوضح تحسن الـ agent
4. **تقدم اللياقة** — رسم بياني للمتغيرات عبر الزمن
5. **التوصيات** — ما ينصح به كل agent
6. **جدول المقارنة** — مقارنة أداء الطرق الثلاث
7. **سجل المحاكاة** — تفاصيل كل يوم

---

## 📌 11. طريقة التشغيل

```bash
# 1. تنصيب المكتبات
pip install -r requirements.txt

# 2. تشغيل الداشبورد
python run.py
# افتح المتصفح: http://127.0.0.1:5000

# أو تشغيل العرض السريع (بدون متصفح)
python run.py --cli
```

### خطوات الاستخدام في الداشبورد:
1. **اختر ملف شخصي** واضغط "Apply Profile"
2. **اضغط "Train Agents"** — ينتظر 10-30 ثانية
3. **اضغط "A* Search"** — يظهر الخطة المثلى
4. **اضغط "Simulate Step"** أو "Auto Simulate" — شاهد النتائج

---

## 📌 12. النتائج والمقارنة

من الاختبار الفعلي:

| المقياس | Q-Learning | Approx Q | A* Search |
|---------|-----------|----------|-----------|
| متوسط المكافأة | ~24 | ~79 | 14.5 |
| أفضل مكافأة | ~44 | ~87 | 14.5 |
| الحالات المستكشفة | 1,209 | N/A | 5 |
| يتعامل مع العشوائية | ✅ | ✅ | ❌ |
| قابلية التعميم | ❌ ضعيفة | ✅ عالية | ❌ ضعيفة |
| التكيف | ✅ عالي | ✅ عالي | ❌ منخفض |

### الملاحظات:
- **Approximate Q-Learning** حقق أفضل أداء لأنه يعمم على حالات جديدة
- **A* Search** وجد الخطة المثلى بسرعة (5 خطوات فقط) لكن لا يتكيف مع التغيير
- **Q-Learning** جيد لكن يحتاج وقت أطول لاستكشاف الحالات

---

## 📌 13. ملخص الملفات المهمة

### [config.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/config.py)
كل الثوابت والإعدادات: أحجام فضاءات الحالة، معاملات التعلم، إعدادات السيرفر.

### [data/workouts.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/data/workouts.py)
قاعدة بيانات التمارين الـ 8 مع تأثيراتها على كل متغير.

### [data/nutrition.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/data/nutrition.py)
قاعدة بيانات خطط التغذية الـ 4 مع السعرات والماكروز.

### [environment/fitness_env.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/environment/fitness_env.py)
**أهم ملف** — البيئة كاملة: الحالات، الانتقالات، المكافآت، التحقق من الهدف.

### [agents/q_learning_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/q_learning_agent.py)
Q-Learning مع Q-table و epsilon-greedy و training loop.

### [agents/approx_q_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/approx_q_agent.py)
Approximate Q-Learning مع feature extraction و weight updates.

### [agents/search_agent.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/agents/search_agent.py)
A* Search مع priority queue و heuristic و Graph Search.

### [run.py](file:///d:/UNVERSTY/Second-Term/Ai/Project/run.py)
نقطة التشغيل — تدعم وضعين: web و cli.
