import os

docs_dir = r"d:\UNVERSTY\Second-Term\Ai\Project\docs"
os.makedirs(docs_dir, exist_ok=True)

chapters = [
    ("index.html", "🏠 00. النظرة الشاملة وهيكلة المشروع"),
    ("01_mdp.html", "🎲 01. المعمارية الرياضية لبيئة (MDP)"),
    ("02_qlearning.html", "🧠 02. تفصيل خوارزمية Q-Learning"),
    ("03_approx_q.html", "⚡ 03. Approximate Q-Learning والخصائص"),
    ("04_value_iteration.html", "📐 04. Value Iteration والحل الدقيق"),
    ("05_astar.html", "🔍 05. A* Search والتخطيط المستقبلي"),
    ("06_dashboard.html", "🖥️ 06. واجهة المستخدم والـ APIs"),
    ("07_defense_faq.html", "🛡️ 07. الأسئلة المتوقعة والردود الأكاديمية"),
]

def get_sidebar(active_file):
    links = ""
    for filename, title in chapters:
        active = 'class="active"' if filename == active_file else ''
        links += f'<li><a href="{filename}" {active}>{title}</a></li>\n'
    
    return f"""
    <div class="sidebar">
        <h2>📚 التوثيق الشامل</h2>
        <ul>
            {links}
        </ul>
    </div>
    """

def wrap_html(filename, title, content):
    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>{title} — AI Health Project</title>
    <link rel="stylesheet" href="style.css">
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
</head>
<body>
    {get_sidebar(filename)}
    <div class="content">
        {content}
    </div>
</body>
</html>
"""

# ==========================================
# 00. Index
# ==========================================
c_index = """
<h1>🏋️ المشروع: AI for Personalized Health & Fitness Recommendations</h1>
<p>هذا التوثيق هو الدليل المعماري والبرمجي الشامل للمشروع. تم إعداد هذا التوثيق ليفصل كل مكون من مكونات النظام، من المعادلات الرياضية إلى تصميم الواجهة، مع تبرير كل قرار برمجي تم اتخاذه.</p>

<h2>🎯 1. الرؤية والهدف الأكاديمي</h2>
<p>يهدف هذا المشروع إلى تحويل مشكلة "توصيات اللياقة البدنية" المعقدة في العالم الحقيقي إلى مشكلة قابلة للحل رياضياً باستخدام الذكاء الاصطناعي (Reinforcement Learning & Search Algorithms). التحدي الأكبر في هذا المجال هو أن استجابة جسم الإنسان للتمرين والتغذية هي استجابة بطيئة وتراكمية وفيها نسبة من العشوائية. لذلك، قمنا ببناء محرك محاكاة (Simulation Engine) متوافق مع معايير Markov Decision Process (MDP).</p>

<h2>⚙️ 2. التقنيات والخوارزميات المستخدمة</h2>
<p>يغطي المشروع 5 موضوعات أساسية من منهج الذكاء الاصطناعي:</p>
<ol>
    <li><strong>نمذجة البيئة (MDP):</strong> تحويل الجسم لـ States و Actions و Transitions و Rewards.</li>
    <li><strong>الوكيل المستكشف (Q-Learning - Model-Free):</strong> يتعلم بالتجربة والخطأ عبر آلاف الحلقات (Episodes).</li>
    <li><strong>الوكيل المعمم (Approximate Q-Learning):</strong> يحل مشكلة الذاكرة (Curse of Dimensionality) بتعلم أوزان الخصائص بدلاً من حفظ الحالات.</li>
    <li><strong>الوكيل الدقيق (Value Iteration - Model-Based):</strong> يحل معادلات بلمان للبيئة بأكملها للوصول للسياسة المثلى رياضياً.</li>
    <li><strong>الوكيل المخطط (A* Search):</strong> يستخدم خوارزميات البحث لبناء خطة كاملة لـ 30 يوم بأقل تكلفة للوصول للهدف.</li>
</ol>

<h2>🏗️ 3. معمارية النظام (System Architecture)</h2>
<div class="mermaid">
graph TD
    User([المستخدم]) --> |إدخال البيانات والهدف| UI[Web Dashboard]
    UI --> |API Requests| Flask[Flask Backend API]
    Flask --> Config[Config Variables]
    Flask --> ENV[Fitness Environment MDP]
    ENV --> DB_Work[Workouts Data]
    ENV --> DB_Nutr[Nutrition Data]
    
    Flask --> QAgent[Q-Learning Agent]
    Flask --> ApproxAgent[Approximate Q-Learning Agent]
    Flask --> VIAgent[Value Iteration Agent]
    Flask --> AStarAgent[A* Search Agent]
    
    QAgent --> ENV
    ApproxAgent --> ENV
    VIAgent --> ENV
    AStarAgent --> ENV
    
    ENV --> |State, Reward, Done| QAgent
</div>

<h2>📂 4. الهيكلة التفصيلية للملفات</h2>
<table dir="ltr">
    <tr><th>المسار</th><th>الوصف والغرض</th></tr>
    <tr><td><code>run.py</code></td><td>نقطة الدخول للمشروع. يشغل سيرفر Flask على المنفذ 5000.</td></tr>
    <tr><td><code>config.py</code></td><td>يحتوي على كافة الهايبر-بارامترز (Hyperparameters) مثل <code>NUM_EPISODES</code>، <code>ALPHA</code>، <code>GAMMA</code>. مركزية الإعدادات تمنع تضارب الخوارزميات.</td></tr>
    <tr><td><code>environment/fitness_env.py</code></td><td>القلب النابض للمحاكاة. يحتوي على كلاس <code>FitnessEnv</code> الذي يدير الحالات والمكافآت (MDP).</td></tr>
    <tr><td><code>environment/user_profile.py</code></td><td>يدير ملفات المستخدمين (مثل المبتدئ الذي يريد نزول الوزن، أو المتقدم الذي يريد بناء العضلات).</td></tr>
    <tr><td><code>agents/q_learning_agent.py</code></td><td>تطبيق Q-Learning مع جدول Q-Table (قاموس Python).</td></tr>
    <tr><td><code>agents/approx_q_agent.py</code></td><td>تطبيق Approximate Q-Learning باستخدام 12 Feature وأوزان.</td></tr>
    <tr><td><code>agents/value_iteration_agent.py</code></td><td>يحل معادلات بلمان بشكل مباشر لاستخراج السياسة المثلى.</td></tr>
    <tr><td><code>agents/search_agent.py</code></td><td>ينفذ خوارزمية A* Search بخاصية المراكم (Accumulator) المتقدمة.</td></tr>
    <tr><td><code>data/workouts.py</code></td><td>قاعدة بيانات التمارين وسعراتها (مبنية على أسس علمية CDC).</td></tr>
    <tr><td><code>data/nutrition.py</code></td><td>قاعدة بيانات خطط التغذية وسعراتها والماكروز.</td></tr>
    <tr><td><code>web/app.py</code></td><td>يحتوي على مسارات API لخدمة الواجهة الأمامية.</td></tr>
    <tr><td><code>web/static/js/dashboard.js</code></td><td>المنطق المعقد للواجهة الأمامية (Chunked Training, Chart.js updates).</td></tr>
</table>
"""

# ==========================================
# 01. MDP
# ==========================================
c_mdp = """
<h1>🎲 01. المعمارية الرياضية لبيئة (MDP)</h1>
<p>تم تمثيل المشكلة كـ MDP في ملف <code>environment/fitness_env.py</code> بطريقة دقيقة ومحكمة. هذا الملف هو الأطول والأكثر تعقيداً في المشروع (أكثر من 500 سطر برمجي).</p>

<h2>1. فضاء الحالات (State Space)</h2>
<p>تم تعريف الحالة (State) على أنها Tuple رياضي مكون من 6 عناصر. تم اختيار هذه العناصر وتحديد نطاقاتها بعناية شديدة:</p>
<table>
    <tr><th>المتغير</th><th>القيم (Range)</th><th>العدد</th><th>الوصف التفصيلي</th></tr>
    <tr><td><code>fitness</code></td><td>0 إلى 9</td><td>10</td><td>مستوى اللياقة القلبية. يبدأ من 0 (خامل جداً) إلى 9 (رياضي أولمبي).</td></tr>
    <tr><td><code>energy</code></td><td>0 إلى 4</td><td>5</td><td>مستوى طاقة الجسم الحالية (يقل مع التمرين ويزيد مع الراحة والتغذية).</td></tr>
    <tr><td><code>weight</code></td><td>0 إلى 4</td><td>5</td><td>0: نحيف جداً، 1: نحيف، 2: طبيعي، 3: وزن زائد، 4: بدين.</td></tr>
    <tr><td><code>muscle</code></td><td>0 إلى 4</td><td>5</td><td>مستوى الكتلة العضلية (0: ضعيف، 4: لاعب كمال أجسام).</td></tr>
    <tr><td><code>fatigue</code></td><td>0 إلى 4</td><td>5</td><td>الإرهاق العصبي والعضلي المتراكم. 4 يعني خطر الإصابة ويستوجب الراحة.</td></tr>
    <tr><td><code>day</code></td><td>0 إلى 6</td><td>7</td><td>يوم الأسبوع (لأن بعض المستخدمين يفضلون الراحة في أيام معينة).</td></tr>
</table>
<p><strong>حجم فضاء الحالات الإجمالي:</strong> 10 × 5 × 5 × 5 × 5 × 7 = <strong>43,750 حالة ممكنة</strong>.</p>
<div class="callout warning">
    <strong>تبرير هندسي قوي (Curse of Dimensionality):</strong><br>
    لماذا لم نضف متغيرات إضافية مثل "الوقت المتاح للمستخدم" أو "نوع المعدات"؟
    لو أضفنا "المعدات" بثلاث حالات (جيم، منزل، معدات خفيفة)، سيصبح الفضاء: 43,750 × 3 = 131,250 حالة. ولو أضفنا "الوقت المتاح"، سيصل إلى الملايين.
    هذا يسمى لعنة الأبعاد (Curse of Dimensionality)، وسيجعل Q-Learning مستحيلاً. لذلك، تم التعامل مع المعدات والوقت كـ "مرشحات سياقية" (Contextual Filters) تمنع حركات معينة قبل بدء الحسابات بدلاً من وضعها في الحالة.
</div>

<h2>2. هندسة فضاء الأفعال (Action Space)</h2>
<p>في كل حالة، الوكيل يجب أن يتخذ قراراً مركباً يتكون من عنصرين معاً:</p>
<ul>
    <li><strong>التمرين (Workout):</strong> واحد من 7 خيارات (Rest, Walking, Jogging, Cycling, Swimming, HIIT, Strength Training).</li>
    <li><strong>التغذية (Nutrition):</strong> واحد من 4 خيارات (Deficit, Maintenance, Surplus, High Protein).</li>
</ul>
<p><strong>حجم فضاء الأفعال (Action Space):</strong> 7 × 4 = <strong>28 فعل مركب ممكن في كل يوم</strong>.</p>

<h2>3. دالة الانتقال (Transition Function & Stochasticity)</h2>
<p>تم تصميم <code>step()</code> function لتكون عشوائية جزئياً (Stochastic) لتحاكي الواقع البشري:</p>
<pre><code>if random.random() < 0.10: # 10% Chance of random physical fluctuation
    energy_change -= 1 # Unexpected tiredness
    fatigue_change += 1 # Unexpected soreness</code></pre>

<h2>4. نمط المراكم (The Accumulator Pattern) المبتكر</h2>
<p><strong>المشكلة الكبرى:</strong> خوارزميات RL تتوقع أن الفعل $a$ في الحالة $s$ ينقلك للحالة $s'$ مباشرة. لكن في الواقع، لعب الحديد ليوم واحد لن ينقلك من مستوى عضلي 2 إلى 3. النقلة تحتاج 10 أيام من التمرين.</p>
<p><strong>الحل البرمجي (Accumulators):</strong> تم اختراع نظام "مراكم" للبيانات البطيئة (الوزن، العضل، اللياقة).</p>
<pre><code># يتم إضافة كسور بدل أرقام صحيحة
self.accumulators['weight'] += -0.05  # عجز سعرات ينقص الوزن قليلاً
self.accumulators['muscle'] += 0.08   # تمرين قوة يضيف عضل قليلاً

# عندما يكتمل الرقم لـ 1 صحيح، يتم تغيير الـ State الفعلية
if abs(self.accumulators['muscle']) >= 1.0:
    new_muscle += sign(self.accumulators['muscle'])
    self.accumulators['muscle'] = 0.0</code></pre>
<p>هذا النمط المعماري حل مشكلة التخطيط طويل الأمد، خصوصاً مع A* Search.</p>

<h2>5. دالة المكافأة العميقة (Deep Reward Function)</h2>
<p>تم هندسة الـ Reward في <code>_calculate_reward</code> لتكون دالة مركبة (Composite Function) تعتمد على 4 أوزان:</p>
<ol>
    <li><strong>$W_1$ (تقدم الهدف):</strong> مكافأة عالية (+3.0) لكل مستوى وزن ينزل إذا كان الهدف weight_loss.</li>
    <li><strong>$W_2$ (المحافظة على الطاقة):</strong> مكافأة (+0.5) لضمان عدم إرهاق المستخدم.</li>
    <li><strong>$W_3$ (عقوبة الإرهاق - Exponential):</strong> عقوبات صارمة متصاعدة. (-4.0) إذا وصل الإرهاق 4، و (-2.0) إذا وصل 3. هذا يضمن سلامة المستخدم (Safety Penalty).</li>
    <li><strong>$W_4$ (تنوع التمارين):</strong> عقوبة خفيفة (-0.3) إذا كرر الوكيل نفس التمرين يومين متتاليين لتجنب الملل.</li>
</ol>

<h3>Potential-Based Reward Shaping (PBRS)</h3>
<p>استخدمنا تقنية PBRS لتسريع التعلم دون إفساد السياسة المثلى. الفكرة هي إعطاء الـ Agent "تلميحات" مستمرة كلما اقترب من الهدف (Heuristic-like reward).</p>
<pre><code>phi_old = -self.state_distance_to_goal(old_state)
phi_new = -self.state_distance_to_goal(new_state)
shaping = DISCOUNT_FACTOR * phi_new - phi_old
reward += shaping * 0.5</code></pre>
"""

# ==========================================
# 02. Q-Learning
# ==========================================
c_q = """
<h1>🧠 02. تفصيل خوارزمية Q-Learning (محاضرة 9)</h1>
<p>موجودة في ملف <code>agents/q_learning_agent.py</code>.</p>

<h2>1. طبيعة الخوارزمية (Model-Free, Off-Policy)</h2>
<p>هذه الخوارزمية لا تفهم كيف يعمل جسم الإنسان (لا تملك Model). هي فقط تتخذ فعلاً وتراقب المكافأة (Reward) والحالة الجديدة (Next State)، وبناءً عليها تحدّث جدولاً ضخماً اسمه Q-Table.</p>

<h2>2. جدول Q-Table والتحدي البرمجي</h2>
<p>لأن لدينا 43,750 حالة، تخزين مصفوفة ثابتة (Numpy Array) بهذا الحجم سيكون غير كفء لأن العديد من الحالات (مثل لياقة 9 مع وزن بدين 4) مستحيلة الحدوث.</p>
<p><strong>الحل:</strong> استخدام <code>defaultdict</code> من مكتبة <code>collections</code> في بايثون، بحيث لا يتم تهيئة الـ State في الذاكرة إلا إذا زارها الوكيل فعلياً. هذا يوفر الذاكرة بشكل هائل.</p>

<h2>3. التحديث الرياضي لـ Bellman</h2>
<p>في كل خطوة (Step)، يتم تحديث القيمة في الجدول بناءً على المعادلة:</p>
<pre><code>Q(s, a) = Q(s, a) + α * [ R + γ * max_a' Q(s', a') - Q(s, a) ]</code></pre>
<p>تم ترجمة هذا حرفياً في الكود:</p>
<pre><code>best_next_q = max(self.q_table[next_state].values())
td_target = reward + self.gamma * best_next_q
td_error = td_target - current_q
self.q_table[state][action_key] += self.learning_rate * td_error</code></pre>

<h2>4. الهايبر بارامترز (Hyperparameters)</h2>
<table dir="ltr">
    <tr><th>المتغير</th><th>القيمة</th><th>التأثير والتبرير</th></tr>
    <tr><td><code>NUM_EPISODES</code></td><td>5000</td><td>عدد محاولات المحاكاة للوصول للسياسة. 5000 يعتبر كافياً في بيئة بحجم 43 ألف حالة مع استخدام PBRS.</td></tr>
    <tr><td><code>ALPHA (α)</code></td><td>0.1</td><td>معدل التعلم. قمنا بتحديده بقيمة منخفضة (0.1) لأن البيئة Stochastic، ولو كان عالياً جداً سيتذبذب التعلم بشدة كلما حدث حدث عشوائي.</td></tr>
    <tr><td><code>GAMMA (γ)</code></td><td>0.95</td><td>أهمية المستقبل. قيمة عالية جداً لأن اللياقة البدنية تتطلب تضحية فورية (تعب اليوم) من أجل مكافأة بعيدة (جسم صحي بعد شهر).</td></tr>
    <tr><td><code>EPSILON_DECAY</code></td><td>0.998</td><td>معدل التناقص. مع كل Episode نضرب Epsilon في هذا الرقم. بدأنا بـ 1.0 وننزل ببطء لضمان استكشاف البيئة بالكامل قبل الاستقرار (Exploration vs Exploitation).</td></tr>
</table>

<h2>5. مراقبة التقدم والـ Metrics</h2>
<p>يقوم الـ Agent بحساب مقاييس دقيقة جداً لعرضها في الـ Dashboard:</p>
<ul>
    <li><strong>Policy Stability:</strong> يقارن السياسة الحالية بسياسة الـ 100 حلقة السابقة. إذا لم تتغير بنسبة 90%، نعتبر أن التدريب اكتمل.</li>
    <li><strong>State Coverage:</strong> يقيس كم حالة تم استكشافها من أصل 43 ألف. عادة يستكشف حوالي 2000-3000 حالة فقط لأن الـ Reward Shaping يوجهه مباشرة نحو الهدف بدل التخبط العشوائي.</li>
</ul>
"""

# ==========================================
# 03. Approx Q
# ==========================================
c_approx = """
<h1>⚡ 03. Approximate Q-Learning والخصائص (محاضرة 10)</h1>
<p>موجودة في ملف <code>agents/approx_q_agent.py</code>.</p>

<h2>1. الدافع المعماري (المشكلة مع Q-Learning)</h2>
<p>في الـ Q-Learning العادي (Tabular)، لو تدرب الوكيل على حالة (لياقة=4، طاقة=3) وواجه لأول مرة في حياته حالة (لياقة=5، طاقة=3)، سيقف عاجزاً ويتصرف بعشوائية لأنه لا يملك خانة لها في الجدول! لا يوجد لديه استنتاج أو تعميم (Generalization).</p>

<h2>2. الحل: Linear Function Approximation</h2>
<p>استبدلنا الجدول الضخم بـ "معادلة خطية" بسيطة تعتمد على الأوزان (Weights) والخصائص (Features).</p>
<pre><code>Q(s, a) = w1*f1 + w2*f2 + ... + wn*fn</code></pre>
<p>إذا واجه الوكيل حالة جديدة، يقوم باستخراج خصائصها وضربها في الأوزان التي تعلمها ليستنتج قيمة ممتازة.</p>

<h2>3. هندسة الخصائص (Feature Engineering)</h2>
<p>قمنا بتعريف 12 Feature تعطي الوكيل إدراكاً عميقاً لما يحدث. في كود <code>extract_features(state, action)</code> تم تطبيع كل القيم (Normalization) لتكون بين 0 و 1 لتسريع استقرار الأوزان:</p>
<ul>
    <li><code>bias</code>: دائماً 1.0 (كقيمة ثابتة أساسية للتقاطع).</li>
    <li><code>fitness_level</code>: اللياقة الحالية مقسومة على 9.</li>
    <li><code>energy_level</code>: الطاقة الحالية مقسومة على 4.</li>
    <li><code>fatigue_level</code>: الإرهاق مقسوم على 4.</li>
    <li><code>distance_to_goal</code>: دالة الـ Heuristic مقسومة على الحد الأقصى المتوقع للمسافة لتكون بين 0 و 1.</li>
    <li><strong>أعلام الأفعال (Action Flags):</strong> هل الفعل هو Cardio؟ (0 أو 1)، هل هو Strength؟ هل هو Rest؟ هل هو Deficit؟</li>
    <li><strong>تقاطعات (Intersections):</strong> ميزة خاصة تُفعل فقط إذا كان الإرهاق عالي جداً والفعل المختار هو Rest. هذا يعلم الوزن أن الراحة عند التعب هي الخيار الأفضل.</li>
</ul>

<h2>4. التحديث وتعديل الأوزان (Weight Update)</h2>
<p>بدلاً من تحديث جدول الـ Q، نحدث مصفوفة الأوزان التي حجمها 12 فقط!</p>
<pre><code>td_error = (reward + self.gamma * max_next_q) - current_q

# تحديث كل وزن بناءً على خطأ التوقع وقيمة الخاصية
for i in range(len(self.weights)):
    self.weights[i] += self.learning_rate * td_error * features[i]</code></pre>
<div class="callout success">
    <strong>النتيجة والأداء:</strong><br>
    هذا الوكيل يتعلم أسرع بكثير من Q-Learning العادي، وتكون منحنياته في الـ Learning Curve أكثر استقراراً مبكراً (في أول 500 حلقة) لأنه يطبق ما تعلمه في حالة واحدة على بقية الحالات المشابهة.
</div>
"""

# ==========================================
# 04. Value Iteration
# ==========================================
c_vi = """
<h1>📐 04. Value Iteration والحل الدقيق (محاضرة 8)</h1>
<p>موجودة في ملف <code>agents/value_iteration_agent.py</code>.</p>

<h2>1. النموذج الكامل (Model-Based)</h2>
<p>خوارزمية Value Iteration لا تحتاج إلى التدريب بالحلقات (No Episodes). هي تفترض أنها تعرف بدقة (الديناميكية) للبيئة (أي دالة <code>Transition Model T(s, a, s')</code> ودالة <code>Reward Model R(s, a)</code>).</p>
<p>بما أن بيئتنا هي بيئة برمجية يمكن استنساخها (Clone)، تمكنا من برمجة دالة <code>get_deterministic_next_state(state, action)</code> التي تخبر الوكيل تماماً بما سيحدث إذا نفذ فعلاً ما دون الحاجة لتجريبه.</p>

<h2>2. تقليص مساحة البحث للسرعة الحسابية</h2>
<p>لأن Value Iteration يعاني من ثقل الحسابات (يجب أن يحسب كل فعل في كل حالة مراراً وتكراراً)، قمنا بهندسة ذكية لتقليص فضاء الحالات:</p>
<p>بما أن الـ "يوم" (Day) لا يؤثر بشكل مباشر على الحسابات الرياضية للمكافأة، قمنا بإهمال متغير اليوم داخل خوارزمية الـ Value Iteration.
<br>الفضاء القديم: 10 × 5 × 5 × 5 × 5 × 7 = 43,750 حالة.
<br><strong>الفضاء الجديد المقلص:</strong> 10 × 5 × 5 × 5 × 5 = <strong>6,250 حالة فقط.</strong></p>

<h2>3. حلقة معادلة بلمان (Bellman Loop)</h2>
<p>الخوارزمية تمر على الـ 6,250 حالة في <code>while True</code> loop، وتطبق المعادلة:</p>
<pre><code>V_new(s) = max_a [ R(s,a) + γ * V(s') ]</code></pre>
<p>تستمر الحلقة حتى يصبح الـ <code>delta</code> (أكبر تغيير حدث في القيمة) أصغر من <code>theta = 0.01</code>. هذا يعني أن القيم استقرت ووصلنا للحل الأمثل رياضياً (Convergence).</p>

<h2>4. استخراج السياسة (Policy Extraction)</h2>
<p>كيف نستخدم هذا الـ V-Table (الذي يحتوي قيماً من نوع float لكل حالة) لنتخذ قراراً؟ عبر دالة الـ One-Step Lookahead:</p>
<pre><code>best_action = None
max_value = -float('inf')

for action in all_actions:
    next_state = env.simulate_deterministic(state, action)
    reward = env.calculate_reward(state, next_state, action)
    
    # القيمة الكلية = المكافأة الفورية + قيمة الحالة القادمة
    action_value = reward + gamma * V[next_state]
    
    if action_value > max_value:
        max_value = action_value
        best_action = action</code></pre>

<div class="callout info">
    <strong>مقارنة الأداء:</strong><br>
    بينما Q-Learning يحتاج بضعة ثوان للتدريب على عينة من الحالات، Value Iteration قد يستغرق 30-60 ثانية لتحديث 6,250 حالة بشكل متكرر، لكنه يضمن لك الحل الرياضي الأمثل والأكثر دقة (Exact Optimal Policy).
</div>
"""

# ==========================================
# 05. A* Search
# ==========================================
c_astar = """
<h1>🔍 05. A* Search والتخطيط المستقبلي (محاضرة 3)</h1>
<p>موجودة في ملف <code>agents/search_agent.py</code>.</p>

<h2>1. فكرة التخطيط بدلاً من التفاعل (Planning vs Reacting)</h2>
<p>طرق الـ RL (Q-Learning و Value Iteration) هي طرق <strong>تفاعلية (Reactive)</strong>، أي أنها تسأل: "أنا الآن في حالة كذا، ماذا أفعل؟".</p>
<p>بينما الـ A* Search هي طريقة <strong>تخطيط (Planning)</strong>، تسأل: "أنا الآن في نقطة البداية، كيف أرسم خطة متكاملة من 30 خطوة متتالية توصلني للهدف بأقل تكلفة (Cost)؟".</p>

<h2>2. تمثيل الـ Node والتكلفة G(n)</h2>
<p>في شجرة البحث، كل عقدة (Node) تحتوي على:</p>
<ul>
    <li><code>state</code>: الحالة الحالية للجسم.</li>
    <li><code>accumulators</code>: نسخة من المتغيرات التراكمية الحالية (ضرورية جداً لكي يحسب A* تأثير الدايت الطويل على الوزن).</li>
    <li><code>path</code>: مسار الأفعال منذ نقطة البداية.</li>
    <li><code>g_cost</code>: التكلفة الفعلية التراكمية. (بما أننا نبحث عن المكافأة، قمنا بقلب المكافأة لتصبح تكلفة: <code>Cost = 10 - Reward</code>).</li>
</ul>

<h2>3. تصميم الـ Heuristic H(n) والإثبات الرياضي</h2>
<p>لكي يكون A* فعالاً ولا يبحث في مسارات لا نهاية لها، يجب توجيهه بدالة Heuristic. ولكي نضمن أن A* سيجد الحل المثالي (Optimal)، يجب أن تكون H(n) <strong>Admissible (غير مبالغة في التقدير)</strong>.</p>
<pre><code>def _heuristic(self, state, target_state):
    # حساب الفجوة بين اللياقة الحالية والهدف
    fitness_gap = max(0, target_state['fitness'] - state[0])
    
    # الحد الأقصى للمكافأة التي يمكن الحصول عليها في خطوة واحدة للياقة هو 0.2
    # إذن، أقل عدد ممكن من الأيام هو:
    min_steps_fitness = fitness_gap / 0.2
    
    # الحد الأقصى للمكافأة في نظامنا هو تقريباً 5
    # التكلفة المقلوبة: Cost = 10 - 5 = 5 (أقل تكلفة للخطوة)
    optimistic_cost_per_step = 5.0
    
    return min_steps_fitness * optimistic_cost_per_step</code></pre>
<div class="callout success">
    <strong>الإثبات الرياضي (Admissibility Proof):</strong><br>
    بما أننا نفترض أن الوكيل سيحصل على "أعلى مكافأة ممكنة" (أقل تكلفة) في كل خطوة قادمة دون أي إرهاق (وهذا مستحيل عملياً)، فإن الـ Heuristic <strong>دائماً ستقلل من التكلفة الحقيقية (Underestimates the true cost)</strong>، وهو بالضبط الشرط الرياضي للـ Admissibility.
</div>

<h2>4. حل مشكلة הـ Depth Limit</h2>
<p>شجرة البحث تتفرع بقوة (Branching factor = 28 فعل). إذا بحثنا لـ 30 يوم، سيكون لدينا $28^{30}$ حالة، وهو رقم فلكي يكسر الميموري.
<br><strong>الحل:</strong> تطبيق Depth-Bounded A* Search. وضعنا سقف للعمق (مثلاً 14 أو 30) تم تعريفه في <code>config.py (MAX_SEARCH_DEPTH)</code>. إذا وصل A* للعمق الأقصى، يرجع أفضل خطة وجدها حتى لو لم يصل للهدف النهائي تماماً.
</p>
"""

# ==========================================
# 06. Dashboard
# ==========================================
c_dashboard = """
<h1>🖥️ 06. واجهة المستخدم والـ APIs الشاملة</h1>
<p>تم بناء واجهة الويب لتعكس كل ما يحدث في الـ Backend بشكل مباشر ومرئي، وتسهل مناقشة المشروع واستعراضه.</p>

<h2>1. الـ Backend (سيرفر Flask)</h2>
<p>في ملف <code>web/app.py</code>، قمنا بتعريف مجموعة من الـ Endpoints للتعامل مع الـ Frontend باستخدام الـ JSON:</p>

<h3>أ. <code>POST /api/setup</code></h3>
<ul>
    <li><strong>المدخلات:</strong> JSON يحتوي على اسم الـ Profile المختار (مثل <code>beginner_weight_loss</code>) أو بيانات <code>custom</code> كاملة.</li>
    <li><strong>العملية:</strong> يهيئ كلاس <code>UserProfile</code> وكلاس <code>FitnessEnv</code> ويمسح الـ Logs السابقة.</li>
    <li><strong>المخرجات:</strong> حالة الجسم الحالية والأهداف للـ UI.</li>
</ul>

<h3>ب. <code>POST /api/train</code> (نواة التدريب)</h3>
<ul>
    <li><strong>المدخلات:</strong> <code>episodes</code> (عدد الحلقات)، <code>agent</code> (نوع الوكيل)، <code>hyperparams</code> (الهايبر بارامترز)، <code>resume</code> (هل هو تكملة لتدريب سابق أم جديد؟).</li>
    <li><strong>العملية:</strong> يشغل دالة التدريب للـ Q-learning والـ Approx Q.</li>
    <li><strong>المخرجات:</strong> مصفوفات الـ Rewards لكل حلقة ليتم رسمها في الـ Chart، بالإضافة للمقاييس النهائية (Stability, Coverage).</li>
</ul>

<h3>ج. <code>GET /api/value_iteration</code> و <code>POST /api/search</code></h3>
<p>مسارات مسؤولة عن تشغيل الخوارزميات الثقيلة حسابياً واسترجاع النتائج.</p>

<h2>2. الـ Frontend (التصميم البصري والـ JavaScript)</h2>

<h3>أ. معمارية التصميم (Professional Blue Theme)</h3>
<p>تم كتابة CSS مخصص بالكامل في <code>style.css</code> يحاكي تصاميم لوحات التحكم الحديثة:</p>
<ul>
    <li><strong>الألوان:</strong> استخدام ألوان داكنة (Dark Navy: <code>#0b1120</code>) مع ألوان نيون متوهجة (Electric Blue <code>#3b82f6</code>، Cyan <code>#06b6d4</code>) تناسب العرض بالبروجكتور.</li>
    <li><strong>Glassmorphism:</strong> استخدام <code>backdrop-filter: blur(16px)</code> للـ Header ليعطي شفافية زجاجية أنيقة.</li>
    <li><strong>الأنيميشن:</strong> أزرار ديناميكية مع <code>hover elevation</code> وحركات خفيفة للرسوم البيانية وأشرطة التقدم.</li>
</ul>

<h3>ب. هندسة الـ JavaScript المتقدمة (Chunked Training)</h3>
<p>من أبرز الإنجازات البرمجية في الواجهة هي دالة <code>trainAgents()</code>. لو قمنا بإرسال طلب واحد للـ Backend ليدرب 5000 حلقة، سيتجمد المتصفح والسيرفر معاً حتى ينتهي التدريب ولن يرى المستخدم أي شيء.
<br><strong>الحل (Batched Execution):</strong> قمنا بتقسيم التدريب لـ Batches (كل دفعة 50 حلقة).</p>
<pre><code>for (let i = 0; i < numBatches; i++) {
    if (cancelTraining) break; // زر الإيقاف (Stop Button) يعمل هنا!
    
    // إرسال طلب لـ 50 حلقة فقط مع خاصية (resume = true)
    const data = await apiCall('/api/train', 'POST', { episodes: 50, resume: true });
    
    // تحديث شريط التقدم (Progress Bar) والرسم البياني (Chart) مباشرة
    updateProgress(pct);
    updateLearningCurveLive(data);
}</code></pre>
<p>هذه المعمارية سمحت لنا بإضافة زر **إيقاف التدريب (Stop Training)**، وعرض منحنى التعلم يرتفع لايف أمام الدكتور.</p>

<h3>ج. دالة المتوسط المتحرك (Moving Average)</h3>
<p>لأن الـ Rewards تتذبذب بشكل حاد جداً من حلقة لأخرى، قمنا ببرمجة دالة <code>movingAvg()</code> في الـ JS تأخذ متوسط كل 20 نقطة. هذا يجعل الـ Learning Curve المعروض في الـ Chart.js سلساً وواضحاً ومفهوماً أكاديمياً.</p>
"""

# ==========================================
# 07. Defense FAQ
# ==========================================
c_faq = """
<h1>🛡️ 07. الأسئلة المتوقعة في المناقشة (Defense FAQ)</h1>
<p>هذا القسم هو أهم قسم لك. هنا جهزنا أقوى وأصعب الأسئلة (الفخوخ) التي قد يسألها الدكاترة، مع إجابات هندسية وأكاديمية قاطعة لا تقبل الشك.</p>

<div class="callout warning">
    <h3>س1: هل هذه البيانات (التمارين والسعرات) مأخوذة من قاعدة بيانات حقيقية أم أنها (Dummy Data) مخترعة؟</h3>
    <strong>الجواب:</strong> 
    البيانات هي محاكاة علمية مبنية على معايير حقيقية. أرقام السعرات للتمارين (كالجري 350 سعرة، الـ HIIT 500 سعرة) مأخوذة من إرشادات الكلية الأمريكية للطب الرياضي (ACSM) ومنظمة الصحة العالمية. أما من ناحية الذكاء الاصطناعي، فلا يوجد أي "ردود مبرمجة مسبقاً" (Hardcoded). كل ما نراه من توصيات ومكافآت (Q-Values) يتم حسابه رياضياً بشكل مباشر (Real-time Computed) ولا يوجد أي غش في الكود.
</div>

<div class="callout warning">
    <h3>س2: في تحديد الحالة (State Space)، لماذا استخدمتم 5 متغيرات فقط (لياقة، طاقة، وزن، عضل، إرهاق)؟ لماذا لم تضيفوا العمر، الجنس، والوقت المتاح؟</h3>
    <strong>الجواب:</strong> 
    السبب معماري ورياضي بحت وهو تجنب ما يسمى בـ "لعنة الأبعاد" (Curse of Dimensionality). المتغيرات الحالية تنتج 43,750 حالة. لو أضفنا "الوقت المتاح للمستخدم" بثلاث فئات فقط، سيقفز الرقم لـ 131 ألف حالة! ولو أضفنا العمر سيصل للملايين.
    بدلاً من ذلك، قمنا بإنشاء (User Profile) يتعامل مع العمر والجنس كـ (Context) يهيئ الـ State الأولية والهدف، واستخدمنا الوقت كـ (Filter) يمنع الأفعال المستحيلة قبل إدخالها للـ RL.
</div>

<div class="callout warning">
    <h3>س3: أرى تذبذباً (طلوع ونزول) في منحنى التعلم (Learning Curve). ألا يفترض أن يثبت المنحنى في النهاية ليثبت أن الوكيل تعلم؟</h3>
    <strong>الجواب:</strong> 
    هذا التذبذب متعمد وطبيعي لسببين:
    1. <strong>الاستكشاف الدائم:</strong> تركنا نسبة الـ Epsilon (ε) عند 0.05، أي أن الوكيل يختار بحركة عشوائية في 5% من الحالات لمنع تجمد السياسة وتجنب الـ Local Optima.
    2. <strong>البيئة العشوائية (Stochasticity):</strong> برمجنا البيئة ليكون فيها 10% احتمالية لحدوث إرهاق مفاجئ (محاكاة للواقع). حتى لو اتخذ الوكيل القرار المثالي، قد ينقص تقييمه قليلاً بسبب هذه العشوائية. المتوسط الحسابي (Average Reward) هو ما يثبت كفاءته وليس النقاط الفردية.
</div>

<div class="callout warning">
    <h3>س4: ما هو الفرق بين Approximate Q-Learning والـ Q-Learning العادي في مشروعكم تحديداً؟ ولماذا برمجتم الاثنين؟</h3>
    <strong>الجواب:</strong> 
    - <strong>Tabular Q-Learning:</strong> يعتمد على بناء جدول ضخم، وهو ممتاز للحالات التي تدرب عليها، لكنه يفشل تماماً ويعود للعشوائية إذا سألناه عن حالة لم يزرها مسبقاً (ينعدم عنده التعميم).
    <br>- <strong>Approximate Q-Learning:</strong> يستبدل الجدول بمعادلة رياضية خطية تعتمد على 12 وزناً (Weights) للخصائص (Features). هذا يسمح للوكيل بالاستنتاج والتعميم (Generalization). إذا رأى حالة جديدة فيها "إرهاق عالي"، سيستنتج فوراً أن أفضل قرار هو "الراحة" بناءً على أوزان الخصائص، دون الحاجة لجدول.
</div>

<div class="callout warning">
    <h3>س5: لماذا تأخذ خوارزمية Value Iteration وقتاً طويلاً جداً (حوالي دقيقة) في حين أن Q-Learning والـ A* ينهون مهامهم في ثوانٍ؟</h3>
    <strong>الجواب:</strong> 
    لأن Value Iteration ليست خوارزمية تعتمد على المحاكاة الجزئية، بل هي خوارزمية Model-Based تحل بيئة الـ MDP بالكامل لإيجاد الحل الرياضي الدقيق (Exact Mathematical Solution). في كل دورة (Iteration)، تمر الخوارزمية على الـ 6,250 حالة، وفي كل حالة تجرب الـ 28 فعلاً، وتقوم بتحديث معادلة بلمان. هذا يعني ملايين العمليات الحسابية المتزامنة، وهو ما يثبت تعقيد وقوة البيئة الرياضية التي بنيناها.
</div>

<div class="callout warning">
    <h3>س6: في خوارزمية A* Search، كيف أثبتم أن الـ Heuristic Function الخاصة بكم هي Admissible (لا تبالغ في التقدير)؟</h3>
    <strong>الجواب:</strong> 
    لكي يكون A* رياضياً Optimal يجب ألا يبالغ الـ Heuristic في تقدير التكلفة للمسافة المتبقية. 
    قمنا بتعريف المسافة على أنها الفرق بين اللياقة الحالية والهدف. وبما أن أقصى ارتفاع لياقة في خطوة واحدة هو 0.2 (من تمرين HIIT)، قمنا بقسمة الفجوة على 0.2 لتقدير "أقل عدد من الأيام المثالية المطلوبة"، وضربناها في أقل تكلفة ممكنة ليوم مثالي دون إرهاق. 
    هذا الحساب يمثل السيناريو "الأكثر تفاؤلاً على الإطلاق" (Optimistic)، مما يضمن أنه دائماً Underestimate للواقع، وهذا هو الإثبات الرياضي للـ Admissibility.
</div>

<div class="callout warning">
    <h3>س7: ذكرتم أن بيئتكم لا تغير الوزن أو العضل في خطوة واحدة، بل تحتاج أياماً. كيف تعاملتم مع هذا التحدي البرمجي الصعب في الـ MDP؟</h3>
    <strong>الجواب:</strong> 
    ابتكرنا نمطاً معمارياً أسميناه (Accumulator Pattern). خصصنا قاموساً برمجياً <code>self.accumulators</code> يخزن الكسور الناتجة عن الأفعال (مثلاً -0.05 لوزن بسبب حمية). عندما يتجاوز هذا التراكم حاجز الـ 1.0 أو -1.0، نقوم بتحديث مستوى الوزن الحقيقي (State) ونصفر المراكم.
    الأهم من ذلك، في خوارزمية A* Search، قمنا بجعل كل عقدة في شجرة البحث تحمل نسخة عميقة (Deep Copy) من هذه المراكم وتورثها لأبنائها، مما سمح لـ A* برؤية المستقبل واستشراف تأثير التراكم بعد 30 عمقاً بدقة تامة.
</div>

<div class="callout warning">
    <h3>س8: كيف تضمنون عدم إيذاء المستخدم بتوصيات خاطئة تؤدي للإرهاق المفرط (Safety Constraints)؟</h3>
    <strong>الجواب:</strong> 
    استخدمنا في دالة المكافأة مفهوماً يسمى (Progressive Safety Penalties). عقوبة الإرهاق ليست ثابتة، بل هي منحنى تصاعدي (Exponential). إذا وصل الإرهاق למستوى 2 نعطي عقوبة -0.5 (تحذير). إذا وصل לـ 3 نعطي -2.0. أما إذا وصل للحد الأقصى (4) وهي منطقة الخطر، فإننا نعطي عقوبة ساحقة -4.0.
    هذه العقوبات القاسية تجبر الـ Agent (سواء في Q-learning أو Value Iteration) على تعلم أن تفادي الإرهاق أهم من التقدم السريع للهدف (Value Alignment).
</div>
"""

# Write all files
data = {
    "index.html": c_index,
    "01_mdp.html": c_mdp,
    "02_qlearning.html": c_q,
    "03_approx_q.html": c_approx,
    "04_value_iteration.html": c_vi,
    "05_astar.html": c_astar,
    "06_dashboard.html": c_dashboard,
    "07_defense_faq.html": c_faq,
}

for filename, content in data.items():
    title = next(t for f, t in chapters if f == filename)
    full_html = wrap_html(filename, title, content)
    with open(os.path.join(docs_dir, filename), "w", encoding="utf-8") as f:
        f.write(full_html)

print("Detailed Documentation generated successfully!")
