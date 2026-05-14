/**
 * AI Health & Fitness Dashboard — Interactive Controller
 * Handles API calls, chart rendering, and UI updates.
 */

// ═══════════════ CHART INSTANCES ═══════════════
let learningCurveChart = null;
let fitnessProgressChart = null;

// ═══════════════ STATE ═══════════════
let searchResults = null;
let qTrainingData = null;
let approxTrainingData = null;
let viData = null;
let isAutoSimulating = false;

// ═══════════════ CHART CONFIGURATION (Impeccable Register) ═══════════════
// Restrained OKLCH-derived palette, minimal grid, bottom legend, no fill.
const PALETTE = {
    indigo:  '#5b3ac2',   // oklch(0.45 0.18 280) — primary accent
    teal:    '#1590a8',   // oklch(0.58 0.12 200) — secondary
    emerald: '#1a8a68',   // oklch(0.54 0.13 160) — tertiary
    coral:   '#b94444',   // oklch(0.52 0.15 25)  — danger/fatigue
    grid:    'rgba(0,0,0,0.04)',
    tick:    '#8b8fa3',
    legend:  '#5c5f73',
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300, easing: 'easeOutQuart' },
    interaction: { mode: 'index', intersect: false },
    plugins: {
        legend: {
            position: 'bottom',
            labels: {
                color: PALETTE.legend,
                font: { family: 'Inter, sans-serif', size: 11, weight: 500 },
                padding: 16,
                usePointStyle: true,
                pointStyleWidth: 10,
            }
        },
        tooltip: {
            backgroundColor: 'rgba(30,30,50,0.92)',
            titleFont: { family: 'Inter, sans-serif', size: 12 },
            bodyFont: { family: 'Inter, sans-serif', size: 11 },
            padding: 10,
            cornerRadius: 8,
            displayColors: true,
            boxPadding: 4,
        }
    },
    scales: {
        x: {
            grid: { display: false },
            ticks: {
                color: PALETTE.tick,
                font: { family: 'Inter, sans-serif', size: 10 },
                maxTicksLimit: 12,
                maxRotation: 0,
            },
            border: { color: PALETTE.grid },
        },
        y: {
            grid: { color: PALETTE.grid, lineWidth: 1 },
            ticks: {
                color: PALETTE.tick,
                font: { family: 'Inter, sans-serif', size: 10 },
                padding: 8,
            },
            border: { display: false },
        }
    }
};

// ═══════════════ INITIALIZATION ═══════════════
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupProfile();
});

function initCharts() {
    // Learning Curve Chart
    const lcCtx = document.getElementById('learningCurveChart').getContext('2d');
    learningCurveChart = new Chart(lcCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Q-Learning',
                    data: [],
                    borderColor: PALETTE.indigo,
                    backgroundColor: PALETTE.indigo,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.35,
                    fill: false,
                },
                {
                    label: 'Approx Q-Learning',
                    data: [],
                    borderColor: PALETTE.teal,
                    backgroundColor: PALETTE.teal,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.35,
                    fill: false,
                }
            ]
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                title: { display: false }
            }
        }
    });

    // Fitness Progress Chart
    const fpCtx = document.getElementById('fitnessProgressChart').getContext('2d');
    fitnessProgressChart = new Chart(fpCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Fitness',
                    data: [],
                    borderColor: PALETTE.indigo,
                    backgroundColor: PALETTE.indigo,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.3,
                },
                {
                    label: 'Energy',
                    data: [],
                    borderColor: PALETTE.teal,
                    backgroundColor: PALETTE.teal,
                    borderWidth: 1.5,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.3,
                },
                {
                    label: 'Muscle',
                    data: [],
                    borderColor: PALETTE.emerald,
                    backgroundColor: PALETTE.emerald,
                    borderWidth: 1.5,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.3,
                },
                {
                    label: 'Fatigue',
                    data: [],
                    borderColor: PALETTE.coral,
                    backgroundColor: PALETTE.coral,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                    borderDash: [4, 3],
                }
            ]
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    min: 0,
                    max: 9,
                    ticks: {
                        ...CHART_DEFAULTS.scales.y.ticks,
                        stepSize: 1,
                    }
                }
            }
        }
    });
}

// ═══════════════ API HELPERS ═══════════════
async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) options.body = JSON.stringify(body);

    const res = await fetch(endpoint, options);
    return res.json();
}

function showLoading(text = 'Processing...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-overlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

// ═══════════════ CUSTOM INPUTS ═══════════════
function toggleCustomInputs() {
    const preset = document.getElementById('preset-select').value;
    document.getElementById('custom-inputs').style.display = preset === 'custom' ? 'block' : 'none';
}

// ═══════════════ SETUP ═══════════════
async function setupProfile() {
    const preset = document.getElementById('preset-select').value;
    let body;

    if (preset === 'custom') {
        // Auto-calculate weight level from BMI
        const weightKg = parseInt(document.getElementById('sl-bodyweight').value);
        const heightCm = parseInt(document.getElementById('sl-height').value);
        const bmi = weightKg / ((heightCm / 100) ** 2);
        let weightLevel;
        if (bmi < 18.5) weightLevel = 0;        // Very Underweight
        else if (bmi < 22) weightLevel = 1;      // Underweight
        else if (bmi < 25) weightLevel = 2;      // Normal
        else if (bmi < 30) weightLevel = 3;      // Overweight
        else weightLevel = 4;                     // Obese
        document.getElementById('sl-weight').value = weightLevel;

        body = {
            custom: {
                name: 'Custom User',
                goal_type: document.getElementById('goal-select').value,
                age: parseInt(document.getElementById('sl-age').value),
                weight_kg: weightKg,
                height_cm: heightCm,
                time_available: parseInt(document.getElementById('time-select').value),
                equipment: document.getElementById('equipment-select').value,
                initial_state: {
                    fitness: parseInt(document.getElementById('sl-fitness').value),
                    energy: parseInt(document.getElementById('sl-energy').value),
                    weight: weightLevel,
                    muscle: parseInt(document.getElementById('sl-muscle').value),
                    fatigue: parseInt(document.getElementById('sl-fatigue').value),
                }
            }
        };
    } else {
        body = { preset };
    }

    const data = await apiCall('/api/setup', 'POST', body);

    if (data.status === 'ok') {
        updateProfileInfo(data.profile);
        updateStateDisplay(data.state);
        resetCharts();
        clearSimLog();
    }
}

function updateProfileInfo(profile) {
    document.getElementById('info-name').textContent = profile.name;
    document.getElementById('info-goal').textContent = profile.goal_type.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    document.getElementById('info-age').textContent = profile.age;
    document.getElementById('info-weight').textContent = profile.weight_kg + ' kg';
    // HR_max = 220 - age (standard formula from PERFECT Framework)
    const hrMax = 220 - profile.age;
    document.getElementById('info-hrmax').textContent = hrMax + ' bpm';
}

function updateStateDisplay(state) {
    const maxVals = { fitness: 9, energy: 4, weight: 4, muscle: 4, fatigue: 4 };
    for (const key of ['fitness', 'energy', 'weight', 'muscle', 'fatigue']) {
        const pct = (state[key] / maxVals[key]) * 100;
        document.getElementById(`state-${key}-bar`).style.width = pct + '%';
        document.getElementById(`state-${key}-label`).textContent = state[`${key}_label`];
    }
    document.getElementById('state-day').textContent = '📅 ' + state.day_label;
}

// ═══════════════ TRAINING (BATCH WITH LIVE PROGRESS) ═══════════════
async function trainAgents() {
    const totalEpisodes = parseInt(document.getElementById('episodes-input').value) || 500;
    const batchSize = 50;
    const numBatches = Math.ceil(totalEpisodes / batchSize);

    // Show progress banner (not full-screen, so charts stay visible)
    showLoading(`Training agents — 0 / ${totalEpisodes} episodes`);
    document.getElementById('progress-container').style.display = 'block';
    document.getElementById('progress-fill').style.width = '0%';

    // Scroll to charts so user sees live learning curve
    setTimeout(() => {
        document.getElementById('charts-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);

    let allQRewards = [];
    let allApproxRewards = [];

    try {
        for (let i = 0; i < numBatches; i++) {
            const eps = Math.min(batchSize, totalEpisodes - i * batchSize);
            const resume = i > 0;

            // Build request body — include hyperparams on first batch
            const body = { episodes: eps, agent: 'both', resume: resume };
            if (!resume) {
                body.hyperparams = {
                    alpha: parseFloat(document.getElementById('param-alpha').value) || 0.1,
                    gamma: parseFloat(document.getElementById('param-gamma').value) || 0.95,
                    epsilon_start: parseFloat(document.getElementById('param-epsilon-start').value) || 1.0,
                    epsilon_end: parseFloat(document.getElementById('param-epsilon-end').value) || 0.05,
                };
            }

            const data = await apiCall('/api/train', 'POST', body);

            if (data.status === 'ok') {
                // Accumulate rewards
                if (data.results.q_learning)
                    allQRewards = allQRewards.concat(data.results.q_learning.episode_rewards);
                if (data.results.approximate)
                    allApproxRewards = allApproxRewards.concat(data.results.approximate.episode_rewards);

                // Update progress
                const done = (i + 1) * batchSize;
                const pct = Math.min(100, Math.round((done / totalEpisodes) * 100));
                document.getElementById('progress-fill').style.width = pct + '%';
                document.getElementById('progress-pct').textContent = pct + '%';
                document.getElementById('progress-episodes').textContent = `${Math.min(done, totalEpisodes)} / ${totalEpisodes} episodes`;
                document.getElementById('loading-text').textContent = `Training agents — ${Math.min(done, totalEpisodes)} / ${totalEpisodes} episodes`;

                // Live metrics
                if (data.results.q_learning) {
                    document.getElementById('progress-reward').textContent =
                        `🎯 Avg Reward: ${data.results.q_learning.avg_reward_last_100}`;
                    document.getElementById('progress-epsilon').textContent =
                        `ε: ${data.results.q_learning.final_epsilon}`;
                }

                // Update chart live
                updateLearningCurveLive(allQRewards, allApproxRewards);

                // Store final data
                qTrainingData = data.results.q_learning;
                approxTrainingData = data.results.approximate;
            }
        }

        // Final update
        if (qTrainingData) updateMetrics({ q_learning: qTrainingData, approximate: approxTrainingData });
        updateRecommendations();
        updateComparison();

    } catch (err) {
        console.error('Training error:', err);
    }

    document.getElementById('progress-container').style.display = 'none';
    hideLoading();
}

function updateLearningCurveLive(qRewards, approxRewards) {
    const len = Math.max(qRewards.length, approxRewards.length);
    const labels = Array.from({ length: len }, (_, i) => i + 1);

    const movingAvg = (arr, w = 20) => arr.map((_, i) => {
        const start = Math.max(0, i - w + 1);
        const slice = arr.slice(start, i + 1);
        return slice.reduce((a, b) => a + b, 0) / slice.length;
    });

    learningCurveChart.data.labels = labels;
    if (qRewards.length) learningCurveChart.data.datasets[0].data = movingAvg(qRewards);
    if (approxRewards.length) learningCurveChart.data.datasets[1].data = movingAvg(approxRewards);
    learningCurveChart.update('none'); // no animation for speed
}

function updateMetrics(results) {
    const q = results.q_learning || {};
    document.getElementById('metric-episodes').textContent = q.total_episodes || 0;
    document.getElementById('metric-avg-reward').textContent = q.avg_reward_last_100 || '--';
    document.getElementById('metric-best-reward').textContent = q.best_reward || '--';
    document.getElementById('metric-epsilon').textContent = q.final_epsilon || '1.0';
    document.getElementById('metric-states').textContent = q.states_explored || 0;
    // Research-backed metrics
    document.getElementById('metric-stability').textContent =
        q.policy_stability !== undefined ? q.policy_stability + '%' : '--';
    document.getElementById('metric-goal-rate').textContent =
        q.goal_achievement_rate !== undefined ? q.goal_achievement_rate + '%' : '--';
    document.getElementById('metric-coverage').textContent =
        q.state_coverage !== undefined ? q.state_coverage + '%' : '--';
}

function updateLearningCurve(results) {
    const q = results.q_learning;
    const approx = results.approximate;

    if (!q && !approx) return;

    const episodes = q ? q.episode_rewards.length : approx.episode_rewards.length;
    const labels = Array.from({ length: episodes }, (_, i) => i + 1);

    // Compute moving average (window=20)
    const movingAvg = (arr, w = 20) => {
        return arr.map((_, i) => {
            const start = Math.max(0, i - w + 1);
            const slice = arr.slice(start, i + 1);
            return slice.reduce((a, b) => a + b, 0) / slice.length;
        });
    };

    learningCurveChart.data.labels = labels;
    if (q) learningCurveChart.data.datasets[0].data = movingAvg(q.episode_rewards);
    if (approx) learningCurveChart.data.datasets[1].data = movingAvg(approx.episode_rewards);
    learningCurveChart.update();
}

// ═══════════════ A* SEARCH ═══════════════
async function runSearch() {
    const depth = parseInt(document.getElementById('search-depth').value) || 14;
    showLoading('Running A* Search...');

    try {
        const data = await apiCall('/api/search', 'POST', { max_depth: depth });

        if (data.status === 'ok') {
            searchResults = data.result;
            displaySearchPlan(data.result);
            updateComparison();
        }
    } catch (err) {
        console.error('Search error:', err);
    }
    hideLoading();
}

function displaySearchPlan(result) {
    const container = document.getElementById('search-plan');

    let infoHtml = `
        <div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">
            <div class="param"><span>Status</span><code>${result.status === 'goal_reached' ? '✅ Goal Reached' : '⚠️ Depth Limited'}</code></div>
            <div class="param"><span>Plan Length</span><code>${result.plan_length} days</code></div>
            <div class="param"><span>Est. Reward</span><code>${result.total_estimated_reward}</code></div>
            <div class="param"><span>Nodes Expanded</span><code>${result.nodes_expanded}</code></div>
            <div class="param"><span>Search Time</span><code>${result.search_time}s</code></div>
        </div>
    `;

    let planHtml = '<div class="search-plan-list">';
    for (const day of result.plan) {
        planHtml += `
            <div class="plan-day">
                <div class="plan-day-num">Day ${day.day}</div>
                <div class="plan-day-action">${day.workout_emoji} ${day.workout}</div>
                <div class="plan-day-action">${day.nutrition_emoji} ${day.nutrition}</div>
            </div>
        `;
    }
    planHtml += '</div>';

    container.innerHTML = infoHtml + planHtml;
}

// ═══════════════ RECOMMENDATIONS ═══════════════
async function updateRecommendations() {
    try {
        const data = await apiCall('/api/recommend');
        if (data.status === 'ok') {
            displayRecommendation('q-recommendation', data.recommendations.q_learning, 'Q-Learning');
            displayRecommendation('approx-recommendation', data.recommendations.approximate, 'Approx Q');
            if (data.recommendations.value_iteration) {
                displayRecommendation('vi-recommendation', data.recommendations.value_iteration, 'Value Iteration');
            }
        }
    } catch (err) {
        console.error('Recommendation error:', err);
    }
}

// ═══════════════ VALUE ITERATION ═══════════════
async function runValueIteration() {
    showLoading('Running Value Iteration (solving MDP with Bellman equations)...');
    try {
        const data = await apiCall('/api/value_iteration', 'POST', { iterations: 50 });
        if (data.status === 'ok') {
            viData = data.result;
            displayVIResult(data.result);
            updateRecommendations();
            updateComparison();
        }
    } catch (err) {
        console.error('VI error:', err);
    }
    hideLoading();
}

function displayVIResult(result) {
    const container = document.getElementById('vi-recommendation');
    container.innerHTML = `
        <div style="display:flex;gap:0.75rem;margin-bottom:1rem;flex-wrap:wrap;">
            <div class="param"><span>Converged</span><code>${result.converged ? '✅ Yes' : '⚠️ No'}</code></div>
            <div class="param"><span>Iterations</span><code>${result.iterations}</code></div>
            <div class="param"><span>States</span><code>${result.num_states}</code></div>
            <div class="param"><span>Time</span><code>${result.solve_time}s</code></div>
            <div class="param"><span>Max V*</span><code>${result.max_value}</code></div>
            <div class="param"><span>Avg V*</span><code>${result.avg_value}</code></div>
        </div>
        <div class="rec-placeholder" style="padding:0.5rem;font-style:normal;color:var(--text-secondary);font-size:0.8rem;">
            ✅ Optimal policy computed for all ${result.num_states} states via Bellman equations
        </div>
    `;
}

function displayRecommendation(containerId, rec, label) {
    const container = document.getElementById(containerId);
    const desc = rec.action_description;

    // Decode action index → workout + nutrition names
    const WORKOUTS = ['😴 Rest', '🚶 Walking', '🏃 Jogging', '🔥 HIIT', '🏋️ Strength', '🏊 Swimming', '🚴 Cycling'];
    const NUTRITIONS = ['📉 Deficit', '⚖️ Maintenance', '📈 Surplus', '🥩 High Protein'];
    const NUM_NUT = 4;
    function decodeAction(actionIdx) {
        const w = Math.floor(actionIdx / NUM_NUT);
        const n = actionIdx % NUM_NUT;
        return `${WORKOUTS[w] || '?'} + ${NUTRITIONS[n] || '?'}`;
    }

    const alternatives = rec.top_5_actions.slice(1, 4).map(a =>
        `<div style="display:flex;justify-content:space-between;padding:0.2rem 0;border-bottom:1px solid var(--border);">
            <span>${decodeAction(a.action)}</span>
            <code style="font-size:0.75rem;color:var(--text-3);">${a.q_value}</code>
        </div>`
    ).join('');

    container.innerHTML = `
        <div class="rec-action">
            <div class="rec-action-icon">${desc.workout_emoji}</div>
            <div class="rec-action-details">
                <h4>${desc.workout} + ${desc.nutrition}</h4>
                <p>${desc.nutrition_emoji} ${desc.nutrition} nutrition plan</p>
            </div>
            <div class="rec-qvalue">${(rec.q_value ?? rec.v_value).toFixed(2)}</div>
        </div>
        <div style="font-size:0.8rem;margin-top:0.5rem;">
            <strong style="color:var(--text-2);">Alternatives:</strong>
            <div style="margin-top:0.3rem;">${alternatives}</div>
        </div>
    `;
}

// ═══════════════ SIMULATION ═══════════════
async function simulateStep() {
    try {
        const data = await apiCall('/api/simulate', 'POST', { agent: 'q_learning' });

        if (data.status === 'ok') {
            updateStateDisplay(data.step.new_state);
            addLogEntry(data.step);
            updateFitnessProgress(data.history);
            updateSimDay(data.step.step);
            updateRecommendations();

            if (data.done) {
                addLogEntry({ step: '🎉', action: { description: 'GOAL REACHED!' }, reward: data.total_reward, special: true });
            }
            if (data.truncated) {
                addLogEntry({ step: '⏱️', action: { description: 'Episode ended (max steps)' }, reward: data.total_reward, special: true });
            }
        }
    } catch (err) {
        console.error('Simulate error:', err);
    }
}

async function autoSimulate() {
    if (isAutoSimulating) {
        isAutoSimulating = false;
        document.getElementById('btn-auto-sim').innerHTML = '<span>⏩</span> Auto Simulate';
        return;
    }

    isAutoSimulating = true;
    document.getElementById('btn-auto-sim').innerHTML = '<span>⏹️</span> Stop';

    for (let i = 0; i < 30 && isAutoSimulating; i++) {
        await simulateStep();
        await new Promise(r => setTimeout(r, 300));
    }

    isAutoSimulating = false;
    document.getElementById('btn-auto-sim').innerHTML = '<span>⏩</span> Auto Simulate';
}

function updateFitnessProgress(history) {
    const labels = history.map((_, i) => `Day ${i}`);
    fitnessProgressChart.data.labels = labels;
    fitnessProgressChart.data.datasets[0].data = history.map(h => h.fitness);
    fitnessProgressChart.data.datasets[1].data = history.map(h => h.energy);
    fitnessProgressChart.data.datasets[2].data = history.map(h => h.muscle);
    fitnessProgressChart.data.datasets[3].data = history.map(h => h.fatigue);
    fitnessProgressChart.update();
}

function updateSimDay(day) {
    document.getElementById('metric-sim-day').textContent = day;
}

function addLogEntry(step) {
    const logEl = document.getElementById('sim-log');
    if (logEl.querySelector('.log-empty')) logEl.innerHTML = '';

    const isPositive = step.reward >= 0;
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    if (step.special) {
        entry.innerHTML = `
            <span class="log-step">${step.step}</span>
            <span class="log-action">${step.action.description}</span>
            <span class="log-reward ${isPositive ? 'positive' : 'negative'}">${step.reward >= 0 ? '+' : ''}${step.reward}</span>
        `;
    } else {
        entry.innerHTML = `
            <span class="log-step">Day ${step.step}</span>
            <span class="log-action">${step.action.workout_emoji} ${step.action.workout} + ${step.action.nutrition_emoji} ${step.action.nutrition}</span>
            <span class="log-reward ${isPositive ? 'positive' : 'negative'}">${isPositive ? '+' : ''}${step.reward}</span>
        `;
    }

    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
}

function clearSimLog() {
    document.getElementById('sim-log').innerHTML = '<div class="log-empty">Each simulation step picks the agent\'s best action for the current state and advances one day. Click "Simulate Step" to begin.</div>';
}

// ═══════════════ RESET ═══════════════
async function resetEnv() {
    const data = await apiCall('/api/reset', 'POST');
    if (data.status === 'ok') {
        updateStateDisplay(data.state);
        clearSimLog();
        document.getElementById('metric-sim-day').textContent = '0';

        // Reset fitness progress chart
        fitnessProgressChart.data.labels = [];
        fitnessProgressChart.data.datasets.forEach(ds => ds.data = []);
        fitnessProgressChart.update();
    }
}

function resetCharts() {
    learningCurveChart.data.labels = [];
    learningCurveChart.data.datasets.forEach(ds => ds.data = []);
    learningCurveChart.update();

    fitnessProgressChart.data.labels = [];
    fitnessProgressChart.data.datasets.forEach(ds => ds.data = []);
    fitnessProgressChart.update();
}

// ═══════════════ COMPARISON TABLE ═══════════════
function updateComparison() {
    if (!qTrainingData && !searchResults) return;

    document.getElementById('comparison-section').style.display = 'block';
    const tbody = document.getElementById('comparison-tbody');

    const qAvg = qTrainingData ? qTrainingData.avg_reward_last_100 : '--';
    const aAvg = approxTrainingData ? approxTrainingData.avg_reward_last_100 : '--';
    const viAvg = viData ? viData.avg_value : '--';
    const sReward = searchResults ? searchResults.total_estimated_reward : '--';

    const qBest = qTrainingData ? qTrainingData.best_reward : '--';
    const aBest = approxTrainingData ? approxTrainingData.best_reward : '--';
    const viMax = viData ? viData.max_value : '--';

    const qTime = qTrainingData ? qTrainingData.training_time + 's' : '--';
    const aTime = approxTrainingData ? approxTrainingData.training_time + 's' : '--';
    const viTime = viData ? viData.solve_time + 's' : '--';
    const sTime = searchResults ? searchResults.search_time + 's' : '--';

    const qStates = qTrainingData ? qTrainingData.states_explored : '--';
    const viStates = viData ? viData.num_states : '--';
    const sNodes = searchResults ? searchResults.nodes_expanded : '--';

    tbody.innerHTML = `
        <tr><td>Avg/Max Value</td><td>${qAvg}</td><td>${aAvg}</td><td>${viAvg}</td><td>${sReward}</td></tr>
        <tr><td>Best Reward</td><td>${qBest}</td><td>${aBest}</td><td>${viMax}</td><td>${sReward}</td></tr>
        <tr><td>Computation Time</td><td>${qTime}</td><td>${aTime}</td><td>${viTime}</td><td>${sTime}</td></tr>
        <tr><td>States/Nodes</td><td>${qStates}</td><td>N/A (weights)</td><td>${viStates}</td><td>${sNodes}</td></tr>
        <tr><td>Needs Model?</td><td>❌ No</td><td>❌ No</td><td>✅ Yes</td><td>✅ Yes</td></tr>
        <tr><td>Handles Stochasticity</td><td>✅ Yes</td><td>✅ Yes</td><td>⚠️ Deterministic</td><td>❌ No</td></tr>
        <tr><td>Adaptability</td><td>⬆️ High</td><td>⬆️ High</td><td>⬇️ Low</td><td>⬇️ Low</td></tr>
        <tr><td>Generalization</td><td>⬇️ Low</td><td>⬆️ High</td><td>⬇️ Low</td><td>⬇️ Low</td></tr>
    `;
}
