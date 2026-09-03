/**
 * RiskLedger — Enterprise Payment Risk Intelligence Console
 * Frontend State Engine & Real-Time API Connector
 */

let currentNav = 'overview';
let rawTxnsData = [];
let filteredTxns = [];
let evalData = null;
let riskBudgetData = null;
let simDebounceTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  await fetchAllData();
  setupEventListeners();
  renderActiveView();
}

async function fetchAllData() {
  try {
    const [evalRes, txnsRes, budgetRes, threshRes] = await Promise.all([
      fetch('/api/evaluate'),
      fetch('/api/transactions?limit=150'),
      fetch('/api/risk-budget'),
      fetch('/api/threshold-analysis'),
    ]);

    if (evalRes.ok) evalData = await evalRes.json();
    if (threshRes.ok) {
      const threshData = await threshRes.json();
      if (!evalData) evalData = {};
      evalData.cost_curve = threshData.cost_curve;
      evalData.operating_threshold = threshData.operating_threshold;
    }

    if (txnsRes.ok) {
      const data = await txnsRes.json();
      rawTxnsData = data.transactions || [];
      filteredTxns = [...rawTxnsData];
      renderOverviewKPIs(data.stats);
    }
    if (budgetRes.ok) {
      riskBudgetData = await budgetRes.json();
      renderRiskBudgetView();
    }

    if (evalData) {
      renderOverviewView();
      renderOptimizerCurve(evalData);
      renderPoliciesView(evalData);
      renderEvaluationView(evalData);
      renderAuditLogView();
    }
  } catch (err) {
    console.error('Failed to load RiskLedger infrastructure data:', err);
  }
}

function setupEventListeners() {
  const searchInput = document.getElementById('global-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => handleGlobalSearch(e.target.value));
  }
}

/* --- NAVIGATION SYSTEM --- */
function switchNav(navId, btnEl) {
  currentNav = navId;
  document.querySelectorAll('.side-nav-btn').forEach(btn => btn.classList.remove('active'));
  
  if (btnEl) {
    btnEl.classList.add('active');
  } else {
    const defaultBtn = document.querySelector(`.side-nav-btn[data-tab="${navId}"]`);
    if (defaultBtn) defaultBtn.classList.add('active');
  }

  document.querySelectorAll('.view-panel').forEach(panel => panel.classList.remove('active'));
  const targetPanel = document.getElementById(`view-${navId}`);
  if (targetPanel) targetPanel.classList.add('active');

  const breadcrumbEl = document.getElementById('topbar-breadcrumb');
  if (breadcrumbEl) {
    const titleMap = {
      overview: 'Overview',
      monitor: 'Risk Monitor',
      transactions: 'Transactions Ledger',
      policies: 'Risk Policies',
      riskbudget: 'Risk Budget',
      analysis: 'Decision Analysis',
      thresholds: 'Threshold Analysis',
      audit: 'Audit Log',
      evaluation: 'Model Evaluation'
    };
    breadcrumbEl.textContent = titleMap[navId] || 'Console';
  }

  renderActiveView();
}

function renderActiveView() {
  if (currentNav === 'overview') renderOverviewView();
  else if (currentNav === 'monitor') renderMonitorTable();
  else if (currentNav === 'transactions') renderTransactionsLedger();
  else if (currentNav === 'riskbudget') renderRiskBudgetView();
  else if (currentNav === 'analysis') updateSim();
  else if (currentNav === 'thresholds' && evalData) renderOptimizerCurve(evalData);
}

/* --- OVERVIEW VIEW --- */
function renderOverviewKPIs(stats) {
  if (!stats) return;

  const totalVol = stats.total_volume || 0;
  const volLakhs = (totalVol / 100000).toFixed(1);

  const txnsEl = document.getElementById('ov-txns');
  if (txnsEl) txnsEl.textContent = (stats.total_transactions || 50000).toLocaleString();

  const highEl = document.getElementById('ov-high-risk');
  if (highEl) highEl.textContent = (stats.high_risk_count || 0).toLocaleString();

  const highPctEl = document.getElementById('ov-high-pct');
  if (highPctEl) highPctEl.textContent = `${(stats.fraud_rate || 2.9).toFixed(1)}% prevalence`;

  const expEl = document.getElementById('ov-fraud-exposure');
  if (expEl) expEl.textContent = `\u20B9${volLakhs}L`;

  // Risk Distribution Bar
  const low = stats.low_risk_count || 42500;
  const med = stats.medium_risk_count || 5500;
  const high = stats.high_risk_count || 2000;
  const total = low + med + high || 1;

  const barLow = document.getElementById('dist-bar-low');
  if (barLow) barLow.style.width = `${(low / total) * 100}%`;
  const barMed = document.getElementById('dist-bar-med');
  if (barMed) barMed.style.width = `${(med / total) * 100}%`;
  const barHigh = document.getElementById('dist-bar-high');
  if (barHigh) barHigh.style.width = `${(high / total) * 100}%`;

  const cntLow = document.getElementById('dist-cnt-low');
  if (cntLow) cntLow.textContent = low.toLocaleString();
  const cntMed = document.getElementById('dist-cnt-med');
  if (cntMed) cntMed.textContent = med.toLocaleString();
  const cntHigh = document.getElementById('dist-cnt-high');
  if (cntHigh) cntHigh.textContent = high.toLocaleString();
}

function renderOverviewView() {
  if (!evalData) return;
  const op = evalData.operating_point || evalData;
  if (!op) return;

  const fpCost = evalData.false_positive_cost || (op.confusion_matrix ? op.confusion_matrix.FP * 50 : 56950);
  const fnCost = evalData.false_negative_cost || 49492;
  const totalLoss = evalData.total_cost || (fpCost + fnCost);

  const fnEl = document.getElementById('ov-imp-fn');
  if (fnEl) fnEl.textContent = `\u20B9${fnCost.toLocaleString('en-IN')}`;

  const fpEl = document.getElementById('ov-imp-fp');
  if (fpEl) fpEl.textContent = `\u20B9${fpCost.toLocaleString('en-IN')}`;

  const totalEl = document.getElementById('ov-imp-total');
  if (totalEl) totalEl.textContent = `\u20B9${(totalLoss / 100000).toFixed(2)}L`;

  const totalSubEl = document.getElementById('ov-imp-total-sub');
  if (totalSubEl) totalSubEl.textContent = `\u20B9${totalLoss.toLocaleString('en-IN')}`;

  renderDecisionActivityTable();
}

function renderDecisionActivityTable() {
  const tbody = document.getElementById('overview-activity-tbody');
  if (!tbody) return;

  const attentionTxns = rawTxnsData.filter(t => t.risk_band === 'HIGH' || t.risk_band === 'MEDIUM').slice(0, 10);
  const displayTxns = attentionTxns.length > 0 ? attentionTxns : rawTxnsData.slice(0, 10);

  tbody.innerHTML = displayTxns.map(t => {
    const score = t.risk_score;
    const band = t.risk_band || 'LOW';
    const rec = t.recommendation || 'ALLOW';
    const timeStr = t.timestamp ? t.timestamp.split(' ')[1]?.substring(0, 5) : '18:42';

    return `
      <tr onclick="inspectTxn('${t.transaction_id}')">
        <td class="mono font-bold" style="color:var(--color-primary);">${t.transaction_id}</td>
        <td class="mono" style="color:var(--text-dim);">${timeStr}</td>
        <td class="mono font-bold">\u20B9${t.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
        <td>${t.merchant_category}</td>
        <td class="mono font-bold">${score.toFixed(1)}% <span class="score-pill ${band.toLowerCase()}">${band}</span></td>
        <td><span class="rec-badge ${rec.toLowerCase()}">${rec === 'REVIEW' ? 'VERIFY' : rec}</span></td>
      </tr>
    `;
  }).join('');
}

/* --- RISK MONITOR & TABLES --- */
function renderMonitorTable() {
  const tbody = document.getElementById('monitor-tbody');
  if (!tbody) return;

  if (filteredTxns.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center p-4">No transactions match current filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = filteredTxns.slice(0, 50).map(t => {
    const score = t.risk_score;
    const band = t.risk_band || 'LOW';
    const rec = t.recommendation || 'ALLOW';
    const catPolicy = evalData && evalData.category_policies ? evalData.category_policies[t.merchant_category] : null;
    const threshVal = catPolicy ? catPolicy.threshold.toFixed(3) : '0.130';

    return `
      <tr onclick="inspectTxn('${t.transaction_id}')">
        <td class="mono font-bold" style="color:var(--color-primary);">${t.transaction_id}</td>
        <td class="mono" style="font-size:0.72rem;color:var(--text-dim);">${t.timestamp || '2026-04-30 23:45'}</td>
        <td class="mono font-bold">\u20B9${t.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
        <td>${t.merchant_category}</td>
        <td class="mono font-bold">${score.toFixed(1)}% <span class="score-pill ${band.toLowerCase()}">${band}</span></td>
        <td class="mono" style="color:var(--text-muted);">${threshVal}</td>
        <td><span class="rec-badge ${rec.toLowerCase()}">${rec === 'REVIEW' ? 'VERIFY' : rec}</span></td>
        <td style="font-size:0.72rem;color:var(--text-dim);">${t.merchant_category} Policy</td>
      </tr>
    `;
  }).join('');
}

function filterMonitor(band, btnEl) {
  document.querySelectorAll('.filter-controls .f-btn').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  if (band === 'ALL') {
    filteredTxns = [...rawTxnsData];
  } else {
    filteredTxns = rawTxnsData.filter(t => t.risk_band === band);
  }
  renderMonitorTable();
}

function renderTransactionsLedger() {
  const tbody = document.getElementById('txns-all-tbody');
  if (!tbody) return;

  tbody.innerHTML = rawTxnsData.slice(0, 50).map(t => `
    <tr onclick="inspectTxn('${t.transaction_id}')">
      <td class="mono" style="color:var(--color-primary);">${t.transaction_id}</td>
      <td class="mono" style="font-size:0.72rem;color:var(--text-dim);">${t.timestamp || '2026-04-30 23:45'}</td>
      <td class="mono font-bold">\u20B9${t.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td>${t.merchant_id || 'merchant_a'}</td>
      <td>${t.merchant_category}</td>
      <td class="mono font-bold">${t.risk_score.toFixed(1)}%</td>
      <td><span class="rec-badge ${(t.recommendation || 'ALLOW').toLowerCase()}">${t.recommendation || 'ALLOW'}</span></td>
    </tr>
  `).join('');
}

/* --- POLICIES VIEW --- */
function renderPoliciesView(data) {
  const tbody = document.getElementById('policy-cat-tbody');
  if (!tbody || !data.category_policies) return;

  const policies = data.category_policies;
  tbody.innerHTML = Object.keys(policies).sort().map(cat => {
    const p = policies[cat];
    const posture = p.threshold <= 0.15 ? 'Strict' : (p.threshold <= 0.35 ? 'Moderate' : 'Lenient');
    return `
      <tr>
        <td class="font-bold">${cat}</td>
        <td class="mono highlight">${p.threshold.toFixed(3)}</td>
        <td style="font-size:0.75rem;color:var(--text-muted);">${posture}</td>
      </tr>
    `;
  }).join('');
}

/* --- RISK BUDGET VIEW --- */
function renderRiskBudgetView() {
  if (!riskBudgetData) return;

  const b = riskBudgetData;
  const lbl = document.getElementById('rb-limit-label');
  if (lbl) lbl.textContent = `\u20B9${b.current_exposure.toLocaleString()} of \u20B9${b.daily_exposure_limit.toLocaleString()}`;
  
  const fill = document.getElementById('rb-progress-fill');
  if (fill) fill.style.width = `${b.utilization_pct}%`;

  const utilTxt = document.getElementById('rb-util-text');
  if (utilTxt) utilTxt.textContent = `${b.utilization_pct}% utilized`;

  const remTxt = document.getElementById('rb-rem-text');
  if (remTxt) remTxt.textContent = `Remaining capacity: \u20B9${b.remaining_capacity.toLocaleString()}`;

  const curExp = document.getElementById('rb-cur-exp');
  if (curExp) curExp.textContent = `\u20B9${b.current_exposure.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

  const remCap = document.getElementById('rb-rem-cap');
  if (remCap) remCap.textContent = `\u20B9${b.remaining_capacity.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

  const cnt = document.getElementById('rb-txn-cnt');
  if (cnt) cnt.textContent = b.transactions_consuming_exposure;

  const expl = document.getElementById('rb-policy-expl');
  if (expl) expl.textContent = b.policy_rule;
}

/* --- DECISION ANALYSIS & COUNTERFACTUAL SIMULATOR --- */
function updateSim() {
  clearTimeout(simDebounceTimer);
  simDebounceTimer = setTimeout(async () => {
    const amtEl = document.getElementById('sim-amount');
    const avgEl = document.getElementById('sim-avg');
    const failedEl = document.getElementById('sim-failed');
    const hourEl = document.getElementById('sim-hour');
    const distEl = document.getElementById('sim-dist');
    const catEl = document.getElementById('sim-cat');

    if (!amtEl || !avgEl) return;

    const amount = parseFloat(amtEl.value);
    const avg = parseFloat(avgEl.value);
    const failed = parseInt(failedEl.value);
    const hour = parseInt(hourEl.value);
    const dist = parseFloat(distEl.value);
    const cat = catEl.value;

    const valAmt = document.getElementById('sim-val-amount');
    if (valAmt) valAmt.textContent = `\u20B9${amount.toLocaleString()}`;
    const valAvg = document.getElementById('sim-val-avg');
    if (valAvg) valAvg.textContent = `\u20B9${avg.toLocaleString()}`;
    const valFailed = document.getElementById('sim-val-failed');
    if (valFailed) valFailed.textContent = failed;
    const valHour = document.getElementById('sim-val-hour');
    if (valHour) valHour.textContent = `${hour.toString().padStart(2, '0')}:00 UTC`;
    const valDist = document.getElementById('sim-val-dist');
    if (valDist) valDist.textContent = `${dist} km`;
    const valCat = document.getElementById('sim-val-cat');
    if (valCat) valCat.textContent = cat;

    try {
      const res = await fetch('/api/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount: amount,
          customer_avg_amount_30d: avg,
          failed_attempts_last_hour: failed,
          hour_of_day: hour,
          distance_from_usual_location_km: dist,
          merchant_category: cat,
          device_age_days: 1.5,
          is_new_device: 1,
          seconds_since_last_transaction: 45.0,
          txn_count_last_hour: 4
        })
      });

      if (res.ok) {
        const data = await res.json();
        renderSimResults(data);
      }
    } catch (e) {
      console.error('Sim score error:', e);
    }
  }, 80);
}

function renderSimResults(data) {
  const score = data.risk_score;
  const rec = data.decision || data.recommendation;
  const band = data.risk_score >= 35 ? 'HIGH' : (data.risk_score >= 14 ? 'MEDIUM' : 'LOW');

  const scoreNum = document.getElementById('sim-score-num');
  if (scoreNum) scoreNum.textContent = score.toFixed(1);
  const scoreCircle = document.getElementById('sim-score-circle');
  if (scoreCircle) scoreCircle.style.setProperty('--score-pct', `${score}%`);
  
  const recBadge = document.getElementById('sim-rec-badge');
  if (recBadge) {
    recBadge.className = `rec-badge ${rec.toLowerCase()}`;
    recBadge.textContent = rec === 'REVIEW' ? 'VERIFY' : rec;
  }

  const bandText = document.getElementById('sim-band-text');
  if (bandText) {
    bandText.textContent = `${score.toFixed(1)}% (${band})`;
    bandText.className = band === 'HIGH' ? 'text-danger' : (band === 'MEDIUM' ? 'text-warning' : 'text-success');
  }

  const probText = document.getElementById('sim-prob-text');
  if (probText) probText.textContent = `Policy Threshold: ${data.effective_threshold.toFixed(3)}`;

  renderShapHorizontalBars('sim-shap-bars', data.explanation || data.top_reasons);

  const cf = data.counterfactual;
  const cfBox = document.getElementById('sim-cf-text');
  if (cfBox) {
    if (cf && (cf.block_to_verify || cf.safe_amount_limit)) {
      const b2v = cf.block_to_verify || (cf.safe_amount_limit * 2.2);
      const v2a = cf.verify_to_allow || cf.safe_amount_limit;
      cfBox.innerHTML = `At \u20B9${b2v.toFixed(0)} \u2192 BLOCK \u2192 VERIFY<br/>At \u20B9${v2a.toFixed(0)} \u2192 VERIFY \u2192 ALLOW`;
    } else {
      cfBox.textContent = `Transaction amount sit comfortably within decision boundaries.`;
    }
  }
}

function renderShapHorizontalBars(containerId, topReasons) {
  const container = document.getElementById(containerId);
  if (!container || !topReasons) return;

  const maxVal = Math.max(...topReasons.map(r => Math.abs(r.shap_value)), 0.01);

  container.innerHTML = topReasons.map(r => {
    const val = r.shap_value;
    const isPos = val >= 0;
    const pct = Math.min(100, Math.max(10, (Math.abs(val) / maxVal) * 100));
    const sign = isPos ? '+' : '';
    const colorClass = isPos ? 'pos' : 'neg';

    return `
      <div class="shap-h-row">
        <div class="shap-h-meta">
          <span class="mono" style="color:var(--text-muted);">${r.feature}</span>
          <span class="mono font-bold ${isPos ? 'text-danger' : 'text-success'}">${sign}${val.toFixed(3)}</span>
        </div>
        <div class="shap-h-bar-bg">
          <div class="shap-h-fill ${colorClass}" style="width: ${pct}%;"></div>
        </div>
      </div>
    `;
  }).join('');
}

/* --- THRESHOLD ANALYSIS & SVG COST CURVE --- */
function renderOptimizerCurve(data) {
  if (!data || !data.cost_curve) return;
  drawCostCurveSVG(data.cost_curve, data.operating_threshold || 0.13);
}

function drawCostCurveSVG(curve, activeThreshold) {
  const svg = document.getElementById('cost-curve-svg');
  if (!svg || !curve || curve.length === 0) return;

  const width = 800;
  const height = 170;
  const padding = 20;

  const costs = curve.map(c => c.total_cost);
  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);

  const points = curve.map((c, i) => {
    const x = padding + (i / (curve.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((c.total_cost - minCost) / (maxCost - minCost || 1)) * (height - 2 * padding);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const activeIdx = curve.findIndex(c => Math.abs(c.threshold - activeThreshold) < 0.02);
  let activeCx = width / 2, activeCy = height / 2;
  if (activeIdx >= 0) {
    activeCx = padding + (activeIdx / (curve.length - 1)) * (width - 2 * padding);
    activeCy = height - padding - ((curve[activeIdx].total_cost - minCost) / (maxCost - minCost || 1)) * (height - 2 * padding);
  }

  svg.innerHTML = `
    <defs>
      <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
      </linearGradient>
    </defs>
    <polygon fill="url(#costGrad)" points="${padding},${height - padding} ${points} ${width - padding},${height - padding}" />
    <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="${points}" />
    <line x1="${activeCx}" y1="${padding}" x2="${activeCx}" y2="${height - padding}" stroke="#ef4444" stroke-dasharray="4,4" stroke-width="1.5" />
    <circle cx="${activeCx}" cy="${activeCy}" r="5" fill="#ef4444" stroke="#ffffff" stroke-width="2" />
  `;
}

function updateOptimizerCurve(val) {
  const t = parseFloat(val);
  const threshValEl = document.getElementById('opt-thresh-val');
  if (threshValEl) threshValEl.textContent = t.toFixed(3);
  if (!evalData || !evalData.cost_curve) return;

  const closest = evalData.cost_curve.reduce((prev, curr) => 
    Math.abs(curr.threshold - t) < Math.abs(prev.threshold - t) ? curr : prev
  );

  const fpCostEl = document.getElementById('opt-cost-fp');
  if (fpCostEl) fpCostEl.textContent = `\u20B9${closest.cost_fp.toLocaleString()}`;
  const fnCostEl = document.getElementById('opt-cost-fn');
  if (fnCostEl) fnCostEl.textContent = `\u20B9${closest.cost_fn.toLocaleString()}`;
  const totalCostEl = document.getElementById('opt-cost-total');
  if (totalCostEl) totalCostEl.textContent = `\u20B9${closest.total_cost.toLocaleString()}`;

  const precEl = document.getElementById('opt-prec');
  if (precEl) precEl.textContent = closest.precision.toFixed(3);
  const recEl = document.getElementById('opt-rec');
  if (recEl) recEl.textContent = closest.recall.toFixed(3);
  const fpEl = document.getElementById('opt-fp');
  if (fpEl) fpEl.textContent = closest.false_positives.toLocaleString();

  drawCostCurveSVG(evalData.cost_curve, closest.threshold);
}

/* --- AUDIT LOG VIEW --- */
function renderAuditLogView() {
  const tbody = document.getElementById('audit-tbody');
  if (!tbody) return;

  tbody.innerHTML = rawTxnsData.slice(0, 30).map(t => {
    const score = t.risk_score;
    const rec = t.recommendation || 'ALLOW';
    const reason = t.top_reasons && t.top_reasons[0] ? t.top_reasons[0].explanation : 'Baseline signature within bounds';

    return `
      <tr onclick="inspectTxn('${t.transaction_id}')">
        <td class="mono" style="font-size:0.72rem;color:var(--text-dim);">${t.timestamp || '2026-04-30 23:45'}</td>
        <td class="mono" style="color:var(--color-primary);">${t.transaction_id}</td>
        <td style="font-size:0.72rem;">XGBoost</td>
        <td class="mono font-bold">${score.toFixed(1)}%</td>
        <td style="font-size:0.72rem;">${t.merchant_category}</td>
        <td><span class="rec-badge ${rec.toLowerCase()}">${rec === 'REVIEW' ? 'VERIFY' : rec}</span></td>
        <td style="font-size:0.74rem;color:var(--text-muted);">${reason}</td>
      </tr>
    `;
  }).join('');
}

function renderEvaluationView(data) {
  if (!data) return;
}

/* --- TRANSACTION DETAIL INSPECTION DRAWER --- */
function inspectTxn(txnId) {
  const txn = rawTxnsData.find(t => t.transaction_id === txnId);
  if (!txn) return;

  const txnIdEl = document.getElementById('modal-txn-id');
  if (txnIdEl) txnIdEl.textContent = txn.transaction_id;

  const titleEl = document.getElementById('modal-amount-title');
  if (titleEl) titleEl.textContent = `\u20B9${txn.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

  const rec = txn.recommendation || 'ALLOW';
  const recBadge = document.getElementById('modal-rec-badge');
  if (recBadge) {
    recBadge.className = `rec-badge ${rec.toLowerCase()}`;
    recBadge.textContent = rec === 'REVIEW' ? 'VERIFY REQUIRED' : rec;
  }

  const scoreEl = document.getElementById('modal-score');
  if (scoreEl) scoreEl.textContent = `${txn.risk_score.toFixed(1)}%`;

  const threshEl = document.getElementById('modal-thresh');
  if (threshEl) threshEl.textContent = `${(txn.effective_threshold * 100).toFixed(1)}%`;

  const decEl = document.getElementById('modal-dec');
  if (decEl) decEl.textContent = rec === 'REVIEW' ? 'VERIFY' : rec;

  renderShapHorizontalBars('modal-shap-bars', txn.top_reasons || []);

  const baseProbEl = document.getElementById('modal-base-prob');
  if (baseProbEl) baseProbEl.textContent = `${(txn.base_fraud_rate || 3.0).toFixed(1)}%`;

  const outputProbEl = document.getElementById('modal-output-prob');
  if (outputProbEl) outputProbEl.textContent = `${txn.risk_score.toFixed(1)}%`;

  const cfText = document.getElementById('modal-cf-text');
  if (cfText) {
    if (txn.counterfactual && (txn.counterfactual.block_to_verify || txn.counterfactual.safe_amount_limit)) {
      const b2v = txn.counterfactual.block_to_verify || (txn.counterfactual.safe_amount_limit * 2.2);
      const v2a = txn.counterfactual.verify_to_allow || txn.counterfactual.safe_amount_limit;
      cfText.innerHTML = `At \u20B9${b2v.toFixed(0)} \u2192 BLOCK \u2192 VERIFY<br/>At \u20B9${v2a.toFixed(0)} \u2192 VERIFY \u2192 ALLOW`;
    } else {
      cfText.textContent = `Transaction amount sit comfortably within decision boundaries.`;
    }
  }

  const overlay = document.getElementById('modal-overlay');
  if (overlay) overlay.classList.remove('hidden');
}

function closeModal() {
  const overlay = document.getElementById('modal-overlay');
  if (overlay) overlay.classList.add('hidden');
}

function handleGlobalSearch(query) {
  if (!query) {
    filteredTxns = [...rawTxnsData];
  } else {
    const q = query.toLowerCase();
    filteredTxns = rawTxnsData.filter(t => 
      t.transaction_id.toLowerCase().includes(q) || 
      (t.merchant_id && t.merchant_id.toLowerCase().includes(q)) ||
      t.merchant_category.toLowerCase().includes(q)
    );
  }
  renderMonitorTable();
}

// Global scope attachments for inline HTML onclick handlers
window.switchNav = switchNav;
window.filterMonitor = filterMonitor;
window.updateSim = updateSim;
window.updateOptimizerCurve = updateOptimizerCurve;
window.inspectTxn = inspectTxn;
window.closeModal = closeModal;
window.handleGlobalSearch = handleGlobalSearch;
