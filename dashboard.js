'use strict';

const C = {
  bg:      '#0b0f18',
  surface: '#111827',
  s2:      '#1a2236',
  border:  'rgba(255,255,255,0.07)',
  text:    '#e2e8f0',
  muted:   '#6b7a99',
  accent:  '#06b6d4',
  accent2: '#8b5cf6',
  green:   '#10b981',
  red:     '#f43f5e',
  yellow:  '#f59e0b',
};

Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = "'Inter','Segoe UI',system-ui,sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = C.surface;
Chart.defaults.plugins.tooltip.titleColor = C.text;
Chart.defaults.plugins.tooltip.bodyColor = C.muted;
Chart.defaults.plugins.tooltip.borderColor = C.border;
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;

function sparkline(id, data, color) {
  const ctx = document.getElementById(id).getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map((_, i) => i),
      datasets: [{ data, borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true,
        backgroundColor: (ctx) => {
          const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 32);
          g.addColorStop(0, color + '40');
          g.addColorStop(1, color + '00');
          return g;
        }
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
      animation: false,
    }
  });
}

sparkline('spark1', [32000,35000,38000,36000,40000,43000,45000,47000,48291], C.accent);
sparkline('spark2', [155,160,158,162,170,175,180,182,184], C.yellow);
sparkline('spark3', [0.82,0.825,0.830,0.832,0.835,0.838,0.840,0.845,0.847], C.green);
sparkline('spark4', [76,77,77.5,78,78,78.5,79,79.2,79.4], C.accent2);
sparkline('spark5', [2.8,2.7,2.6,2.5,2.4,2.3,2.2,2.15,2.1], C.green);
sparkline('spark6', [72,70,69,68,67,66,65,63,61.8], C.red);

/* Query Volume & Latency */
(function() {
  const labels = [];
  const queries = [], p50 = [], p95 = [], p99 = [];
  for (let i = 23; i >= 0; i--) {
    const h = new Date(); h.setHours(h.getHours() - i, 0, 0, 0);
    labels.push(h.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }));
    const base = 1800 + Math.sin(i * 0.4) * 400 + Math.random() * 200;
    queries.push(Math.round(base));
    p50.push(+(160 + Math.random() * 30).toFixed(0));
    p95.push(+(280 + Math.random() * 60).toFixed(0));
    p99.push(+(420 + Math.random() * 80).toFixed(0));
  }

  const ctx = document.getElementById('queryChart').getContext('2d');
  const gArea = ctx.createLinearGradient(0, 0, 0, 240);
  gArea.addColorStop(0, C.accent + '40');
  gArea.addColorStop(1, C.accent + '00');

  new Chart(ctx, {
    data: {
      labels,
      datasets: [
        { type: 'bar', label: 'Queries/hr', data: queries, backgroundColor: C.accent + '33',
          borderColor: C.accent + '66', borderWidth: 1, borderRadius: 3, yAxisID: 'y', order: 2 },
        { type: 'line', label: 'p50 (ms)', data: p50, borderColor: C.green, borderWidth: 2,
          tension: 0.4, pointRadius: 0, yAxisID: 'y2', order: 1 },
        { type: 'line', label: 'p95 (ms)', data: p95, borderColor: C.yellow, borderWidth: 1.5,
          tension: 0.4, pointRadius: 0, yAxisID: 'y2', borderDash: [5,3], order: 1 },
        { type: 'line', label: 'p99 (ms)', data: p99, borderColor: C.red, borderWidth: 1.5,
          tension: 0.4, pointRadius: 0, yAxisID: 'y2', borderDash: [3,3], order: 1 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { boxWidth: 12, font: { size: 11 }, color: C.muted } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 }, maxTicksLimit: 12 } },
        y: {
          type: 'linear', position: 'left', grid: { color: C.border },
          ticks: { font: { size: 11 } },
          title: { display: true, text: 'Queries/hr', font: { size: 11 }, color: C.muted }
        },
        y2: {
          type: 'linear', position: 'right', grid: { drawOnChartArea: false },
          ticks: { font: { size: 11 }, callback: v => v + 'ms' },
          title: { display: true, text: 'Latency (ms)', font: { size: 11 }, color: C.muted }
        }
      }
    }
  });
})();

/* Relevance Score Distribution */
(function() {
  const ctx = document.getElementById('scoreChart').getContext('2d');
  const buckets = ['0.0–0.2','0.2–0.4','0.4–0.6','0.6–0.7','0.7–0.8','0.8–0.9','0.9–1.0'];
  const counts  = [120, 340, 810, 1250, 3800, 12400, 9200];
  const colors  = [C.red, C.red, C.yellow, C.yellow, C.accent, C.green, C.green];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: buckets,
      datasets: [{ label: 'Queries', data: counts, backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors, borderWidth: 1, borderRadius: 5, barPercentage: 0.7 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: C.border }, ticks: { font: { size: 11 } },
          title: { display: true, text: 'Query count', font: { size: 11 }, color: C.muted } }
      }
    }
  });
})();

/* Top Retrieved Documents */
(function() {
  const ctx = document.getElementById('docsChart').getContext('2d');
  const docs = ['product_manual_v3.pdf','faq_enterprise.md','api_reference.json',
                 'pricing_2026.xlsx','release_notes_4.2.txt'];
  const hits = [4820, 3910, 3240, 2780, 2150];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: docs.map(d => d.length > 22 ? d.slice(0,20)+'…' : d),
      datasets: [{ label: 'Retrieval count', data: hits,
        backgroundColor: [C.accent+'bb', C.accent2+'bb', C.green+'bb', C.yellow+'bb', C.accent+'66'],
        borderRadius: 5, barPercentage: 0.55 }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: C.border }, ticks: { font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });
})();

/* Pipeline Stage Latency */
(function() {
  const ctx = document.getElementById('latencyBreakdown').getContext('2d');
  const stages = ['Query Embed','ANN Search','Re-rank','Chunk Fetch','LLM Prompt','Generation'];
  const avgMs  = [18, 62, 45, 22, 14, 295];
  const colors = [C.green, C.yellow, C.accent, C.green, C.accent2, C.accent];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: stages,
      datasets: [{ label: 'Avg latency (ms)', data: avgMs,
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors, borderWidth: 1,
        borderRadius: 5, barPercentage: 0.6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { grid: { color: C.border }, ticks: { font: { size: 11 }, callback: v => v + 'ms' } }
      }
    }
  });
})();

/* Error Classification Donut */
(function() {
  const ctx = document.getElementById('errorChart').getContext('2d');
  const labels = ['Low Relevance','Timeout','Embedding Fail','No Results','Token Overflow'];
  const data   = [38, 22, 18, 14, 8];
  const colors = [C.yellow, C.red, C.accent2, C.muted, C.accent];

  new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors.map(c=>c+'cc'), borderColor: '#111827', borderWidth: 3, hoverOffset: 6 }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '68%',
      plugins: { legend: { display: false }, tooltip: { callbacks: {
        label: ctx => ` ${ctx.label}: ${ctx.raw}%`
      }}}
    }
  });

  const legend = document.getElementById('errorLegend');
  labels.forEach((l, i) => {
    legend.innerHTML += `<div class="legend-row">
      <div class="legend-dot" style="background:${colors[i]}"></div>
      <span class="legend-label">${l}</span>
      <span class="legend-pct" style="color:${colors[i]}">${data[i]}%</span>
    </div>`;
  });
})();

/* Query Log Table */
const queries = [
  { ts:'04/22 14:32:18', q:'How do I configure SSO with SAML?', chunks:5, score:0.91, lat:142, tokens:1840, status:'ok' },
  { ts:'04/22 14:32:05', q:'What are the rate limits for the API?', chunks:3, score:0.87, lat:118, tokens:1220, status:'ok' },
  { ts:'04/22 14:31:52', q:'Explain the pricing tiers for enterprise', chunks:7, score:0.79, lat:388, tokens:2480, status:'slow' },
  { ts:'04/22 14:31:41', q:'Quarterly release schedule 2026', chunks:2, score:0.54, lat:205, tokens:980, status:'ok' },
  { ts:'04/22 14:31:30', q:'Webhook payload schema for billing events', chunks:4, score:0.88, lat:156, tokens:1640, status:'ok' },
  { ts:'04/22 14:31:12', q:'MFA setup for mobile app', chunks:6, score:0.93, lat:133, tokens:2100, status:'ok' },
  { ts:'04/22 14:30:58', q:'Custom embedding model integration guide', chunks:0, score:0.0,  lat:512, tokens:0,    status:'fail' },
  { ts:'04/22 14:30:44', q:'How to export data to S3?', chunks:4, score:0.82, lat:174, tokens:1510, status:'ok' },
];

const tbody = document.getElementById('queryTableBody');
queries.forEach(r => {
  const pillClass = r.status === 'ok' ? 'pill-ok' : r.status === 'slow' ? 'pill-slow' : 'pill-fail';
  const pillText  = r.status === 'ok' ? '✓ OK' : r.status === 'slow' ? '⚠ Slow' : '✕ Failed';
  const scoreColor = r.score > 0.8 ? C.green : r.score > 0.6 ? C.yellow : C.red;
  const latColor   = r.lat < 200 ? C.green : r.lat < 400 ? C.yellow : C.red;
  tbody.innerHTML += `<tr>
    <td>${r.ts}</td>
    <td class="text" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.q}</td>
    <td style="text-align:center">${r.chunks}</td>
    <td>
      <div class="score-bar">
        <div class="score-track"><div class="score-fill" style="width:${r.score*100}%;background:${scoreColor}"></div></div>
        <span style="color:${scoreColor}">${r.score.toFixed(2)}</span>
      </div>
    </td>
    <td style="color:${latColor}">${r.lat}ms</td>
    <td>${r.tokens > 0 ? r.tokens.toLocaleString() : '—'}</td>
    <td><span class="status-pill ${pillClass}">${pillText}</span></td>
    <td><button class="trace-btn">Trace →</button></td>
  </tr>`;
});

/* Tab switcher */
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.closest('.chart-controls') || btn.closest('.table-actions');
    if (!group) return;
    group.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});
