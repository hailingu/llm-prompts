// Initialize Reveal.js and Chart.js
const deck = new Reveal({
  hash: true,
  plugins: []
});
deck.initialize({
  dependencies: [
    { src: 'https://unpkg.com/reveal.js@4/plugin/notes/notes.js', async: true },
    { src: 'https://unpkg.com/reveal.js@4/plugin/overview/overview.js', async: true },
    { src: 'https://unpkg.com/reveal.js@4/plugin/markdown/markdown.js', async: true },
    { src: 'https://unpkg.com/reveal.js@4/plugin/zoom/zoom.js', async: true },
    { src: 'https://unpkg.com/reveal.js@4/plugin/accessible/accessible.js', async: true }
  ]
});

// Example Chart.js usage (bar chart) — accessibility-minded
function renderLatencyChart() {
  const ctx = document.getElementById('latencyChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Baseline', 'Optimized'],
      datasets: [{
        label: 'p95 latency (ms)',
        data: [120, 84],
        backgroundColor: ['var(--color-primary)', 'var(--color-secondary)'],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: true }
      },
      scales: {
        y: { beginAtZero: true }
      }
    }
  });
}

// DOM Ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { renderLatencyChart(); });
} else { renderLatencyChart(); }

// Small enhancement: mark data charts with ARIA roles and descriptions
const chartCanvas = document.getElementById('latencyChart');
if (chartCanvas) {
  chartCanvas.setAttribute('role', 'img');
  chartCanvas.setAttribute('aria-label', 'Bar chart showing p95 latency: Baseline 120 ms and Optimized 84 ms');
}
