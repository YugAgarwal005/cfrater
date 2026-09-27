/**
 * CF Rating Predictor — Main Application Script
 * Changes: live auto-tag detection, single constraints textarea,
 *          removed position selector, smarter rule-based demo mode
 */

/* ================================================================
   CONFIG
   ================================================================ */
const API_BASE = window.location.origin;

const ALL_TAGS = [
  "implementation","math","greedy","dp","data structures",
  "brute force","constructive algorithms","graphs","sortings",
  "binary search","dfs and similar","trees","strings","number theory",
  "combinatorics","geometry","bitmasks","two pointers","dsu",
  "shortest paths","probabilities","divide and conquer","hashing",
  "games","flows","interactive","matrices","fft","graph matchings",
  "ternary search","meet-in-the-middle","2-sat","string suffix structures",
];

const RATING_TIERS = [
  { max: 1199, label: "Newbie",            class: "tier-pupil"     },
  { max: 1399, label: "Pupil",             class: "tier-pupil"     },
  { max: 1599, label: "Specialist",        class: "tier-specialist"},
  { max: 1899, label: "Expert",            class: "tier-expert"    },
  { max: 2099, label: "Candidate Master",  class: "tier-cm"        },
  { max: 2299, label: "Master",            class: "tier-master"    },
  { max: 2399, label: "Int. Master",       class: "tier-im"        },
  { max: 2599, label: "Grandmaster",       class: "tier-gm"        },
  { max: 2999, label: "Int. GM",           class: "tier-gm"        },
  { max: Infinity, label: "Legendary GM", class: "tier-legend"    },
];

/* ================================================================
   TAG DETECTION RULES (used for live suggestions)
   ================================================================ */
const TAG_RULES = [
  { tag: "dp",                    regex: /\b(dynamic.programm|memoiz|dp\b|knapsack|subproblem|optimal substructure)\b/i },
  { tag: "graphs",                regex: /\b(graph|node|edge|vertex|vertices|adjacen|connected component|bipartite)\b/i },
  { tag: "trees",                 regex: /\b(rooted.tree|binary tree|parent.*child|subtree|forest|lca|ancestor|leaf node)\b/i },
  { tag: "binary search",         regex: /\b(binary.search|bisect|lower.bound|upper.bound|monoton)\b/i },
  { tag: "greedy",                regex: /\b(greedy|always.pick|maximum.*first|minimum.*first|exchange argument)\b/i },
  { tag: "math",                  regex: /\b(prime|modulo|factorial|gcd|lcm|fibonacci|euler|totient|combinatorics|pigeonhole)\b/i },
  { tag: "number theory",         regex: /\b(prime|divisor|gcd|lcm|euler|totient|sieve|modular inverse|coprime)\b/i },
  { tag: "sortings",              regex: /\b(sort|sorted|ascending|descending|order.*element|arrange)\b/i },
  { tag: "strings",               regex: /\b(string|substring|palindrome|character|prefix|suffix|lexicograph)\b/i },
  { tag: "geometry",              regex: /\b(point|line segment|circle|polygon|convex.hull|intersection|collinear|triangle|distance)\b/i },
  { tag: "dsu",                   regex: /\b(union.find|disjoint.set|connected.component|merge.set)\b/i },
  { tag: "shortest paths",        regex: /\b(shortest.path|dijkstra|bellman.ford|floyd|distance.*graph)\b/i },
  { tag: "dfs and similar",       regex: /\b(depth.first|dfs|backtrack|recursive.*graph|flood.fill)\b/i },
  { tag: "bitmasks",              regex: /\b(bitmask|subset|bitwise|AND.*OR|XOR.*bit)\b/i },
  { tag: "two pointers",          regex: /\b(two.pointer|sliding.window|left.*right.*pointer)\b/i },
  { tag: "hashing",               regex: /\b(hash.map|hash.set|unordered|rolling.hash|polynomial.hash)\b/i },
  { tag: "data structures",       regex: /\b(segment.tree|fenwick|BIT|heap|priority.queue|trie|stack|deque|sparse.table)\b/i },
  { tag: "flows",                 regex: /\b(max.flow|min.cut|network.flow|matching.*bipartite)\b/i },
  { tag: "games",                 regex: /\b(game.theory|nim|grundy|sprague|mex|optimal.*play|winning.*position)\b/i },
  { tag: "constructive algorithms", regex: /\b(construct|build.*answer|explicitly construct|exist.*solution)\b/i },
  { tag: "probabilities",         regex: /\b(probability|expected.value|random|chance|likelihood)\b/i },
  { tag: "interactive",           regex: /\b(interactive|ask.*query|queries.*answer|hidden.*number)\b/i },
  { tag: "divide and conquer",    regex: /\b(divide.and.conquer|merge.sort|split.*half|recursive.*halve)\b/i },
  { tag: "fft",                   regex: /\b(fast.fourier|FFT|NTT|polynomial.multiplication|convolution)\b/i },
];

/* ================================================================
   PARTICLE CANVAS
   ================================================================ */
(function initParticles() {
  const canvas = document.getElementById("particles-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W, H, particles;

  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }

  function createParticles(n) {
    return Array.from({ length: n }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.5 + 0.5, a: Math.random() * 0.5 + 0.1,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(129,140,248,${p.a})`; ctx.fill();
    }
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(129,140,248,${0.08 * (1 - dist / 100)})`;
          ctx.lineWidth = 0.5; ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", () => { resize(); particles = createParticles(60); });
  particles = createParticles(60);
  draw();
})();

/* ================================================================
   COUNTER ANIMATION
   ================================================================ */
function animateCounter(el, target, duration = 1500) {
  const start = performance.now();
  const update = (time) => {
    const progress = Math.min((time - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(eased * target).toLocaleString();
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(entry.target, parseInt(entry.target.dataset.target, 10));
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });
document.querySelectorAll("[data-target]").forEach(el => observer.observe(el));

/* ================================================================
   TAG SYSTEM
   ================================================================ */
const tagsChipsEl     = document.getElementById("tag-chips");
const tagsSelectedEl  = document.getElementById("tags-selected");
const tagsInputEl     = document.getElementById("tags-input");
const autoTagRow      = document.getElementById("auto-tag-row");
const autoTagChipsEl  = document.getElementById("auto-tag-chips");
const autoTagAddAll   = document.getElementById("auto-tag-add-all");

let selectedTags = new Set();

function renderTagChips() {
  tagsChipsEl.innerHTML = "";
  ALL_TAGS.slice(0, 20).forEach(tag => {
    const chip = document.createElement("span");
    chip.className = "tag-chip" + (selectedTags.has(tag) ? " active" : "");
    chip.textContent = tag;
    chip.addEventListener("click", () => toggleTag(tag));
    tagsChipsEl.appendChild(chip);
  });
}

function toggleTag(tag, source = "click") {
  if (selectedTags.has(tag)) {
    selectedTags.delete(tag);
  } else {
    selectedTags.add(tag);
    // Mark auto-chip as added
    if (source === "auto") {
      document.querySelectorAll(".auto-tag-chip").forEach(c => {
        if (c.dataset.tag === tag) c.classList.add("added");
      });
    }
  }
  renderSelectedTags();
  renderTagChips();
}

function addTag(tag) {
  selectedTags.add(tag.toLowerCase().trim());
  renderSelectedTags();
  renderTagChips();
}

function renderSelectedTags() {
  tagsSelectedEl.innerHTML = "";
  selectedTags.forEach(tag => {
    const span = document.createElement("span");
    span.className = "tag-selected";
    span.innerHTML = `${tag} <span class="tag-x">×</span>`;
    span.addEventListener("click", () => toggleTag(tag));
    tagsSelectedEl.appendChild(span);
  });
}

// Type to add tags
tagsInputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === ",") {
    e.preventDefault();
    const val = tagsInputEl.value.trim().replace(/,$/, "");
    if (val) { addTag(val); tagsInputEl.value = ""; }
  }
});

// "Add all" button
autoTagAddAll.addEventListener("click", () => {
  document.querySelectorAll(".auto-tag-chip:not(.added)").forEach(c => {
    addTag(c.dataset.tag);
    c.classList.add("added");
  });
});

renderTagChips();

/* ================================================================
   LIVE AUTO-TAG DETECTION
   Debounced — fires 600ms after user stops typing in statement
   ================================================================ */
let autoTagDebounce = null;

function detectTagsFromText(text) {
  if (!text || text.length < 30) return [];
  const detected = [];
  for (const { tag, regex } of TAG_RULES) {
    if (regex.test(text) && !selectedTags.has(tag)) {
      detected.push(tag);
    }
  }
  return detected;
}

function showAutoTagSuggestions(tags) {
  if (!tags.length) {
    autoTagRow.hidden = true;
    return;
  }
  autoTagChipsEl.innerHTML = "";
  tags.forEach(tag => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "auto-tag-chip" + (selectedTags.has(tag) ? " added" : "");
    chip.dataset.tag = tag;
    chip.innerHTML = `+ ${tag}`;
    chip.addEventListener("click", () => toggleTag(tag, "auto"));
    autoTagChipsEl.appendChild(chip);
  });
  autoTagRow.hidden = false;
}

function runAutoTagDetection() {
  const text = [
    document.getElementById("problem-statement").value,
    document.getElementById("input-spec").value,
    document.getElementById("constraints-text").value,
  ].join(" ");
  const suggestions = detectTagsFromText(text);
  showAutoTagSuggestions(suggestions);
}

// Attach live detection to all relevant fields
["problem-statement", "input-spec", "constraints-text"].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", () => {
    clearTimeout(autoTagDebounce);
    autoTagDebounce = setTimeout(runAutoTagDetection, 600);
  });
});

/* ================================================================
   CHARACTER COUNTER
   ================================================================ */
const stmtTextarea = document.getElementById("problem-statement");
const stmtCounter  = document.getElementById("statement-counter");

stmtTextarea.addEventListener("input", () => {
  const len = stmtTextarea.value.length;
  stmtCounter.textContent = len.toLocaleString() + " chars";
  stmtCounter.style.color = len > 5000 ? "var(--amber)" : "var(--text-muted)";
});

/* ================================================================
   PASTE → AUTO-DETECT TAGS
   ================================================================ */
stmtTextarea.addEventListener("paste", () => {
  // Small delay so value is updated after paste
  setTimeout(runAutoTagDetection, 100);
});

/* ================================================================
   FORM SUBMISSION
   ================================================================ */
const form       = document.getElementById("predictor-form");
const predictBtn = document.getElementById("predict-btn");
const btnText    = predictBtn.querySelector(".btn-text");
const btnLoader  = predictBtn.querySelector(".btn-loader");
const btnIcon    = predictBtn.querySelector(".btn-icon");
const resultPanel = document.getElementById("result-panel");
const errorPanel  = document.getElementById("error-panel");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await runPrediction();
});

async function runPrediction() {
  const statement = document.getElementById("problem-statement").value.trim();
  if (!statement) { shakeInput("problem-statement"); return; }

  setLoading(true);
  hideAll();

  // Parse time/memory from constraints textarea
  const constraintsRaw = document.getElementById("constraints-text").value.trim();
  const timeLimitMatch  = constraintsRaw.match(/time\s*limit[:\s]*(\d+(?:\.\d+)?)\s*(second|sec)/i);
  const memLimitMatch   = constraintsRaw.match(/memory\s*limit[:\s]*(\d+)\s*(mb|megabyte|mib)/i);

  // If user hasn't manually selected tags, auto-detect from all text fields
  let activeTags = Array.from(selectedTags);
  if (activeTags.length === 0) {
    const fullTextForTags = [
      statement,
      document.getElementById("input-spec").value,
      constraintsRaw
    ].join(" ");
    const autoDetected = detectTagsFromText(fullTextForTags);
    if (autoDetected.length > 0) {
      activeTags = autoDetected;
      autoDetected.forEach(t => selectedTags.add(t));
      renderSelectedTags();
      renderTagChips();
      autoTagRow.hidden = true;
    }
  }

  const payload = {
    name:          document.getElementById("problem-name").value.trim(),
    statement,
    input_spec:    document.getElementById("input-spec").value.trim(),
    output_spec:   document.getElementById("output-spec").value.trim(),
    constraints:   constraintsRaw,
    tags:          activeTags,
    time_limit:    timeLimitMatch  ? `${timeLimitMatch[1]} seconds`    : "2 seconds",
    memory_limit:  memLimitMatch   ? `${memLimitMatch[1]} megabytes`   : "256 megabytes",
    solution_code: document.getElementById("solution-code").value.trim() || null,
    _nonce:        Date.now(),   // prevent any browser response caching
  };

  console.log("[predict] sending request:", {
    statement: payload.statement.slice(0, 80),
    tags: payload.tags,
    nonce: payload._nonce,
  });

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache, no-store",
        "Pragma": "no-cache",
      },
      cache: "no-store",
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }

    const data = await res.json();
    console.log("[predict] raw API response:", data);

    if (data.error) throw new Error(data.error);
    renderResult(data);

  } catch (err) {
    console.error("[predict] API error:", err.message);
    // Only use smart demo when the server itself is unreachable (network error)
    // For any other error, show the error panel
    if (err.message.startsWith("Failed to fetch") || err.message.includes("NetworkError")) {
      console.warn("[predict] server unreachable — using smart local estimate");
      renderResult(smartDemoEstimate(payload));
    } else {
      showError(err.message);
    }
  } finally {
    setLoading(false);
  }
}

/* ================================================================
   SMART RULE-BASED DEMO ESTIMATE
   Used when no trained model exists yet
   ================================================================ */
function smartDemoEstimate(payload) {
  const tags = payload.tags || [];
  const constraints = (payload.constraints || "") + " " + (payload.input_spec || "");
  const text = (payload.statement || "").toLowerCase();

  // Tag → base rating lookup
  const TAG_BASE = {
    "implementation": 1200, "math": 1400, "greedy": 1500, "dp": 1700,
    "data structures": 1800, "brute force": 1100, "constructive algorithms": 1600,
    "graphs": 1700, "sortings": 1200, "binary search": 1500, "dfs and similar": 1500,
    "trees": 1800, "strings": 1600, "number theory": 1700, "combinatorics": 1800,
    "geometry": 2000, "bitmasks": 1700, "two pointers": 1500, "dsu": 1800,
    "shortest paths": 1800, "probabilities": 1900, "divide and conquer": 1900,
    "hashing": 1700, "games": 1900, "flows": 2200, "interactive": 1800,
    "matrices": 2000, "fft": 2500, "graph matchings": 2300,
    "ternary search": 1800, "meet-in-the-middle": 2100, "2-sat": 2400,
    "string suffix structures": 2500,
  };

  let baseRating = 1400;
  if (tags.length > 0) {
    const tagRatings = tags.map(t => TAG_BASE[t] || 1500);
    baseRating = Math.round(tagRatings.reduce((a, b) => a + b, 0) / tagRatings.length);
  }

  // Constraint-based adjustment
  const nMatch = constraints.match(/n\s*[≤<=]+\s*([\d\s\*e\^]+)/i);
  if (nMatch) {
    const raw = nMatch[1].trim().replace(/\s/g, "").replace(/10\^(\d+)/g, (_, e) => `1e${e}`);
    try {
      const n = parseFloat(eval(raw));
      if (n <= 20)     baseRating = Math.max(baseRating, 1800); // Likely exponential/brute
      if (n <= 1000)   baseRating = Math.max(baseRating - 100, 900);
      if (n >= 1e5)    baseRating = Math.min(baseRating + 100, 2800);
      if (n >= 1e9)    baseRating = Math.max(baseRating + 200, 1800); // Likely math/binary search
    } catch {}
  }

  // Multiple test cases → slightly harder
  if (/\bt\s*test\s*case|\bq\s*quer/i.test(text)) baseRating += 50;

  // Multiple tags → harder (more concepts needed)
  if (tags.length >= 3) baseRating += 100;
  if (tags.length >= 5) baseRating += 150;

  // Snap to nearest 100
  baseRating = Math.round(baseRating / 100) * 100;
  baseRating = Math.max(800, Math.min(baseRating, 3000));

  const spread = tags.length ? 100 : 200;
  return {
    predicted_rating: baseRating,
    predicted_continuous: baseRating,
    estimated_range: [Math.max(800, baseRating - spread), Math.min(3000, baseRating + spread)],
    confidence: tags.length >= 2 ? 38 : 22,
    confidence_label: "Low",
    ensemble_std: spread,
    model_predictions: {
      "rule_based_estimator": baseRating,
    },
    disclaimer: "⚠️ This is a rule-based estimate (no trained model yet). Collect data and train for accurate ML predictions.",
    _demo_mode: true,
    _smart_demo: true,
  };
}

/* ================================================================
   RESULT RENDERING
   ================================================================ */
function renderResult(data) {
  const rating     = data.predicted_rating || 1500;
  const range      = data.estimated_range  || [rating - 100, rating + 100];
  const confidence = data.confidence       || 50;
  const models     = data.model_predictions || {};

  // Rating number + color
  const ratingEl = document.getElementById("result-rating");
  ratingEl.textContent = rating;
  const tier = getTier(rating);
  ratingEl.className = "rating-number " + tier.class;

  // Range bar
  const MIN_RATING = 800, MAX_RATING = 3000;
  const pct = v => ((v - MIN_RATING) / (MAX_RATING - MIN_RATING)) * 100;
  document.getElementById("range-fill").style.cssText =
    `left:${pct(range[0])}%; width:${pct(range[1]) - pct(range[0])}%`;
  document.getElementById("range-dot").style.left = pct(rating) + "%";
  document.getElementById("range-text").textContent =
    `Estimated Range: ${range[0]} to ${range[1]}`;

  // Confidence
  document.getElementById("result-confidence").textContent = confidence + "%";
  document.getElementById("confidence-fill").style.width = confidence + "%";

  // Tier
  const tierEl = document.getElementById("result-tier");
  tierEl.textContent = tier.label;
  tierEl.className = "metric-value " + tier.class;

  // Range width
  document.getElementById("result-range-width").textContent = "±" + (range[1] - range[0]) / 2;

  // Model breakdown
  renderModelList(models, data._smart_demo);

  // Disclaimer
  document.getElementById("result-disclaimer").textContent = data.disclaimer || "";

  // Demo badge
  const demoBadge = document.getElementById("demo-badge");
  demoBadge.hidden = !data._demo_mode;
  if (data._smart_demo) {
    demoBadge.textContent = "⚡ Smart estimate mode — run data collection + training for real ML predictions";
  } else if (data._demo_mode) {
    demoBadge.textContent = "🧪 Demo mode — train the model with real data for accurate predictions";
  }

  resultPanel.hidden = false;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderModelList(models, isSmartDemo = false) {
  const listEl = document.getElementById("model-list");
  listEl.innerHTML = "";

  if (isSmartDemo) {
    listEl.innerHTML = `
      <div class="model-row" style="grid-template-columns:1fr auto">
        <span class="model-name">Rule-based estimator (tags + constraints)</span>
        <span class="model-pred">${models["rule_based_estimator"] || "—"}</span>
      </div>
      <div style="font-size:0.8rem;color:var(--text-muted);padding:0.5rem 0.8rem;border:1px dashed var(--border);border-radius:var(--radius-sm);margin-top:0.25rem">
        Waiting for trained models — run <code style="color:var(--indigo)">setup.bat train</code>
      </div>`;
    return;
  }

  const modelLabels = {
    "xgb_regressor":    ["XGBoost",        "reg"],
    "lgb_regressor":    ["LightGBM",       "reg"],
    "rf_regressor":     ["Random Forest",  "reg"],
    "ridge_regressor":  ["Ridge Regression","reg"],
    "xgb_classifier":  ["XGBoost",        "cls"],
    "rf_classifier":   ["Random Forest",  "cls"],
  };

  Object.entries(models).forEach(([key, val]) => {
    if (val === null || val === undefined) return;
    const [label, type] = modelLabels[key] || [key, "reg"];
    const row = document.createElement("div");
    row.className = "model-row";
    row.innerHTML = `
      <span class="model-name">${label}</span>
      <span class="model-type-badge ${type === "reg" ? "badge-reg" : "badge-cls"}">
        ${type === "reg" ? "Regressor" : "Classifier"}
      </span>
      <span class="model-pred">${val}</span>`;
    listEl.appendChild(row);
  });

  if (!listEl.children.length) {
    listEl.innerHTML = '<div class="model-row" style="justify-content:center;color:var(--text-muted)">No model breakdown available</div>';
  }
}

function getTier(rating) {
  return RATING_TIERS.find(t => rating <= t.max) || RATING_TIERS[RATING_TIERS.length - 1];
}

/* ================================================================
   UI HELPERS
   ================================================================ */
function setLoading(loading) {
  predictBtn.disabled = loading;
  btnText.textContent = loading ? "Analyzing…" : "Predict Rating";
  btnLoader.hidden = !loading;
  btnIcon.style.display = loading ? "none" : "";
}

function showError(msg) {
  errorPanel.hidden = false;
  document.getElementById("error-msg").textContent = msg;
}

function hideAll() {
  resultPanel.hidden = true;
  errorPanel.hidden  = true;
}

function shakeInput(id) {
  const el = document.getElementById(id);
  el.style.animation = "none";
  el.offsetHeight;
  el.style.animation = "shake 0.4s ease";
  el.focus();
}

const shakeStyle = document.createElement("style");
shakeStyle.textContent = `
  @keyframes shake {
    0%,100%{transform:translateX(0)} 20%{transform:translateX(-8px)}
    40%{transform:translateX(8px)} 60%{transform:translateX(-5px)} 80%{transform:translateX(5px)}
  }`;
document.head.appendChild(shakeStyle);

/* ================================================================
   COPY RESULT
   ================================================================ */
document.getElementById("copy-result-btn").addEventListener("click", () => {
  const text = `CF Rating Predictor Result:\n` +
    `Predicted Rating: ${document.getElementById("result-rating").textContent}\n` +
    `${document.getElementById("range-text").textContent}\n` +
    `Confidence: ${document.getElementById("result-confidence").textContent}\n` +
    `\n⚠️ Estimated only — not an official Codeforces rating.`;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("copy-result-btn");
    btn.textContent = "✓ Copied!";
    setTimeout(() => {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" stroke-width="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" stroke="currentColor" stroke-width="2"/></svg> Copy Result`;
    }, 2000);
  });
});

/* ================================================================
   RESET
   ================================================================ */
document.getElementById("reset-btn").addEventListener("click", () => {
  hideAll();
  form.reset();
  selectedTags.clear();
  renderSelectedTags();
  renderTagChips();
  autoTagRow.hidden = true;
  stmtCounter.textContent = "0 chars";
  window.scrollTo({ top: document.getElementById("predictor").offsetTop - 80, behavior: "smooth" });
});

/* ================================================================
   SERVER HEALTH CHECK
   ================================================================ */
(async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      console.info("%c✓ ML models loaded — real predictions active", "color:#34d399;font-weight:bold");
    } else {
      console.info("%c⚡ No trained models — smart estimate mode active", "color:#fbbf24;font-weight:bold");
    }
  } catch {
    console.info("Server not reachable — smart local estimates will be used");
  }
})();

/* ================================================================
   SMOOTH NAV SCROLL
   ================================================================ */
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener("click", e => {
    const target = document.querySelector(link.getAttribute("href"));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: "smooth" }); }
  });
});

console.log("%cCF Rating Predictor loaded ✓", "color:#818cf8;font-weight:bold;font-size:14px;");
