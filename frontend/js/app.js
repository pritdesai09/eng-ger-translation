/* ══════════════════════════════════════════════════════════════════════
   EN ↔ DE Neural Translator — Frontend Logic
   ══════════════════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:8000";

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  model:     "transformer",   // 'transformer' | 'seq2seq'
  direction: "en_de",         // 'en_de' | 'de_en'
  loading:   false,
};

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const srcText      = $("srcText");
const outputText   = $("outputText");
const translateBtn = $("translateBtn");
const btnSpinner   = $("btnSpinner");
const charCount    = $("charCount");
const latencyBadge = $("latencyBadge");
const modelBadge   = $("modelBadge");
const tokenSection = $("tokenSection");
const tokenList    = $("tokenList");
const errorBanner  = $("errorBanner");
const errorMsg     = $("errorMsg");
const srcLang      = $("srcLang");
const tgtLang      = $("tgtLang");
const srcPanelLang = $("srcPanelLang");
const tgtPanelLang = $("tgtPanelLang");
const tokenToggle  = $("tokenToggle");
const copyBtn      = $("copyBtn");
const clearBtn     = $("clearBtn");
const swapBtn      = $("swapBtn");

// ── Language labels ───────────────────────────────────────────────────────────
const LANGS = {
  en: "English",
  de: "German",
};

function getLangs() {
  return state.direction === "en_de"
    ? { src: "en", tgt: "de" }
    : { src: "de", tgt: "en" };
}

function updateLangLabels() {
  const { src, tgt } = getLangs();
  srcLang.textContent      = LANGS[src];
  tgtLang.textContent      = LANGS[tgt];
  srcPanelLang.textContent = LANGS[src];
  tgtPanelLang.textContent = LANGS[tgt];
  srcText.placeholder = `Enter ${LANGS[src]} text...`;
}

// ── Model segmented control ───────────────────────────────────────────────────
document.querySelectorAll(".seg-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.model = btn.dataset.value;
    updateActiveCard();
  });
});

function updateActiveCard() {
  document.querySelectorAll(".info-card").forEach(c => c.classList.remove("active-model"));
  const card = $(`card-${state.model}`);
  if (card) card.classList.add("active-model");
}

// ── Direction swap ────────────────────────────────────────────────────────────
swapBtn.addEventListener("click", () => {
  state.direction = state.direction === "en_de" ? "de_en" : "en_de";
  updateLangLabels();
  clearOutput();
});

// ── Character counter ─────────────────────────────────────────────────────────
srcText.addEventListener("input", () => {
  const len = srcText.value.length;
  charCount.textContent = `${len} / 500`;
  charCount.style.color = len > 450 ? "#ff9090" : "var(--text-3)";
});

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  srcText.value = "";
  charCount.textContent = "0 / 500";
  clearOutput();
});

function clearOutput() {
  outputText.innerHTML = `<span class="output-placeholder">Translation will appear here...</span>`;
  latencyBadge.style.display = "none";
  modelBadge.style.display   = "none";
  tokenSection.style.display = "none";
  tokenList.innerHTML        = "";
}

// ── Copy ──────────────────────────────────────────────────────────────────────
copyBtn.addEventListener("click", () => {
  const text = outputText.querySelector(".output-text")?.textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    copyBtn.textContent = "✓ Copied!";
    setTimeout(() => { copyBtn.textContent = "⎘ Copy"; }, 1800);
  });
});

// ── Token toggle ──────────────────────────────────────────────────────────────
tokenToggle.addEventListener("click", () => {
  const visible = tokenList.style.display !== "none";
  tokenList.style.display   = visible ? "none" : "flex";
  tokenToggle.textContent   = visible ? "Show ▼" : "Hide ▲";
});

// ── Keyboard shortcut: Ctrl+Enter ─────────────────────────────────────────────
srcText.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") doTranslate();
});

// ── Translate ─────────────────────────────────────────────────────────────────
translateBtn.addEventListener("click", doTranslate);

async function doTranslate() {
  const text = srcText.value.trim();

  if (!text) {
    showError("Please enter some text to translate.");
    return;
  }

  hideError();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model_type: state.model,
        direction:  state.direction,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Translation failed.");
    }

    const data = await res.json();
    displayResult(data);

  } catch (err) {
    if (err.name === "TypeError") {
      showError("Cannot connect to the backend. Make sure the server is running on port 8000.");
    } else {
      showError(err.message);
    }
  } finally {
    setLoading(false);
  }
}

function displayResult(data) {
  // Output text
  outputText.innerHTML = `<span class="output-text">${escapeHTML(data.translation)}</span>`;

  // Badges
  latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;
  latencyBadge.style.display = "inline-block";

  const modelLabel = data.model === "transformer" ? "Transformer" : "Seq2Seq LSTM";
  modelBadge.textContent = `🤖 ${modelLabel}`;
  modelBadge.style.display = "inline-block";

  // Tokens
  if (data.source_tokens && data.source_tokens.length > 0) {
    tokenSection.style.display = "block";
    tokenList.style.display    = "flex";
    tokenToggle.textContent    = "Hide ▲";
    tokenList.innerHTML = data.source_tokens
      .map(tok => `<span class="token-chip">${escapeHTML(tok)}</span>`)
      .join("");
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(on) {
  state.loading = on;
  translateBtn.disabled      = on;
  btnSpinner.style.display   = on ? "inline-block" : "none";
  translateBtn.querySelector(".btn-text").textContent = on ? "..." : "Translate";
}

function showError(msg) {
  errorMsg.textContent    = `⚠ ${msg}`;
  errorBanner.style.display = "flex";
}
function hideError() {
  errorBanner.style.display = "none";
}

function escapeHTML(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;")
            .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ── Boot ──────────────────────────────────────────────────────────────────────
updateLangLabels();
updateActiveCard();

// Optionally fetch BLEU scores from backend (if you expose a /metrics endpoint)
// For now just set placeholders — update with your actual trained values
$("bleu-tf-en-de").textContent     = "18.3";
$("bleu-seq2seq-en-de").textContent = "8.5";