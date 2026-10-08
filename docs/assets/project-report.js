"use strict";

(() => {
  const data = window.PROJECT_REPORT_DATA;
  if (!data) return;
  const number = (value, digits = 2) => value === null || value === undefined
    ? "—" : new Intl.NumberFormat("ru-RU", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value);
  const byId = (id) => document.getElementById(id);
  const state = { horizon: 1, split: "holdout", warningWindow: 1 };
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function renderBars(container, rows, field, format) {
    container.replaceChildren();
    const finite = rows.filter((row) => Number.isFinite(row[field]));
    if (!finite.length) {
      container.append(element("p", "margin-note", "Для этого периода и горизонта оценка отсутствует. Неизвестная метрика не заменяется нулём."));
      return;
    }
    const maximum = Math.max(...finite.map((row) => row[field]), 0.001);
    finite.forEach((row) => {
      const line = element("div", "bar-row");
      line.dataset.family = row.family || "";
      if (row.model === "SeasonalNaiveYoY") line.classList.add("is-baseline");
      const label = element("div", "bar-label", row.label || row.method || row.model);
      if (state.horizon === 12 && row.fallback_count > 0 && field === "mae_macro") {
        label.append(element("small", "", "SeasonalNaive fallback"));
      }
      const track = element("div", "bar-track");
      track.setAttribute("aria-hidden", "true");
      const fill = element("div", "bar-fill");
      fill.style.width = String(row[field] / maximum * 100) + "%";
      track.append(fill);
      line.append(label, track, element("div", "bar-value", format(row[field])));
      container.append(line);
    });
  }

  function renderForecast() {
    const rows = data.forecasting.rows.filter((row) => row.split === state.split && row.horizon === state.horizon);
    renderBars(byId("forecast-chart"), rows, "mae_macro", (value) => number(value));
    byId("forecast-chart-title").textContent = "MAE macro · рубли · h = " + state.horizon + " · " + state.split;
    const measured = rows.find((row) => Number.isFinite(row.mae_macro));
    byId("forecast-status").textContent = measured
      ? number(measured.n_predictions, 0) + " сопоставимых случаев · " + number(measured.n_municipalities, 0) + " МО · " + number(measured.n_origins, 0) + " дат выпуска. Оценена полная стратегия с резервом."
      : "h = 12 не имеет validation-случаев. Сравнение не оценено.";
    document.querySelectorAll("[data-horizon]").forEach((button) =>
      button.setAttribute("aria-pressed", String(Number(button.dataset.horizon) === state.horizon)));
  }

  const detectionMetrics = {
    f1: { label: "F1", help: "F1 — гармоническое среднее precision и recall.", digits: 3 },
    precision: { label: "Precision", help: "Доля сигналов или границ, сопоставленных с размеченным событием.", digits: 3 },
    recall: { label: "Recall", help: "Доля размеченных событий, для которых обнаружен сигнал или граница.", digits: 3 },
    false_alarms_per_12: { field: "false_positives_per_12_months", label: "Ложные сигналы / 12 месяцев", help: "Число ложных сигналов на 12 месяцев мониторинга; меньше — лучше.", digits: 3 }
  };
  function renderDetection() {
    const selected = byId("detection-metric").value;
    const metric = detectionMetrics[selected];
    const rows = data.detection.rows.map((row) => ({ ...row, label: (row.method || row.model) + " · " + row.family }));
    renderBars(byId("detection-chart"), rows, metric.field || selected, (value) => number(value, metric.digits));
    byId("detection-chart-title").textContent = metric.label + " · фиксированные рабочие точки · TEST";
    byId("detection-metric-help").textContent = metric.help;
  }

  function renderLine(id, field, label, digits) {
    const rows = data.financial_indicators.origins;
    const container = byId(id);
    const valid = rows.map((row, index) => ({ row, index })).filter(({ row }) => Number.isFinite(row[field]));
    if (!valid.length) {
      container.append(element("p", "", "Для ряда нет доступных значений."));
      return;
    }
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 520 270");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label + ". Двенадцать значений на датах выпуска; точные числа приведены в таблице.");
    const add = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (text !== undefined) node.textContent = text;
      svg.append(node);
      return node;
    };
    add("title", {}, label + " на датах выпуска прогноза");
    const values = valid.map(({ row }) => row[field]);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const padding = Math.max((high - low) * 0.15, 0.5);
    const bottom = low - padding;
    const top = high + padding;
    const x = (index) => 54 + index / Math.max(rows.length - 1, 1) * 444;
    const y = (value) => 225 - (value - bottom) / (top - bottom) * 199;
    for (let step = 0; step <= 3; step++) {
      const value = bottom + (top - bottom) * step / 3;
      add("line", { x1: 54, x2: 498, y1: y(value), y2: y(value), class: "grid-line" });
      add("text", { x: 43, y: y(value) + 4, "text-anchor": "end" }, number(value, 1));
    }
    [0, 3, 6, 9, 11].filter((index) => index < rows.length).forEach((index) => {
      const date = rows[index].forecast_origin.slice(0, 7);
      add("text", { x: x(index), y: 251, "text-anchor": "middle" }, date.slice(5) + "." + date.slice(2, 4));
    });
    let path = "";
    let previousIndex = -2;
    valid.forEach(({ row, index }) => {
      path += (index === previousIndex + 1 ? "L" : "M") + x(index) + "," + y(row[field]) + " ";
      previousIndex = index;
    });
    add("path", { d: path, class: "series" });
    valid.forEach(({ row, index }) => {
      const dot = add("circle", { cx: x(index), cy: y(row[field]), r: 4, class: "point" });
      const title = document.createElementNS(ns, "title");
      title.textContent = row.forecast_origin.slice(0, 10) + ": " + number(row[field], digits);
      dot.append(title);
    });
    container.replaceChildren(svg);
  }

  function renderCohort() {
    const k = state.warningWindow;
    const cohort = data.early_warning.real.cohorts.find((row) => row.k === k);
    const originRows = data.early_warning.real.origins;
    const container = byId("warning-cohort");
    container.replaceChildren();
    const names = { positive: "Положительная", negative: "Отрицательная", unknown: "Неизвестная" };
    originRows.forEach((row) => {
      const status = row["k" + k + "_label"];
      const cell = element("div", "cohort-date", row.forecast_origin.slice(5, 7) + "." + row.forecast_origin.slice(2, 4));
      cell.dataset.status = status;
      cell.setAttribute("aria-label", row.forecast_origin + ": " + names[status]);
      cell.append(element("span", "", status === "positive" ? "+" : status === "negative" ? "−" : "?"));
      container.append(cell);
    });
    byId("cohort-status").textContent = cohort.eligible + " известных дат · " + cohort.positive + " положительных · " + cohort.negative + " отрицательных · " + cohort.unknown + " неизвестных";
    document.querySelectorAll("[data-warning-window]").forEach((button) =>
      button.setAttribute("aria-pressed", String(Number(button.dataset.warningWindow) === k)));
  }

  const normalize = (value) => value.toLocaleLowerCase("ru-RU").replaceAll("ё", "е").normalize("NFKC");
  function filterGlossary() {
    const query = normalize(byId("glossary-search").value.trim());
    const selectedGroup = byId("glossary-group").value;
    let count = 0;
    data.glossary.terms.forEach((term) => {
      const target = byId(term.id);
      const matches = (!selectedGroup || selectedGroup === term.group_id)
        && normalize(term.en + " " + term.ru + " " + term.definition).includes(query);
      target.hidden = !matches;
      if (matches) count++;
    });
    document.querySelectorAll(".glossary-group").forEach((group) => {
      group.hidden = !Array.from(group.querySelectorAll(".glossary-term")).some((term) => !term.hidden);
    });
    byId("glossary-count").textContent = "Показано " + count + " из " + data.glossary.terms.length + " терминов";
    byId("glossary-empty").hidden = count !== 0;
  }
  function resetGlossary() {
    byId("glossary-search").value = "";
    byId("glossary-group").value = "";
    filterGlossary();
  }
  function revealHash() {
    let id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
    const target = byId(id);
    if (!target || !target.classList.contains("glossary-term")) return;
    resetGlossary();
    target.open = true;
    requestAnimationFrame(() => {
      target.scrollIntoView({ block: "start", behavior: "auto" });
      target.querySelector("summary").focus({ preventScroll: true });
    });
  }
  function setTheme(dark, persist = false) {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    const toggle = byId("theme-toggle");
    toggle.textContent = dark ? "Светлая тема" : "Тёмная тема";
    toggle.setAttribute("aria-pressed", String(dark));
    toggle.setAttribute("aria-label", dark ? "Включить светлую тему" : "Включить тёмную тему");
    if (persist) {
      try { localStorage.setItem("sberindex-report-theme", dark ? "dark" : "light"); } catch { /* File access can disable storage. */ }
    }
  }

  document.querySelectorAll("[data-horizon]").forEach((button) => button.addEventListener("click", () => {
    state.horizon = Number(button.dataset.horizon);
    renderForecast();
  }));
  byId("forecast-split").addEventListener("change", (event) => {
    state.split = event.target.value;
    renderForecast();
  });
  byId("detection-metric").addEventListener("change", renderDetection);
  document.querySelectorAll("[data-warning-window]").forEach((button) => button.addEventListener("click", () => {
    state.warningWindow = Number(button.dataset.warningWindow);
    renderCohort();
  }));
  data.glossary.groups.forEach((group) => {
    const option = element("option", "", group.title);
    option.value = group.id;
    byId("glossary-group").append(option);
  });
  byId("glossary-search").addEventListener("input", filterGlossary);
  byId("glossary-group").addEventListener("change", filterGlossary);
  byId("glossary-reset").addEventListener("click", () => {
    resetGlossary();
    byId("glossary-search").focus();
  });
  byId("theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme !== "dark", true));
  window.addEventListener("hashchange", revealHash);
  let savedTheme = null;
  try { savedTheme = localStorage.getItem("sberindex-report-theme"); } catch { /* Optional preference. */ }
  setTheme(savedTheme ? savedTheme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  renderForecast();
  renderDetection();
  renderLine("rate-chart", "key_rate_level", "Ключевая ставка Банка России, проценты", 2);
  renderLine("fx-chart", "usd_rub_last", "Официальный USD/RUB Банка России, рубли за доллар", 4);
  renderCohort();
  filterGlossary();
  document.documentElement.classList.add("is-interactive");
  const sourceTable = document.querySelector(".data-table-disclosure");
  if (sourceTable) sourceTable.open = false;
  revealHash();
})();
