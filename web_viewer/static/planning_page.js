(function () {
  const data = window.PLANNING_PAGE_DATA || {};
  const expId = data.expId;
  if (!expId) return;

  const runBtn = document.getElementById("planning-run-btn");
  const statusEl = document.getElementById("planning-status");
  const checkpointInput = document.getElementById("planning-checkpoint");

  function cacheBust(url) {
    if (!url) return "";
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}t=${Date.now()}`;
  }

  function parseTile(text) {
    const parts = String(text || "").split(",").map((s) => s.trim());
    if (parts.length !== 2) throw new Error(`Invalid tile '${text}', expected row,col.`);
    const row = Number(parts[0]);
    const col = Number(parts[1]);
    if (!Number.isFinite(row) || !Number.isFinite(col)) {
      throw new Error(`Invalid tile '${text}', expected numeric row,col.`);
    }
    return [Math.trunc(row), Math.trunc(col)];
  }

  function collectConfig() {
    const result = {};
    document.querySelectorAll(".planning-config-input").forEach((el) => {
      const key = el.dataset.key;
      if (!key) return;
      const raw = el.value;
      if (raw === "true") {
        result[key] = true;
        return;
      }
      if (raw === "false") {
        result[key] = false;
        return;
      }
      if (raw !== "" && !Number.isNaN(Number(raw))) {
        result[key] = raw.includes(".") ? Number(raw) : Number.parseInt(raw, 10);
        return;
      }
      result[key] = raw;
    });
    return result;
  }

  function collectTraces() {
    const traces = [];
    document.querySelectorAll(".planning-column").forEach((colEl, idx) => {
      const label = colEl.dataset.label || `test${idx + 1}`;
      const start = parseTile(colEl.querySelector(".trace-start")?.value || "");
      const goal = parseTile(colEl.querySelector(".trace-goal")?.value || "");
      traces.push({ label, start, goal });
    });
    return traces;
  }

  function setImage(el, url) {
    if (!el) return;
    if (url) {
      el.src = cacheBust(url);
      el.style.display = "";
    } else {
      el.removeAttribute("src");
      el.style.display = "none";
    }
  }

  function updateResult(result) {
    const checkpointStep = result && result.checkpoint_step != null ? result.checkpoint_step : "-";
    const stepLabel = checkpointStep === "-" ? "" : `step=${checkpointStep}`;
    document.querySelectorAll(".checkpoint-step").forEach((el) => {
      el.textContent = stepLabel;
    });

    const columnsByLabel = {};
    (result?.columns || []).forEach((col) => {
      columnsByLabel[col.label] = col;
    });

    document.querySelectorAll(".planning-column").forEach((colEl) => {
      const label = colEl.dataset.label;
      const col = columnsByLabel[label] || null;
      setImage(colEl.querySelector(".execution-trace"), col?.execution_trace_url || "");
      setImage(colEl.querySelector(".planning-graph-h"), col?.planning_graph_h_url || "");
      setImage(colEl.querySelector(".h-grid-dist"), col?.h_grid_dist_url || "");
      setImage(colEl.querySelector(".planning-lattice"), col?.planning_lattice_url || "");
    });

    document.getElementById("planning-run-id").textContent = result?.run_id || "-";
    document.getElementById("planning-checkpoint-label").textContent = result?.checkpoint || "-";
    const configUrlEl = document.getElementById("planning-config-url");
    const runInfoUrlEl = document.getElementById("planning-runinfo-url");
    if (result?.planning_config_url) {
      configUrlEl.href = result.planning_config_url;
      configUrlEl.textContent = "planning_config.txt";
    } else {
      configUrlEl.href = "#";
      configUrlEl.textContent = "-";
    }
    if (result?.run_info_url) {
      runInfoUrlEl.href = result.run_info_url;
      runInfoUrlEl.textContent = "run_info.json";
    } else {
      runInfoUrlEl.href = "#";
      runInfoUrlEl.textContent = "-";
    }
  }

  async function pollJob(url) {
    for (;;) {
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`Polling failed: ${resp.status}`);
      }
      const payload = await resp.json();
      if (payload.status === "completed") {
        return payload.result;
      }
      if (payload.status === "failed") {
        throw new Error(payload.error || "Planning eval failed.");
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  async function runPlanning() {
    if (!runBtn) return;
    runBtn.disabled = true;
    statusEl.textContent = "Running...";
    try {
      const requestBody = {
        planning_config: collectConfig(),
        traces: collectTraces(),
      };
      const checkpointName = checkpointInput?.value?.trim();
      if (checkpointName) {
        requestBody.checkpoint_name = checkpointName;
      }
      const resp = await fetch(`/api/planning/${expId}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `Run failed (${resp.status}).`);
      }
      const job = await resp.json();
      const result = await pollJob(job.poll_url);
      updateResult(result);
      statusEl.textContent = "Completed";
    } catch (err) {
      statusEl.textContent = `Error: ${err.message}`;
    } finally {
      runBtn.disabled = false;
    }
  }

  if (runBtn) {
    runBtn.addEventListener("click", () => {
      runPlanning();
    });
  }

  updateResult(data.latestResult || null);
})();
