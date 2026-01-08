// Common diagnostics page logic shared between all diagnostics views (Z, H, P, ZHP)
// Used by diagnostics_page.html and diagnostics_page_zhs.html templates
//
// Usage: Each page should define DIAGNOSTICS_CONFIG with:
// - targets: Object mapping diagnostic keys to DOM elements and metadata
// - hasAlignmentSummary: boolean (whether to show alignment summary table)
// - hasFrameSteps: boolean (whether to include frameSteps calculation)
// - hasCsvLinks: boolean (whether to include CSV download links)
//
// Then call: initializeDiagnosticsPage(DIAGNOSTICS_CONFIG, serverData);

function initializeDiagnosticsPage(config, serverData) {
    const { targets, hasAlignmentSummary = false, hasFrameSteps = false, hasCsvLinks = false } = config;
    const { steps, diagnostics, frameMap, diagSummary, figure, diagScalars } = serverData;

    const frameSteps = hasFrameSteps
        ? Object.keys(frameMap || {}).map((k) => Number(k)).filter((n) => !Number.isNaN(n)).sort((a, b) => a - b)
        : [];

    const stepLabels = document.querySelectorAll(".step-label-value");
    const lossPlot = document.getElementById("diagnostics-loss-plot");
    const lossWrapper = document.getElementById("diagnostics-loss-wrapper");
    const lossPinButton = document.getElementById("diagnostics-loss-pin");
    const topGrid = document.getElementById("diagnostics-top-grid");
    const toggleBtn = document.getElementById("diagnostics-toggle-btn");

    const frameSlider = document.getElementById("diagnostics-frame");
    const frameCount = document.getElementById("diagnostics-frame-count");
    const frameLabel = document.getElementById("diagnostics-frame-label");
    const frameImg = document.getElementById("diagnostics-frame-img");
    const framePath = document.getElementById("diagnostics-frame-path");
    const frameAction = document.getElementById("diagnostics-frame-action");
    const frameDimensions = document.getElementById("diagnostics-frame-dimensions");

    const summaryWrapper = hasAlignmentSummary ? document.getElementById("diagnostics-summary-wrapper") : null;
    const summaryBody = hasAlignmentSummary ? document.getElementById("diagnostics-summary-body") : null;
    const summaryStep = hasAlignmentSummary ? document.getElementById("diagnostics-summary-step") : null;
    const summaryLinks = hasAlignmentSummary ? document.getElementById("diagnostics-summary-links") : null;
    const summaryCaption = hasAlignmentSummary ? document.getElementById("diagnostics-summary-caption") : null;

    const statusItems = document.querySelectorAll("[data-status-metric]");
    const scalarSteps = Object.keys(diagScalars || {})
        .map((k) => Number(k))
        .filter((n) => !Number.isNaN(n))
        .sort((a, b) => a - b);

    let currentFrames = [];
    let lossPinned = false;

    // Update CSV links for the selected step if present (no-op fallback).
    function populateCsvLinks(step) {
        if (!hasCsvLinks) return;
        const csvLink = document.getElementById("diagnostics-csv-link");
        if (!csvLink || !diagnostics?.csv_base_url) return;
        const nearest = nearestStepAtOrBelow(step);
        const resolvedStep = nearest !== null ? nearest : step;
        csvLink.href = `${diagnostics.csv_base_url}/${resolvedStep}.csv`;
        csvLink.classList.remove("d-none");
    }

    function resolveUrl(diagKey, step) {
        const perDiag = diagnostics[diagKey] || {};
        return perDiag[step] || perDiag[String(step)] || null;
    }

    function updateFrameForIndex(idx) {
        if (!frameImg || !frameLabel || !currentFrames.length) return;
        const clamped = Math.min(Math.max(idx, 0), currentFrames.length - 1);
        const frame = currentFrames[clamped];
        frameImg.src = frame.url;
        frameImg.classList.remove("d-none");
        frameLabel.textContent = `Frame ${clamped + 1} / ${currentFrames.length}`;
        if (frameAction) {
            const label = frame.action || (frame.action_id ? `action ${frame.action_id}` : "");
            frameAction.textContent = label ? `Action: ${label}` : "Action: (unknown)";
        }
        if (framePath) {
            framePath.textContent = frame.source || "Unknown source path";
        }
        if (frameDimensions) {
            frameDimensions.textContent = frameImg.naturalWidth && frameImg.naturalHeight
                ? `Image size: ${frameImg.naturalWidth}×${frameImg.naturalHeight}px`
                : "Loading dimensions…";
        }
    }

    function updateFrameSlider(step) {
        if (!frameSlider || !frameCount || !frameLabel || !frameImg) return;
        const frames = frameMap[step] || frameMap[String(step)] || [];
        currentFrames = frames;
        if (!frames.length) {
            frameSlider.disabled = true;
            frameSlider.max = 0;
            frameSlider.value = 0;
            frameCount.textContent = "0";
            frameLabel.textContent = "No frames for this step";
            frameImg.classList.add("d-none");
            if (framePath) {
                framePath.textContent = "";
            }
            if (frameAction) {
                frameAction.textContent = "";
            }
            if (frameDimensions) {
                frameDimensions.textContent = "";
            }
            return;
        }
        frameSlider.disabled = false;
        frameSlider.max = frames.length - 1;
        const newVal = Math.min(Number(frameSlider.value || 0), frames.length - 1);
        frameSlider.value = newVal;
        frameCount.textContent = String(frames.length);
        updateFrameForIndex(newVal);
    }

    function formatNumber(value, decimals = 3, multiplier = 1) {
        if (value === null || value === undefined) return "—";
        const num = Number(value);
        if (!Number.isFinite(num)) return "—";
        return (num * multiplier).toFixed(decimals);
    }

    function updateStatusStrip(stepValue) {
        if (!statusItems.length || !scalarSteps.length) return;
        let nearest = null;
        for (let i = 0; i < scalarSteps.length; i++) {
            const step = scalarSteps[i];
            if (step > stepValue) {
                continue;
            }
            if (nearest === null || step > nearest) {
                nearest = step;
            }
        }
        const step = nearest !== null ? nearest : scalarSteps[scalarSteps.length - 1];
        const values = (diagScalars || {})[step] || {};
        statusItems.forEach((item) => {
            const key = item.getAttribute("data-status-metric");
            if (!key) return;
            item.textContent = formatNumber(values[key], 3);
        });
    }

    function renderAlignmentSummary(summary) {
        if (!hasAlignmentSummary || !summaryWrapper) return;
        if (!summary || !Array.isArray(summary.rows) || summary.rows.length === 0) {
            summaryWrapper.classList.add("d-none");
            return;
        }
        summaryWrapper.classList.remove("d-none");
        if (summaryStep) {
            summaryStep.textContent = summary.step !== null && summary.step !== undefined ? `Step ${summary.step}` : "Latest";
        }
        if (summaryBody) {
            summaryBody.innerHTML = "";
            const rows = summary.rows.slice(0, 12);
            rows.forEach((row) => {
                const tr = document.createElement("tr");
                const notes = [row.note, row.strength_note].filter((n) => n && n.trim());
                const cells = [
                    row.label || `Action ${row.action_id}`,
                    row.count ?? "—",
                    formatNumber(row.mean, 3),
                    `${formatNumber(row.pct_high, 1, 100)}%`,
                    formatNumber(row.delta_norm_median, 4),
                    formatNumber(row.delta_norm_p90, 4),
                    formatNumber(row.strength_ratio ?? row.mean_dir_norm, 3),
                    notes.length ? notes.join("; ") : "",
                ];
                cells.forEach((value) => {
                    const td = document.createElement("td");
                    td.textContent = value === undefined ? "—" : value;
                    tr.appendChild(td);
                });
                summaryBody.appendChild(tr);
            });
        }
        if (summaryLinks) {
            const links = [];
            if (summary.report_url) {
                const label = summary.report_name || "Report";
                links.push(`<a href="${summary.report_url}" target="_blank" rel="noreferrer">${label}</a>`);
            }
            if (summary.strength_url && summary.strength_url !== summary.report_url) {
                const label = summary.strength_name || "Strength";
                links.push(`<a href="${summary.strength_url}" target="_blank" rel="noreferrer">${label}</a>`);
            }
            const extras = [];
            if (!summary.report_url && summary.report_name) {
                extras.push(summary.report_name);
            }
            if (!summary.strength_url && summary.strength_name) {
                extras.push(summary.strength_name);
            }
            const extraText = extras.length ? extras.join(" · ") : "";
            const linkText = links.length ? links.join(" · ") : "";
            const combined = [linkText, extraText].filter(Boolean).join(" · ");
            summaryLinks.innerHTML = combined ? `Sources: ${combined}` : "";
        }
        if (summaryCaption) {
            summaryCaption.textContent = "Parsed from the latest action_alignment_report and action_alignment_strength outputs.";
        }
    }

    function nearestStepAtOrBelow(target) {
        if (!steps.length) return null;
        let best = null;
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            if (step > target) {
                continue;
            }
            if (best === null || step > best) {
                best = step;
            }
        }
        return best;
    }

    function updateForStep(stepValue) {
        const nearest = nearestStepAtOrBelow(stepValue);
        const step = nearest !== null ? nearest : steps[steps.length - 1];
        stepLabels.forEach((label) => {
            label.textContent = step ?? "—";
        });
        Object.entries(targets).forEach(([key, refs]) => {
            if (!refs.img || !refs.caption) return;
            const url = resolveUrl(key, step);
            const headerText = refs.title ? refs.title : "";
            if (url) {
                refs.img.src = url;
                refs.img.classList.remove("d-none");
                if (refs.header && refs.title) {
                    const headerTitle = refs.header.querySelector(".image-preview-card-title");
                    if (headerTitle) {
                        headerTitle.textContent = headerText;
                    }
                }
                refs.caption.textContent = "";
            } else {
                refs.img.classList.add("d-none");
                refs.caption.textContent = `No image for step ${step} in ${refs.label || key}`;
            }
        });
        populateCsvLinks(step);
        updateFrameSlider(step);
        if (hasAlignmentSummary) {
            renderAlignmentSummary(diagSummary);
        }
        updateStatusStrip(step ?? 0);
    }

    function renderLossPlot() {
        if (!lossPlot) return;
        if (typeof Plotly === "undefined") {
            return;
        }
        if (!figure || !figure.data || !figure.layout) {
            return;
        }
        const layout = {
            ...figure.layout,
            height: 220,
            margin: { ...(figure.layout?.margin || {}), t: 30, b: 50, l: 50, r: 10 },
        };
        const config = buildPlotlyConfig(figure.config || {});
        Plotly.react(lossPlot, figure.data, layout, config).then(() => {
            lossPlot.on("plotly_hover", (event) => {
                const point = event?.points?.[0];
                if (!point || typeof point.x === "undefined") return;
                updateForStep(point.x);
            });
        });
    }

    // Toggle sticky positioning for the diagnostics loss plot.
    function setLossPinned(pinned) {
        if (!lossWrapper || !lossPinButton) return;
        lossPinned = pinned;
        lossWrapper.classList.toggle("sticky-active", pinned);
        lossPinButton.classList.toggle("active", pinned);
        lossPinButton.setAttribute("aria-pressed", pinned ? "true" : "false");
        lossPinButton.setAttribute("title", pinned ? "Unpin diagnostics plot" : "Pin diagnostics plot");
    }

    // Ensure initial state is rendered.
    setLossPinned(true);

    if (frameSlider) {
        frameSlider.addEventListener("input", (ev) => {
            updateFrameForIndex(Number(ev.target.value));
        });
        frameSlider.addEventListener("pointermove", (ev) => {
            if (!currentFrames.length || frameSlider.disabled) return;
            const rect = frameSlider.getBoundingClientRect();
            if (!rect || rect.width === 0) return;
            const ratio = Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1);
            const idx = Math.round(ratio * (currentFrames.length - 1));
            frameSlider.value = String(idx);
            updateFrameForIndex(idx);
        });
    }
    if (frameImg && frameDimensions) {
        frameImg.addEventListener("load", () => {
            if (!frameImg.naturalWidth || !frameImg.naturalHeight) {
                frameDimensions.textContent = "";
                return;
            }
            frameDimensions.textContent = `Image size: ${frameImg.naturalWidth}×${frameImg.naturalHeight}px`;
        });
    }

    renderLossPlot();
    const initialStep = steps.length ? steps[steps.length - 1] : (frameSteps.length ? frameSteps[frameSteps.length - 1] : null);
    const initialFrameStep = frameSteps.length ? frameSteps[0] : initialStep;
    if (initialFrameStep !== null) {
        updateFrameSlider(initialFrameStep);
    }
    if (initialStep !== null) {
        updateForStep(initialStep);
    } else if (hasAlignmentSummary) {
        renderAlignmentSummary(diagSummary);
    }

    if (toggleBtn && topGrid) {
        topGrid.addEventListener("shown.bs.collapse", () => {
            toggleBtn.textContent = "Collapse Delta/Alignment";
            toggleBtn.setAttribute("aria-expanded", "true");
        });
        topGrid.addEventListener("hidden.bs.collapse", () => {
            toggleBtn.textContent = "Expand Delta/Alignment";
            toggleBtn.setAttribute("aria-expanded", "false");
        });
    }
    if (lossPinButton && lossWrapper) {
        lossPinButton.addEventListener("click", () => {
            setLossPinned(!lossPinned);
        });
    }
}
