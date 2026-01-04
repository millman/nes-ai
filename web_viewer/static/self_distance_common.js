// Common self-distance page logic shared between self_distance_page.html, self_distance_h.html, and self_distance_zhs.html
//
// Usage: Each page should define SELF_DISTANCE_CONFIG with:
// - csvUrl: URL to the self-distance CSV file
// - redirectUrl: URL format for experiment selector redirect
// - errorMessage: Error message prefix (e.g., "self_distance_z.csv")
//
// Then call: initializeSelfDistancePage(SELF_DISTANCE_CONFIG, serverData);

function initializeSelfDistancePage(config, serverData) {
    const { csvUrl, redirectUrl, errorMessage } = config;
    const { selfDistanceMap, selfDistanceSteps, diagFigure, initialStep } = serverData;

    if (typeof Plotly === "undefined") {
        const firstPlot = document.getElementById("self-distance-first-plot");
        if (firstPlot) {
            firstPlot.innerHTML = "<p class='text-danger small'>Plotly failed to load.</p>";
        }
        return;
    }

    const plotConfig = buildPlotlyConfig({});
    const firstPlot = document.getElementById("self-distance-first-plot");
    const priorPlot = document.getElementById("self-distance-prior-plot");
    const scatterPlot = document.getElementById("self-distance-scatter-plot");
    const firstPlotCos = document.getElementById("self-distance-first-plot-cos");
    const priorPlotCos = document.getElementById("self-distance-prior-plot-cos");
    const scatterPlotCos = document.getElementById("self-distance-scatter-plot-cos");
    const firstImg = document.getElementById("self-distance-first-img");
    const priorImg = document.getElementById("self-distance-prior-img");
    const currentImg = document.getElementById("self-distance-current-img");
    const firstLabel = document.getElementById("self-distance-first-label");
    const priorLabel = document.getElementById("self-distance-prior-label");
    const currentLabel = document.getElementById("self-distance-current-label");
    const firstAction = document.getElementById("self-distance-first-action");
    const priorAction = document.getElementById("self-distance-prior-action");
    const currentAction = document.getElementById("self-distance-current-action");
    const frameRow = document.getElementById("self-distance-frame-row");
    const stepImg = document.getElementById("self-distance-step-img");
    const stepPath = document.getElementById("self-distance-step-path");
    const diagLossPlot = document.getElementById("self-distance-loss-plot");
    const diagLossWrapper = document.getElementById("self-distance-loss-wrapper");
    const diagLossPin = document.getElementById("self-distance-loss-pin");
    let diagLossPinned = false;

    let actionByTrajectory = {};

    const assetPath = (rel) => `/assets/${serverData.experimentId}/${rel.replace(/^\/+/, "")}`;

    const setImage = (imgEl, labelEl, actionEl, relPath, labelText, actionText) => {
        if (!imgEl || !labelEl) return;
        if (!relPath) {
            imgEl.classList.add("d-none");
            labelEl.textContent = "";
            if (actionEl) actionEl.textContent = "Action: (not available)";
            return;
        }
        imgEl.src = assetPath(relPath);
        imgEl.alt = labelText || relPath;
        imgEl.classList.remove("d-none");
        labelEl.textContent = labelText || relPath;
        if (actionEl) {
            actionEl.textContent = actionText ? `Action: ${actionText}` : "Action: (not available)";
        }
    };

    const updateFrameVisibility = (showPrior) => {
        if (!frameRow) return;
        const priorCol = frameRow.querySelector(".col-md-4:nth-child(2)");
        if (priorCol) {
            if (showPrior) {
                priorCol.classList.remove("d-none");
            } else {
                priorCol.classList.add("d-none");
            }
        }
    };

    const nearestSelfStep = (step) => {
        if (!selfDistanceSteps.length) return null;
        let best = selfDistanceSteps[0];
        let bestDiff = Math.abs(step - selfDistanceSteps[0]);
        for (let i = 1; i < selfDistanceSteps.length; i++) {
            const diff = Math.abs(step - selfDistanceSteps[i]);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = selfDistanceSteps[i];
            }
        }
        return best;
    };

    const updateSelfDistanceImage = (step) => {
        if (!stepImg || !stepPath) return;
        const matched = nearestSelfStep(step);
        if (matched !== null && selfDistanceMap[matched]) {
            stepImg.src = selfDistanceMap[matched].url;
            stepImg.classList.remove("d-none");
            stepPath.textContent = selfDistanceMap[matched].path;
        } else {
            stepImg.classList.add("d-none");
            stepPath.textContent = `No self-distance image for step ${step}`;
        }
    };

    const buildActionMap = (rows) => {
        const map = {};
        for (const row of rows) {
            const traj = row.trajectory || row.trajectory_label || "trajectory";
            if (!map[traj]) {
                map[traj] = {
                    firstAction: row.action_label || "",
                    firstStep: Number(row.timestep),
                    actions: {},
                };
            }
            const step = Number(row.timestep);
            if (row.action_label) {
                map[traj].actions[step] = row.action_label;
            }
        }
        return map;
    };

    const getFirstAction = (trajectory, fallback) => {
        const entry = actionByTrajectory[trajectory];
        if (!entry) return fallback || "";
        return entry.firstAction || fallback || "";
    };

    const getPriorAction = (trajectory, step, fallback) => {
        const entry = actionByTrajectory[trajectory];
        if (!entry) return fallback || "";
        const priorStep = Number(step) - 1;
        if (Number.isFinite(priorStep) && entry.actions && entry.actions[priorStep]) {
            return entry.actions[priorStep];
        }
        if (Number.isFinite(entry.firstStep) && Number.isFinite(step) && step <= entry.firstStep) {
            return entry.firstAction || fallback || "";
        }
        return fallback || "";
    };

    const attachHover = (plotEl, mode) => {
        plotEl.on("plotly_hover", (event) => {
            const point = event?.points?.[0];
            if (!point || !point.customdata) return;
            const payload = point.customdata;
            const trajectory = payload[0];
            const step = payload[1];
            const framePath = payload[2];
            const firstPath = payload[3];
            const priorPath = payload[4];
            const frameLabelText = payload[5] || `t=${step}`;
            const headerText = `${trajectory} • t=${step}`;
            const actionText = payload[6] || "";
            const firstActionText = getFirstAction(trajectory, actionText);
            const priorActionText = getPriorAction(trajectory, step, actionText);

            if (mode === "first") {
                updateFrameVisibility(false);
                setImage(firstImg, firstLabel, firstAction, firstPath, `first: ${frameLabelText}`, firstActionText);
                setImage(currentImg, currentLabel, currentAction, framePath, headerText, actionText);
                if (priorImg && priorLabel) {
                    priorImg.classList.add("d-none");
                    priorLabel.textContent = "";
                    if (priorAction) priorAction.textContent = "";
                }
            } else if (mode === "prior") {
                updateFrameVisibility(true);
                setImage(firstImg, firstLabel, firstAction, firstPath, `first: ${frameLabelText}`, firstActionText);
                setImage(priorImg, priorLabel, priorAction, priorPath, `prior: t=${Math.max(0, step - 1)}`, priorActionText);
                setImage(currentImg, currentLabel, currentAction, framePath, headerText, actionText);
            } else {
                updateFrameVisibility(true);
                setImage(firstImg, firstLabel, firstAction, firstPath, `first: ${frameLabelText}`, firstActionText);
                setImage(priorImg, priorLabel, priorAction, priorPath, `prior: t=${Math.max(0, step - 1)}`, priorActionText);
                setImage(currentImg, currentLabel, currentAction, framePath, headerText, actionText);
            }
        });
    };

    const buildPlots = (rows) => {
        if (!rows.length) {
            if (firstPlot) firstPlot.innerHTML = `<p class='text-muted fst-italic'>No rows in ${errorMessage}.</p>`;
            if (priorPlot) priorPlot.innerHTML = "";
            if (scatterPlot) scatterPlot.innerHTML = "";
            if (firstPlotCos) firstPlotCos.innerHTML = "";
            if (priorPlotCos) priorPlotCos.innerHTML = "";
            if (scatterPlotCos) scatterPlotCos.innerHTML = "";
            return;
        }
        const steps = rows.map(r => Number(r.timestep));
        const distFirst = rows.map(r => Number(r.distance_to_first));
        const distPrior = rows.map(r => Number(r.distance_to_prior));
        const distFirstCos = rows.map(r => Number(r.cosine_distance_to_first ?? r.distance_to_first_cosine ?? 0));
        const distPriorCos = rows.map(r => Number(r.cosine_distance_to_prior ?? r.distance_to_prior_cosine ?? 0));
        const custom = rows.map(r => [
            r.trajectory,
            Number(r.timestep),
            r.frame_path,
            r.first_frame_path,
            r.prior_frame_path,
            r.frame_label,
            r.action_label || "",
        ]);

        const layoutBase = {
            margin: { t: 32, r: 16, l: 50, b: 50 },
            template: "plotly_white",
            height: 300,
            title: { font: { size: 12 } },
            font: { size: 11 },
        };

        const layoutLineChart = {
            ...layoutBase,
            hovermode: "x unified",
            xaxis: {
                title: "timestep",
                showspikes: true,
                spikemode: "across",
                spikethickness: 1,
                spikedash: "dot",
                spikecolor: "#999",
            },
        };

        Plotly.newPlot(
            firstPlot,
            [
                {
                    x: steps,
                    y: distFirst,
                    mode: "lines+markers",
                    name: "distance to first",
                    marker: { size: 4 },
                    line: { width: 1.5 },
                    customdata: custom,
                    hovertemplate: "%{y:.4f}<extra></extra>",
                },
            ],
            {
                ...layoutLineChart,
                title: "Distance to first latent",
                yaxis: { title: "||z0 - zt||" },
            },
            plotConfig,
        ).then(() => attachHover(firstPlot, "first"));

        Plotly.newPlot(
            priorPlot,
            [
                {
                    x: steps,
                    y: distPrior,
                    mode: "lines+markers",
                    name: "distance to prior",
                    marker: { size: 4, color: "orange" },
                    line: { width: 1.5, color: "orange" },
                    customdata: custom,
                    hovertemplate: "%{y:.4f}<extra></extra>",
                },
            ],
            {
                ...layoutLineChart,
                title: "Distance to prior latent",
                yaxis: { title: "||z(t-1) - zt||" },
            },
            plotConfig,
        ).then(() => attachHover(priorPlot, "prior"));

        Plotly.newPlot(
            scatterPlot,
            [
                {
                    x: distFirst,
                    y: distPrior,
                    mode: "markers",
                    name: "distance pairs",
                    marker: {
                        size: 5,
                        color: steps,
                        colorscale: "Viridis",
                        showscale: true,
                        colorbar: { title: "timestep" },
                    },
                    customdata: custom,
                    hovertemplate: "dist vs first=%{x:.4f}<br>dist vs prior=%{y:.4f}<extra></extra>",
                },
            ],
            {
                ...layoutBase,
                hovermode: "closest",
                title: "Distance vs first vs prior",
                xaxis: { title: "||z0 - zt||" },
                yaxis: { title: "||z(t-1) - zt||" },
            },
            plotConfig,
        ).then(() => attachHover(scatterPlot, "scatter"));

        Plotly.newPlot(
            firstPlotCos,
            [
                {
                    x: steps,
                    y: distFirstCos,
                    mode: "lines+markers",
                    name: "cosine distance to first",
                    marker: { size: 4, color: "green" },
                    line: { width: 1.5, color: "green" },
                    customdata: custom,
                    hovertemplate: "%{y:.4f}<extra></extra>",
                },
            ],
            {
                ...layoutLineChart,
                title: "Cosine distance to first latent",
                yaxis: { title: "1 - cos(z0, zt)" },
            },
            plotConfig,
        ).then(() => attachHover(firstPlotCos, "first"));

        Plotly.newPlot(
            priorPlotCos,
            [
                {
                    x: steps,
                    y: distPriorCos,
                    mode: "lines+markers",
                    name: "cosine distance to prior",
                    marker: { size: 4, color: "red" },
                    line: { width: 1.5, color: "red" },
                    customdata: custom,
                    hovertemplate: "%{y:.4f}<extra></extra>",
                },
            ],
            {
                ...layoutLineChart,
                title: "Cosine distance to prior latent",
                yaxis: { title: "1 - cos(z(t-1), zt)" },
            },
            plotConfig,
        ).then(() => attachHover(priorPlotCos, "prior"));

        Plotly.newPlot(
            scatterPlotCos,
            [
                {
                    x: distFirstCos,
                    y: distPriorCos,
                    mode: "markers",
                    name: "cosine distance pairs",
                    marker: {
                        size: 5,
                        color: steps,
                        colorscale: "Plasma",
                        showscale: true,
                        colorbar: { title: "timestep" },
                    },
                    customdata: custom,
                    hovertemplate: "cos dist vs first=%{x:.4f}<br>cos dist vs prior=%{y:.4f}<extra></extra>",
                },
            ],
            {
                ...layoutBase,
                hovermode: "closest",
                title: "Cosine distance: first vs prior",
                xaxis: { title: "1 - cos(z0, zt)" },
                yaxis: { title: "1 - cos(z(t-1), zt)" },
            },
            plotConfig,
        ).then(() => attachHover(scatterPlotCos, "scatter"));
    };

    Papa.parse(csvUrl, {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
            const rows = results.data;
            actionByTrajectory = buildActionMap(rows);
            buildPlots(rows);
            if (rows.length) {
                const firstRow = rows[0];
                const trajectory = firstRow.trajectory || firstRow.trajectory_label || "trajectory";
                const step = Number(firstRow.timestep);
                const actionText = firstRow.action_label || "";
                const firstActionText = getFirstAction(trajectory, actionText);
                const priorActionText = getPriorAction(trajectory, step, actionText);
                setImage(firstImg, firstLabel, firstAction, firstRow.first_frame_path, `first: ${firstRow.frame_label || "t=0"}`, firstActionText);
                setImage(priorImg, priorLabel, priorAction, firstRow.prior_frame_path, "prior: t=0", priorActionText);
                setImage(currentImg, currentLabel, currentAction, firstRow.frame_path, `t=${firstRow.timestep}`, actionText);
                updateFrameVisibility(true);
            }
        },
        error: (err) => {
            if (firstPlot) {
                firstPlot.innerHTML = `<p class="text-danger small">Error loading ${errorMessage}: ${err.message}</p>`;
            }
        }
    });

    updateSelfDistanceImage(initialStep);

    const setLossPinned = (pinned) => {
        if (!diagLossWrapper || !diagLossPin) return;
        diagLossPinned = pinned;
        diagLossWrapper.classList.toggle("sticky-active", pinned);
        diagLossPin.classList.toggle("active", pinned);
        diagLossPin.setAttribute("aria-pressed", pinned ? "true" : "false");
        diagLossPin.setAttribute("title", pinned ? "Unpin training metrics plot" : "Pin training metrics plot");
    };

    if (diagLossPlot && diagFigure && diagFigure.data && diagFigure.layout) {
        const diagLayout = {
            ...diagFigure.layout,
            height: 220,
            margin: { ...(diagFigure.layout?.margin || {}), t: 30, b: 50, l: 50, r: 10 },
        };
        Plotly.react(diagLossPlot, diagFigure.data, diagLayout, buildPlotlyConfig(diagFigure.config || {})).then(() => {
            diagLossPlot.on("plotly_hover", (event) => {
                const point = event?.points?.[0];
                if (!point || typeof point.x === "undefined") return;
                updateSelfDistanceImage(point.x);
            });
        });
    }

    if (diagLossPin && diagLossWrapper) {
        diagLossPin.addEventListener("click", () => {
            setLossPinned(!diagLossPinned);
        });
        setLossPinned(true);
    }

    // Setup experiment selector if present
    const select = document.getElementById("self-distance-select");
    if (select && redirectUrl) {
        select.addEventListener("change", () => {
            const id = select.value;
            window.location.href = redirectUrl + encodeURIComponent(id);
        });
    }

    // Bootstrap tooltips
    const tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach((el) => {
        if (window.bootstrap) {
            new bootstrap.Tooltip(el);
        }
    });
}
