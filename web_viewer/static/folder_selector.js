// Shared folder selector component for image preview selection
// Used by Compare and Detail pages

// Uses IMAGE_FOLDER_SPECS, getImageOption from image_card_common.js

// Folder group presets - maps group names to folder arrays
const FOLDER_GROUP_PRESETS = {
  rollout_f0: ["vis_fixed_0"],
  rollout_f1: ["vis_fixed_1"],
  rollout_r0: ["vis_rolling_0"],
  rollout_r1: ["vis_rolling_1"],
  health_checks: [
    "vis_rollout_divergence",
    "vis_rollout_divergence_z",
    "vis_rollout_divergence_h",
    "vis_rollout_divergence_p",
    "vis_h_ablation",
    "vis_h_drift_by_action",
    "vis_norm_timeseries",
    "vis_z_consistency",
    "vis_z_monotonicity",
  ],
  // Diags: Action alignment, Self-distance, Delta PCA
  diags_z: ["vis_action_alignment_detail_z", "vis_self_distance_z", "vis_delta_z_pca"],
  diags_h: ["vis_action_alignment_detail_h", "vis_self_distance_h", "vis_delta_h_pca"],
  diags_s: ["vis_action_alignment_detail_p", "vis_self_distance_p", "vis_delta_p_pca"],
  // Action: centered delta, PCA, raw delta
  action_z: ["vis_action_alignment_detail_centered_z", "vis_action_alignment_detail_z", "vis_action_alignment_detail_raw_z"],
  action_h: ["vis_action_alignment_detail_centered_h", "vis_action_alignment_detail_h", "vis_action_alignment_detail_raw_h"],
  action_s: ["vis_action_alignment_detail_centered_p", "vis_action_alignment_detail_p", "vis_action_alignment_detail_raw_p"],
  // Self-distance: distance plots
  self_distance_z: ["vis_self_distance_z"],
  self_distance_h: ["vis_self_distance_h"],
  self_distance_s: ["vis_self_distance_p"],
  planning: [
    "vis_planning_pca_test1",
    "vis_planning_pca_test2",
    "vis_planning_exec_test1",
    "vis_planning_exec_test2",
    "vis_planning_action_stats",
    "vis_planning_action_stats_strip",
    "vis_planning_reachable_h",
    "vis_planning_reachable_p",
    "vis_planning_graph_h",
    "vis_planning_graph_p",
  ],
};

/**
 * Creates a folder selector instance that manages folder selection state
 * and syncs with UI elements.
 *
 * @param {Object} config
 * @param {string} config.menuId - ID of the dropdown menu element
 * @param {string[]} [config.initialFolders] - Initial folder selection
 * @param {Function} [config.onSelectionChange] - Callback when selection changes
 * @param {boolean} [config.updateUrl=true] - Whether to update URL on selection change
 * @returns {Object} Selector instance with getSelectedFolders, setSelectedFolders methods
 */
function createFolderSelector(config) {
  const { menuId, initialFolders, onSelectionChange, updateUrl = true } = config;

  let selectedFolders = ["vis_fixed_0"];
  let availableFolders = null;

  function isFolderAvailable(value) {
    if (!availableFolders) return true;
    return availableFolders.has(value);
  }

  function filterAvailableFolders(values) {
    const list = Array.isArray(values) ? values : [];
    if (!availableFolders) return list;
    return list.filter((value) => availableFolders.has(value));
  }

  function normalizeSelectedFolders(values) {
    const normalized = [];
    const seen = new Set();
    (values || []).forEach((value) => {
      const trimmed = value.trim();
      if (!trimmed || seen.has(trimmed)) return;
      if (!getImageOption(trimmed)) return;
      seen.add(trimmed);
      normalized.push(trimmed);
    });
    return filterAvailableFolders(normalized);
  }

  // Find matching folder group for a given folder array
  function findMatchingGroup(folders) {
    for (const [groupName, groupFolders] of Object.entries(FOLDER_GROUP_PRESETS)) {
      if (folders.length === groupFolders.length &&
          folders.every((f, i) => f === groupFolders[i])) {
        return groupName;
      }
    }
    return null;
  }

  function updateFolderUrl(folders) {
    if (!updateUrl) return;
    const url = new URL(window.location.href);

    // Check if folders match a preset group
    const matchingGroup = findMatchingGroup(folders);
    if (matchingGroup) {
      url.searchParams.set("folder_group", matchingGroup);
      url.searchParams.delete("folders");
      url.searchParams.delete("folder");
    } else {
      url.searchParams.delete("folder_group");
      const value = folders.join(",");
      if (value) {
        url.searchParams.set("folders", value);
        url.searchParams.set("folder", folders[0]);
      } else {
        url.searchParams.delete("folders");
        url.searchParams.delete("folder");
      }
    }
    window.history.replaceState({}, "", url.toString());
  }

  function syncDropdownSelection() {
    const menu = document.getElementById(menuId);
    if (!menu) return;
    const checkboxes = menu.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach((checkbox) => {
      checkbox.checked = selectedFolders.includes(checkbox.value);
    });
  }

  function updateAvailabilityUI() {
    const menu = document.getElementById(menuId);
    if (menu) {
      const items = menu.querySelectorAll("[data-value]");
      items.forEach((item) => {
        const value = item.dataset.value;
        const checkbox = item.querySelector('input[type="checkbox"]');
        const available = isFolderAvailable(value);
        if (checkbox) {
          checkbox.disabled = !available;
        }
        item.classList.toggle("disabled", !available);
        item.classList.toggle("text-muted", !available);
      });
    }

    const buttons = document.querySelectorAll("[data-folder-preset]");
    buttons.forEach((button) => {
      const preset = FOLDER_GROUP_PRESETS[button.dataset.folderPreset];
      if (!preset) return;
      const available = filterAvailableFolders(preset);
      const isDisabled = available.length === 0;
      button.disabled = isDisabled;
      button.classList.toggle("disabled", isDisabled);
      button.setAttribute("aria-disabled", isDisabled ? "true" : "false");
    });
  }

  function setSelectedFolders(nextSelected, options = {}) {
    const { triggerCallback = true, doUpdateUrl = true } = options;
    const normalized = normalizeSelectedFolders(nextSelected);
    selectedFolders = normalized.length ? normalized : (filterAvailableFolders(["vis_fixed_0"])[0] ? ["vis_fixed_0"] : []);
    if (!selectedFolders.length && availableFolders && availableFolders.size) {
      selectedFolders = [Array.from(availableFolders)[0]];
    }
    if (doUpdateUrl) {
      updateFolderUrl(selectedFolders);
    }
    syncDropdownSelection();
    if (triggerCallback && onSelectionChange) {
      onSelectionChange(selectedFolders);
    }
  }

  function getSelectedFolders() {
    return [...selectedFolders];
  }

  function mergeSelectionOrder(nextSelected) {
    const normalized = normalizeSelectedFolders(nextSelected);
    const nextSet = new Set(normalized);
    const preserved = selectedFolders.filter((value) => nextSet.has(value));
    const preservedSet = new Set(preserved);
    const added = normalized.filter((value) => !preservedSet.has(value));
    return preserved.concat(added);
  }

  function buildDropdownItems() {
    return IMAGE_FOLDER_SPECS.map((opt) => {
      const item = document.createElement("label");
      item.className = "dropdown-item d-flex align-items-center gap-2";
      item.dataset.value = opt.value;
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "form-check-input m-0";
      checkbox.value = opt.value;
      checkbox.checked = selectedFolders.includes(opt.value);
      const text = document.createElement("span");
      text.className = "small";
      text.textContent = opt.label;
      item.appendChild(checkbox);
      item.appendChild(text);
      return item;
    });
  }

  function initializeDropdown() {
    const menu = document.getElementById(menuId);
    if (!menu) return;
    menu.innerHTML = "";
    const items = buildDropdownItems();
    items.forEach((item) => menu.appendChild(item));
    menu.addEventListener("change", () => {
      const nextSelected = Array.from(
        menu.querySelectorAll('input[type="checkbox"]:checked')
      ).map((checkbox) => checkbox.value);
      const ordered = mergeSelectionOrder(nextSelected);
      setSelectedFolders(ordered);
    });
  }

  function initializePresetButtons() {
    const buttons = document.querySelectorAll("[data-folder-preset]");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const preset = FOLDER_GROUP_PRESETS[button.dataset.folderPreset];
        if (!preset) return;
        const filtered = filterAvailableFolders(preset);
        if (!filtered.length) return;
        setSelectedFolders(filtered);
      });
    });
    updateAvailabilityUI();
  }

  function restoreFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);

    // Check for folder_group first (e.g., ?folder_group=diags_z)
    const folderGroupParam = urlParams.get("folder_group");
    if (folderGroupParam && FOLDER_GROUP_PRESETS[folderGroupParam]) {
      setSelectedFolders(FOLDER_GROUP_PRESETS[folderGroupParam], { triggerCallback: false, doUpdateUrl: false });
      return;
    }

    // Fall back to explicit folders list
    const foldersParam = urlParams.get("folders");
    if (foldersParam) {
      setSelectedFolders(foldersParam.split(","), { triggerCallback: false, doUpdateUrl: false });
      return;
    }

    // Fall back to single folder
    const folderParam = urlParams.get("folder");
    if (folderParam) {
      setSelectedFolders([folderParam], { triggerCallback: false, doUpdateUrl: false });
    }
  }

  function initialize() {
    // Set initial folders if provided
    if (initialFolders && initialFolders.length) {
      selectedFolders = normalizeSelectedFolders(initialFolders);
      if (!selectedFolders.length) {
        selectedFolders = ["vis_fixed_0"];
      }
    }

    // Restore from URL (overrides initialFolders)
    restoreFromUrl();

    // Initialize UI
    initializeDropdown();
    initializePresetButtons();

    // Sync dropdown state
    syncDropdownSelection();
  }

  function setAvailableFolderValues(values, options = {}) {
    const { pruneSelection = true } = options;
    if (values === null || values === undefined) {
      availableFolders = null;
    } else if (values instanceof Set) {
      availableFolders = new Set(values);
    } else {
      availableFolders = new Set(values);
    }
    updateAvailabilityUI();
    if (pruneSelection) {
      const filtered = filterAvailableFolders(selectedFolders);
      if (filtered.length !== selectedFolders.length) {
        setSelectedFolders(filtered);
      } else {
        syncDropdownSelection();
      }
    }
  }

  // Initialize on creation
  initialize();

  return {
    getSelectedFolders,
    setSelectedFolders,
    syncDropdownSelection,
    setAvailableFolderValues,
  };
}
