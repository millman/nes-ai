document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll(".notes-form");
  forms.forEach((form) => wireNotesForm(form));
  wireTitleForms(document);
  wireTagsForms(document);
  wireMetadataToggles(document);
  wireMetadataPopovers(document);
  formatAllNumbers(document);
  wireDashboardNotes(document);
});

/**
 * Format all elements with .format-params and .format-flops classes.
 * @param {Element} root - The root element to search within.
 */
function formatAllNumbers(root) {
  root.querySelectorAll(".format-params").forEach((el) => {
    const value = el.dataset.value;
    el.textContent = formatParamCount(value ? parseInt(value, 10) : null);
  });
  root.querySelectorAll(".format-flops").forEach((el) => {
    const value = el.dataset.value;
    el.textContent = formatFlops(value ? parseInt(value, 10) : null);
  });
}

/**
 * Format a parameter count in human-readable form (e.g., "4.26M").
 * @param {number|null} count - The raw parameter count.
 * @returns {string} - Formatted string or "—" if null/undefined.
 */
function formatParamCount(count) {
  if (count === null || count === undefined) {
    return "—";
  }
  if (count < 1000) {
    return String(count);
  }
  const units = [
    { divisor: 1_000_000_000, suffix: "B" },
    { divisor: 1_000_000, suffix: "M" },
    { divisor: 1_000, suffix: "k" },
  ];
  for (const { divisor, suffix } of units) {
    if (count >= divisor) {
      const value = count / divisor;
      let formatted;
      if (value >= 100) {
        formatted = value.toFixed(0);
      } else if (value >= 10) {
        formatted = value.toFixed(1);
      } else {
        formatted = value.toFixed(2);
      }
      // Remove trailing zeros and decimal point
      formatted = formatted.replace(/\.?0+$/, "");
      return formatted + suffix;
    }
  }
  return String(count);
}

/**
 * Format a FLOPs count in human-readable form (e.g., "229 GFLOPs").
 * @param {number|null} flops - The raw FLOPs count.
 * @returns {string} - Formatted string or "—" if null/undefined.
 */
function formatFlops(flops) {
  if (flops === null || flops === undefined) {
    return "—";
  }
  const units = [
    { divisor: 1_000_000_000_000, suffix: "TFLOPs" },
    { divisor: 1_000_000_000, suffix: "GFLOPs" },
    { divisor: 1_000_000, suffix: "MFLOPs" },
    { divisor: 1_000, suffix: "KFLOPs" },
  ];
  for (const { divisor, suffix } of units) {
    if (flops >= divisor) {
      const value = flops / divisor;
      let formatted;
      if (value >= 100) {
        formatted = value.toFixed(0);
      } else if (value >= 10) {
        formatted = value.toFixed(1);
      } else {
        formatted = value.toFixed(2);
      }
      // Remove trailing zeros and decimal point
      formatted = formatted.replace(/\.?0+$/, "");
      return formatted + " " + suffix;
    }
  }
  return flops + " FLOPs";
}

/**
 * Safely escape HTML entities in a string.
 * @param {string} value - Raw string to escape.
 * @returns {string} - Escaped string safe for HTML.
 */
function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wireNotesForm(form) {
  const expId = form.dataset.expId;
  const textarea = form.querySelector("textarea");
  const button = form.querySelector(".save-notes");
  const status = form.querySelector(".notes-status");
  let lastSavedValue = textarea.value;

  function markDirty() {
    const dirty = textarea.value !== lastSavedValue;
    button.disabled = !dirty;
    status.textContent = dirty ? "Unsaved changes" : "";
  }

  function saveNotes() {
    button.disabled = true;
    status.textContent = "Saving…";
    fetch(`/experiments/${expId}/notes`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ notes: textarea.value }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to save notes");
        }
        return response.json();
      })
      .then(() => {
        lastSavedValue = textarea.value;
        status.textContent = "Saved";
      })
      .catch(() => {
        status.textContent = "Save failed";
        button.disabled = false;
      });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    saveNotes();
  });

  textarea.addEventListener("input", markDirty);
  textarea.addEventListener("blur", () => {
    if (textarea.value === lastSavedValue) {
      return;
    }
    saveNotes();
  });
  button.addEventListener("click", (event) => {
    event.preventDefault();
    saveNotes();
  });
  markDirty();
}

function wireMetadataPopovers(root) {
  const buttons = root.querySelectorAll(".metadata-icon-btn, .git-icon-btn");
  if (!buttons.length || typeof bootstrap === "undefined") {
    return;
  }

  buttons.forEach((button) => {
    // Dispose any existing popover
    const existing = bootstrap.Popover.getInstance(button);
    if (existing) existing.dispose();

    // Use focus trigger - click focuses the button, clicking elsewhere blurs it
    new bootstrap.Popover(button, {
      trigger: "hover focus",
      html: true,
      sanitize: false,
      animation: false,
      container: "body",
      placement: "auto",
      fallbackPlacements: ["bottom", "top", "right", "left"],
      customClass: "metadata-popover",
      content: () => {
        const text = button.dataset.metadata || "";
        return `<pre class="metadata-popover-pre">${escapeHtml(text)}</pre>`;
      },
    });

    // Make button focusable
    button.setAttribute("tabindex", "0");

    // Toggle focus on click - if already focused, blur to close
    button.addEventListener("mousedown", (event) => {
      if (document.activeElement === button) {
        event.preventDefault(); // Prevent the click from re-focusing
        button.blur();
      }
    });
  });
}

function wireDashboardNotes(root) {
  const buttons = root.querySelectorAll(".notes-icon-btn");
  if (!buttons.length || typeof bootstrap === "undefined") {
    return;
  }

  let activePopover = null;

  const renderNotesHtml = (value) => {
    if (!value || !value.trim()) {
      return '<span class="text-white-50">No notes yet</span>';
    }
    return escapeHtml(value).replace(/\n/g, "<br>");
  };

  const updateIconState = (buttonEl, textValue) => {
    const icon = buttonEl.querySelector(".notes-icon");
    const hasNotes = Boolean(textValue && textValue.trim());
    icon?.classList.toggle("has-notes", hasNotes);
    icon?.classList.toggle("no-notes", !hasNotes);
  };

  const closeActivePopover = (event) => {
    if (!activePopover) {
      return;
    }
    if (event && activePopover.tip && activePopover.tip.contains(event.target)) {
      return;
    }
    activePopover.popover.hide();
    if (activePopover.onClose) {
      activePopover.onClose();
    }
    activePopover = null;
  };

  document.addEventListener("click", (event) => {
    if (event.target.closest(".notes-icon-btn")) {
      return;
    }
    // Don't close if clicking inside the popover itself
    if (event.target.closest(".notes-popover")) {
      return;
    }
    closeActivePopover(event);
  });

  buttons.forEach((button) => {
    const expId = button.dataset.expId;
    let notes = button.dataset.notes || "";
    let popoverOpen = false;

    const iconEl = button.querySelector(".notes-icon");
    // Dispose any existing tooltip to avoid "more than one instance" error
    if (iconEl) {
      const existingTooltip = bootstrap.Tooltip.getInstance(iconEl);
      if (existingTooltip) existingTooltip.dispose();
    }
    const tooltip =
      iconEl &&
      new bootstrap.Tooltip(iconEl, {
        trigger: "manual",
        html: true,
        animation: false,
        container: "body",
        placement: "right",
        offset: [0, 8],
        title: () => renderNotesHtml(notes),
        customClass: "notes-tooltip",
      });

    const showTooltip = () => {
      if (!tooltip || popoverOpen) return;
      tooltip.setContent({ ".tooltip-inner": renderNotesHtml(notes) });
      tooltip.show();
    };

    const hideTooltip = () => {
      tooltip?.hide();
    };

    const saveNotes = (textarea, statusEl, saveBtn, onSuccess, onError) => {
      saveBtn.disabled = true;
      statusEl.textContent = "Saving…";
      fetch(`/experiments/${expId}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notes: textarea.value }),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Failed to save notes");
          }
          return response.json();
        })
        .then(() => {
          notes = textarea.value;
          button.dataset.notes = notes;
          statusEl.textContent = "Saved";
          updateIconState(button, notes);
          tooltip.setContent({ ".tooltip-inner": renderNotesHtml(notes) });
          saveBtn.disabled = false;
          if (onSuccess) onSuccess();
        })
        .catch(() => {
          statusEl.textContent = "Save failed";
          saveBtn.disabled = false;
          if (onError) onError();
        });
    };

    // Dispose any existing popover to avoid "more than one instance" error
    const existingPopover = bootstrap.Popover.getInstance(button);
    if (existingPopover) existingPopover.dispose();

    const popover = new bootstrap.Popover(button, {
      trigger: "manual",
      html: true,
      sanitize: false,
      animation: false,
      container: "body",
      placement: "auto",
      fallbackPlacements: ["bottom", "top", "right", "left"],
      title: "Notes",
      template:
        '<div class="popover notes-popover" role="tooltip">' +
        '<div class="popover-arrow"></div>' +
        '<h3 class="popover-header"></h3>' +
        '<div class="popover-body"></div>' +
        "</div>",
      content:
        '<div class="notes-popover-body">' +
        '<textarea class="form-control form-control-sm notes-popover-text" rows="6" placeholder="Add experiment notes..."></textarea>' +
        '<div class="d-flex justify-content-between align-items-center gap-2 mt-2">' +
        '<span class="notes-popover-status small text-muted" aria-live="polite"></span>' +
        '<div class="btn-group btn-group-sm">' +
        '<button type="button" class="btn btn-outline-secondary notes-popover-close">Close</button>' +
        '<button type="button" class="btn btn-primary notes-popover-save">Save</button>' +
        "</div></div></div>",
    });

    updateIconState(button, notes);

    button.addEventListener("mouseenter", showTooltip);
    button.addEventListener("mouseleave", hideTooltip);
    button.addEventListener("focus", showTooltip);
    button.addEventListener("blur", hideTooltip);

    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      hideTooltip();

      // If this button's popover is already active, do nothing (it's already open)
      if (activePopover && activePopover.button === button) {
        const { textarea, statusEl, saveBtn } = activePopover;
        if (textarea && saveBtn && statusEl) {
          saveNotes(textarea, statusEl, saveBtn, () => closeActivePopover());
        } else {
          closeActivePopover();
        }
        return;
      }

      if (activePopover && activePopover.button !== button) {
        closeActivePopover();
      }

      popoverOpen = true;

      // Use one-time shown event to set up popover content after it's rendered
      const onShown = () => {
        button.removeEventListener("shown.bs.popover", onShown);
        const tip = document.querySelector(".notes-popover");
        if (!tip) {
          popoverOpen = false;
          return;
        }

        const textarea = tip.querySelector(".notes-popover-text");
        const saveBtn = tip.querySelector(".notes-popover-save");
        const closeBtn = tip.querySelector(".notes-popover-close");
        const statusEl = tip.querySelector(".notes-popover-status");

        if (textarea) textarea.value = notes;
        if (statusEl) statusEl.textContent = "";

        const markDirty = () => {
          const dirty = textarea && textarea.value !== notes;
          if (statusEl) statusEl.textContent = dirty ? "Unsaved changes" : "";
        };

        if (textarea) textarea.oninput = markDirty;
        if (saveBtn) {
          saveBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            saveNotes(textarea, statusEl, saveBtn);
          };
        }
        if (closeBtn) {
          closeBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            closeActivePopover();
          };
        }

        markDirty();
        activePopover = {
          popover,
          button,
          tip,
          textarea,
          saveBtn,
          statusEl,
          onClose: () => {
            popoverOpen = false;
          },
        };

        // Prevent focusin from bubbling and potentially triggering unwanted handlers
        tip.addEventListener("focusin", (e) => e.stopPropagation());
        tip.addEventListener("focusout", (e) => e.stopPropagation());

        // Delay focus to avoid race conditions with event handlers
        requestAnimationFrame(() => {
          if (textarea) textarea.focus();
        });
      };

      button.addEventListener("shown.bs.popover", onShown);
      popover.show();
    });
  });

  window.addEventListener(
    "scroll",
    (event) => {
      // Don't close if scrolling inside the popover (e.g., textarea scroll on paste)
      if (activePopover && activePopover.tip && activePopover.tip.contains(event.target)) {
        return;
      }
      if (activePopover) {
        closeActivePopover();
      }
    },
    true
  );
}

function wireTitleForms(root) {
  const titleForms = root.querySelectorAll(".title-form");
  titleForms.forEach((form) => wireTitleForm(form));
}

function wireTagsForms(root) {
  const tagForms = root.querySelectorAll(".tags-form");
  tagForms.forEach((form) => wireTagsForm(form));
}

function wireTitleForm(form) {
  const expId = form.dataset.expId;
  const input = form.querySelector(".exp-title-input");
  const status = form.querySelector(".title-status");
  const group = form.querySelector(".title-input-group");
  if (!input) {
    return;
  }
  input.readOnly = true;
  let lastValue = input.value;
  let editing = false;
  const enterEdit = () => {
    if (editing) {
      return;
    }
    editing = true;
    input.readOnly = false;
    form.classList.add("editing");
    input.focus();
    input.select();
  };
  const exitEdit = () => {
    editing = false;
    input.readOnly = true;
    form.classList.remove("editing");
  };
  const updateState = () => {
    if (input.value !== lastValue) {
      status.textContent = "Unsaved";
    } else {
      status.textContent = "";
    }
  };
  const save = () => {
    if (!editing) {
      return;
    }
    if (input.value === lastValue) {
      exitEdit();
      status.textContent = "";
      return;
    }
    status.textContent = "Saving…";
    fetch(`/experiments/${expId}/title`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: input.value }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to save title");
        }
        return response.json();
      })
      .then(() => {
        lastValue = input.value;
        status.textContent = "Saved";
        exitEdit();
      })
      .catch(() => {
        status.textContent = "Save failed";
        input.readOnly = false;
        form.classList.add("editing");
        editing = true;
      });
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    save();
  });
  input.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!editing) {
      enterEdit();
    }
  });
  group?.addEventListener("keydown", (event) => {
    if (!editing && event.key === "Enter") {
      event.preventDefault();
      enterEdit();
    }
  });
  input.addEventListener("input", updateState);
  input.addEventListener("blur", () => {
    if (editing) {
      save();
    }
  });
  input.addEventListener("keydown", (event) => {
    if (!editing) {
      if (event.key === "Enter") {
        event.preventDefault();
        enterEdit();
      }
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      save();
    } else if (event.key === "Escape") {
      event.preventDefault();
      input.value = lastValue;
      exitEdit();
      status.textContent = "";
    }
  });
  updateState();
}

function wireTagsForm(form) {
  const expId = form.dataset.expId;
  const input = form.querySelector(".exp-tags-input");
  const status = form.querySelector(".tags-status");
  const group = form.querySelector(".tags-input-group");
  if (!input) {
    return;
  }
  input.readOnly = true;
  let lastValue = input.value;
  let editing = false;
  const enterEdit = () => {
    if (editing) {
      return;
    }
    editing = true;
    input.readOnly = false;
    form.classList.add("editing");
    input.focus();
    input.select();
  };
  const exitEdit = () => {
    editing = false;
    input.readOnly = true;
    form.classList.remove("editing");
  };
  const updateState = () => {
    if (input.value !== lastValue) {
      status.textContent = "Unsaved";
    } else {
      status.textContent = "";
    }
  };
  const save = () => {
    if (!editing) {
      return;
    }
    if (input.value === lastValue) {
      exitEdit();
      status.textContent = "";
      return;
    }
    status.textContent = "Saving…";
    fetch(`/experiments/${expId}/tags`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tags: input.value }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to save tags");
        }
        return response.json();
      })
      .then(() => {
        lastValue = input.value;
        status.textContent = "Saved";
        exitEdit();
      })
      .catch(() => {
        status.textContent = "Save failed";
        input.readOnly = false;
        form.classList.add("editing");
        editing = true;
      });
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    save();
  });
  input.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!editing) {
      enterEdit();
    }
  });
  group?.addEventListener("keydown", (event) => {
    if (!editing && event.key === "Enter") {
      event.preventDefault();
      enterEdit();
    }
  });
  input.addEventListener("input", updateState);
  input.addEventListener("blur", () => {
    if (editing) {
      save();
    }
  });
  input.addEventListener("keydown", (event) => {
    if (!editing) {
      if (event.key === "Enter") {
        event.preventDefault();
        enterEdit();
      }
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      save();
    } else if (event.key === "Escape") {
      event.preventDefault();
      input.value = lastValue;
      exitEdit();
      status.textContent = "";
    }
  });
  updateState();
}

function wireMetadataToggles(root) {
  const buttons = root.querySelectorAll(".metadata-toggle");
  buttons.forEach((button) => {
    const targetId = button.dataset.target;
    if (!targetId) {
      return;
    }
    const wrapper = root.querySelector(`.metadata-wrapper[data-exp-id="${targetId}"]`);
    if (!wrapper) {
      return;
    }
    button.addEventListener("click", () => {
      const expanded = wrapper.classList.toggle("expanded");
      button.textContent = expanded ? "Collapse" : "Expand";
    });
  });
}
