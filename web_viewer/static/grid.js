document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll(".notes-form");
  forms.forEach((form) => wireNotesForm(form));
  wireTitleForms(document);
  wireTagsForms(document);
  wireMetadataToggles(document);
  formatAllNumbers(document);
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
