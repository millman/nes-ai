document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll(".notes-form");
  forms.forEach((form) => wireNotesForm(form));
  wireTitleForms(document);
  setupColumnResizing();
  setupRowHeightSync();
});

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

function wireTitleForm(form) {
  const expId = form.dataset.expId;
  const input = form.querySelector(".exp-title-input");
  const status = form.querySelector(".title-status");
  if (!input) {
    return;
  }
  let lastValue = input.value;
  const updateState = () => {
    if (input.value !== lastValue) {
      status.textContent = "Unsaved";
    } else {
      status.textContent = "";
    }
  };
  const save = () => {
    if (input.value === lastValue) {
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
      })
      .catch(() => {
        status.textContent = "Save failed";
      });
  };
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    save();
  });
  input.addEventListener("input", updateState);
  input.addEventListener("blur", save);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      save();
    }
  });
  updateState();
}

function setupColumnResizing() {
  const table = document.querySelector(".exp-table");
  if (!table) {
    return;
  }
  const headers = table.querySelectorAll("thead th");
  headers.forEach((header, index) => {
    const colId = header.dataset.colId;
    if (!colId) {
      return;
    }
    const cssVar = `--col-${colId}-width`;
    const handle = document.createElement("div");
    handle.className = "resize-handle";
    header.appendChild(handle);
    handle.addEventListener("mousedown", (event) => {
      event.preventDefault();
      const startX = event.clientX;
      const startWidth = header.offsetWidth;
      handle.classList.add("is-resizing");
      const onMouseMove = (moveEvent) => {
        const delta = moveEvent.clientX - startX;
        const newWidth = Math.max(120, startWidth + delta);
        document.documentElement.style.setProperty(cssVar, `${newWidth}px`);
      };
      const onMouseUp = () => {
        handle.classList.remove("is-resizing");
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
        syncRowHeights();
      };
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    });
  });
}

function setupRowHeightSync() {
  const sync = () => syncRowHeights();
  window.addEventListener("resize", sync);
  const images = document.querySelectorAll(".image-cell img");
  images.forEach((img) => {
    if (img.complete) {
      sync();
    } else {
      img.addEventListener("load", sync);
    }
  });
  window.addEventListener("load", sync);
  requestAnimationFrame(sync);
  setTimeout(sync, 0);
}

function syncRowHeights() {
  const rows = document.querySelectorAll(".exp-table tbody tr");
  rows.forEach((row) => {
    const image = row.querySelector(".image-cell img");
    const targetHeight = image ? image.clientHeight : null;
    if (!targetHeight) {
      return;
    }
    row.style.height = `${targetHeight}px`;
    const metadata = row.querySelector(".metadata-block");
    if (metadata) {
      metadata.style.height = `${targetHeight}px`;
    }
    const form = row.querySelector(".notes-form");
    if (form) {
      form.style.height = `${targetHeight}px`;
      const textarea = form.querySelector("textarea");
      if (textarea) {
        const actions = form.querySelector(".notes-actions");
        const reserve = actions ? actions.offsetHeight + 12 : 12;
        const newHeight = Math.max(60, targetHeight - reserve);
        textarea.style.height = `${newHeight}px`;
      }
    }
  });
}
