document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll(".notes-form");
  forms.forEach((form) => wireNotesForm(form));
  wireTitleForms(document);
  wireMetadataToggles(document);
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
