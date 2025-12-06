document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll(".notes-form");
  forms.forEach((form) => wireNotesForm(form));
  wireTitleForms(document);
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
