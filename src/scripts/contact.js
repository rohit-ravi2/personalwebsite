const DRAFT_KEY = "rohitravi.contact.v1";
const TOPIC_LABELS = {
  collaboration: "Collaboration",
  research: "Research opportunity",
  other: "Something else",
  "": "General message",
};
const PROMPTS = {
  collaboration: "What are you exploring, and where might we work together?",
  research:
    "Tell me about the team, opportunity, or research question you have in mind.",
  other: "What would you like to talk about?",
  "": "An idea, a question, or a brief introduction is a good place to start.",
};
let activeContact;
function initializeContact() {
  activeContact?.abort();
  activeContact = new AbortController();
  const { signal } = activeContact;
  const root = document.querySelector(".contact-panel");
  if (!root) return;
  const find = (selector) => root.querySelector(selector);
  const form = find("[data-contact-form]");
  const fields = find("[data-contact-fields]");
  const email = find("#contact-email"),
    name = find("#contact-name"),
    message = find("#contact-message");
  const send = find("[data-contact-send]"),
    error = find("[data-contact-error]");
  const receipt = find("[data-contact-receipt]");
  let busy = false,
    submissionId = "",
    lastPayload = "";
  const selectedTopic = () =>
    form.querySelector('input[name="topic"]:checked')?.value || "";
  const draft = () => ({
    name: name.value,
    email: email.value,
    message: message.value,
    topic: selectedTopic(),
    submissionId,
    lastPayload,
    updated: Date.now(),
  });
  const clearStorage = () => {
    try {
      sessionStorage.removeItem(DRAFT_KEY);
    } catch {}
  };
  const hasDraft = () =>
    Boolean(name.value || email.value || message.value || selectedTopic());
  function updateFields(save = true) {
    const topic = selectedTopic();
    find("[data-contact-prompt]").textContent = PROMPTS[topic] || PROMPTS[""];
    message.placeholder = topic ? PROMPTS[topic] : "";
    find("[data-contact-count]").textContent =
      `${message.value.length.toLocaleString()} / 5,000`;
    find("[data-contact-clear]").hidden = !hasDraft();
    let saved = false;
    if (save) {
      try {
        if (hasDraft()) {
          sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft()));
          saved = true;
        } else clearStorage();
      } catch {}
    }
    find("[data-contact-draft]").textContent =
      hasDraft() && saved ? "Draft saved in this tab." : "";
  }
  try {
    const saved = JSON.parse(sessionStorage.getItem(DRAFT_KEY) || "null");
    if (saved && saved.updated > Date.now() - 24 * 60 * 60 * 1000) {
      if (typeof saved.name === "string") name.value = saved.name.slice(0, 100);
      if (typeof saved.email === "string")
        email.value = saved.email.slice(0, 254);
      if (typeof saved.message === "string")
        message.value = saved.message.slice(0, 5000);
      if (Object.hasOwn(TOPIC_LABELS, saved.topic)) {
        const radio = [...form.querySelectorAll('input[name="topic"]')].find(
          (input) => input.value === saved.topic,
        );
        if (radio) radio.checked = true;
      }
      if (typeof saved.submissionId === "string")
        submissionId = saved.submissionId;
      if (typeof saved.lastPayload === "string")
        lastPayload = saved.lastPayload;
    }
  } catch {
    clearStorage();
  }
  updateFields();
  send.disabled = false;
  for (const input of form.querySelectorAll("input, textarea")) {
    input.addEventListener(
      "input",
      () => {
        input.setCustomValidity("");
        input.removeAttribute("aria-invalid");
        error.hidden = true;
        updateFields();
      },
      { signal },
    );
    if (input.type === "radio")
      input.addEventListener("change", () => updateFields(), { signal });
  }
  form.addEventListener(
    "invalid",
    (event) => {
      event.target.setAttribute("aria-invalid", "true");
    },
    { capture: true, signal },
  );
  function reset(clearIdentity = true) {
    const identity = { name: name.value, email: email.value };
    form.reset();
    submissionId = "";
    lastPayload = "";
    error.hidden = true;
    clearStorage();
    if (!clearIdentity) {
      name.value = identity.name;
      email.value = identity.email;
    }
    updateFields();
  }
  find("[data-contact-clear]").addEventListener(
    "click",
    () => {
      reset();
      email.focus();
    },
    { signal },
  );
  find("[data-contact-again]").addEventListener(
    "click",
    () => {
      reset(false);
      receipt.hidden = true;
      form.hidden = false;
      message.focus();
    },
    { signal },
  );
  form.addEventListener(
    "submit",
    async (event) => {
      event.preventDefault();
      if (busy) return;
      email.value = email.value.trim();
      message.setCustomValidity(
        message.value.trim().length < 10
          ? "Please add a little more detail—at least 10 characters."
          : "",
      );
      if (!form.reportValidity()) return;
      const payload = {
        name: name.value.trim(),
        email: email.value,
        message: message.value.trim(),
        topic: selectedTopic(),
      };
      const signature = JSON.stringify(payload);
      if (signature !== lastPayload || !/^[a-f0-9-]{36}$/i.test(submissionId))
        submissionId = crypto.randomUUID();
      lastPayload = signature;
      updateFields();
      busy = true;
      fields.disabled = true;
      error.hidden = true;
      form.setAttribute("aria-busy", "true");
      find("[data-contact-send-label]").textContent = "Submitting…";
      find("[data-contact-arrow]").hidden = true;
      try {
        const response = await fetch("/api/contact", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            submissionId,
            website: form.elements.namedItem("website").value,
          }),
          signal: AbortSignal.timeout(22000),
        });
        let result;
        try {
          result = await response.json();
        } catch {
          throw Object.assign(
            new Error(
              "I couldn’t confirm submission. Your message is still here—please try again or email me directly.",
            ),
            { contactError: true },
          );
        }
        if (!response.ok || result.accepted !== true)
          throw Object.assign(
            new Error(
              result.error ||
                "I couldn’t submit your message. Please try again or email me directly.",
            ),
            { contactError: true },
          );
        clearStorage();
        find("[data-contact-thanks]").textContent = payload.name
          ? `Thanks, ${payload.name.split(/\s+/)[0]}.`
          : "Thanks for reaching out.";
        find("[data-contact-reply]").textContent = payload.email;
        find("[data-contact-topic]").textContent = TOPIC_LABELS[payload.topic];
        find("[data-contact-reference]").textContent = String(
          result.reference || submissionId,
        )
          .slice(0, 8)
          .toUpperCase();
        form.hidden = true;
        receipt.hidden = false;
        find("[data-contact-thanks]").focus();
      } catch (caught) {
        updateFields();
        error.textContent =
          caught.name === "TimeoutError" || caught.name === "AbortError"
            ? "I couldn’t confirm submission. Your message is still here—please try again or email me directly."
            : caught.contactError
              ? caught.message
              : "I couldn’t reach the contact service. Your message is still here—please try again or email me directly.";
        error.hidden = false;
      } finally {
        busy = false;
        fields.disabled = false;
        form.removeAttribute("aria-busy");
        find("[data-contact-send-label]").textContent = "Send message";
        find("[data-contact-arrow]").hidden = false;
      }
    },
    { signal },
  );
}
document.addEventListener("astro:page-load", initializeContact);
