import { createHash } from "node:crypto";

export const CONTACT_ORIGIN = "https://www.rohitravi.com";
export const CONTACT_EMAIL = "rohit.ravi2@gmail.com";
export const MAX_MESSAGE_LENGTH = 5000;
export const TOPICS = Object.freeze({
  collaboration: "Collaboration",
  research: "Research opportunity",
  other: "Something else",
  "": "General message",
});
export function failure(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}
export function validateSubmission(body) {
  if (!body || typeof body !== "object" || Array.isArray(body))
    throw failure(400, "Please check your message and try again.");
  if (body.website !== undefined && typeof body.website !== "string")
    throw failure(400, "Please check your message and try again.");
  if (typeof body.website === "string" && body.website.trim())
    throw failure(400, "Please try again, or use the email link below.");
  const name = typeof body.name === "string" ? body.name.trim() : "";
  const email = typeof body.email === "string" ? body.email.trim() : "";
  const message = typeof body.message === "string" ? body.message.trim() : "";
  const topic = typeof body.topic === "string" ? body.topic : "";
  if (name.length > 100 || /[\r\n\x00-\x1f]/.test(name))
    throw failure(400, "Please use a name under 100 characters.");
  if (
    email.length > 254 ||
    !/^[^\s@<>\x00-\x1f]+@[^\s@<>\x00-\x1f]+\.[^\s@<>\x00-\x1f]+$/.test(email)
  )
    throw failure(400, "Please enter a valid reply email address.");
  if (
    message.length < 10 ||
    message.length > MAX_MESSAGE_LENGTH ||
    message.includes("\0")
  )
    throw failure(
      400,
      "Please write a message between 10 and 5,000 characters.",
    );
  if (!Object.hasOwn(TOPICS, topic))
    throw failure(400, "Please choose a listed topic, or leave it unselected.");
  if (
    typeof body.submissionId !== "string" ||
    !/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(
      body.submissionId,
    )
  )
    throw failure(
      400,
      "Please refresh the page before sending. Your draft will be kept.",
    );
  return { name, email, message, topic, submissionId: body.submissionId };
}
export function fingerprint(submission) {
  return createHash("sha256")
    .update(
      JSON.stringify([
        submission.name,
        submission.email,
        submission.message,
        submission.topic,
      ]),
    )
    .digest("hex");
}
export function deliveryPayload(submission) {
  const domain = process.env.RESEND_EMAIL_DOMAIN;
  const defaultSender =
    typeof domain === "string" &&
    /^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$/.test(domain)
      ? `Rohit Ravi website <contact@${domain}>`
      : "Rohit Ravi website <onboarding@resend.dev>";
  return {
    from: process.env.CONTACT_FROM_EMAIL || defaultSender,
    to: [CONTACT_EMAIL],
    reply_to: submission.email,
    subject: `[rohitravi.com] ${TOPICS[submission.topic]}`,
    text: [
      `Name: ${submission.name || "Not provided"}`,
      `Reply address: ${submission.email}`,
      `Topic: ${TOPICS[submission.topic]}`,
      `Reference: ${submission.submissionId}`,
      "",
      submission.message,
      "",
      `Sent through ${CONTACT_ORIGIN}/#contact`,
    ].join("\n"),
  };
}
export async function deliverMessage(submission, fetcher = fetch) {
  const key = process.env.RESEND_API_KEY;
  if (!key)
    throw failure(
      503,
      "Message delivery isn’t available yet. Your message is still here; please email me directly.",
    );
  let response;
  try {
    response = await fetcher("https://api.resend.com/emails", {
      method: "POST",
      signal: AbortSignal.timeout(15000),
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
        "Idempotency-Key": `contact/${submission.submissionId}`,
      },
      body: JSON.stringify(deliveryPayload(submission)),
    });
  } catch {
    throw failure(
      502,
      "I couldn’t confirm submission. Your message is still here—please try again or email me directly.",
    );
  }
  let result;
  try {
    result = await response.json();
  } catch {
    throw failure(
      502,
      "I couldn’t submit your message. Your message is still here—please try again or email me directly.",
    );
  }
  if (
    !response.ok ||
    typeof result?.id !== "string" ||
    !result.id ||
    result.error ||
    result.name === "validation_error"
  ) {
    throw failure(
      502,
      "I couldn’t submit your message. Your message is still here—please try again or email me directly.",
    );
  }
  return { accepted: true, reference: submission.submissionId };
}
