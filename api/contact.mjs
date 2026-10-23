import {
  CONTACT_ORIGIN,
  failure,
  validateSubmission,
  fingerprint,
  deliverMessage,
} from "../contact-lib/contact.mjs";

// Best-effort instance-local abuse and duplicate protection. Resend also
// handles delivery idempotency; these maps are not a distributed serverless rate limit.
const attempts = new Map();
const submissions = new Map();
const WINDOW = 15 * 60 * 1000;
const MAX_BODY = 20000;
function collect(now) {
  for (const [key, value] of attempts)
    if (value.expires <= now) attempts.delete(key);
  for (const [key, value] of submissions)
    if (value.expires <= now) submissions.delete(key);
}
async function bodyOf(req) {
  if (
    !String(req.headers["content-type"] || "")
      .toLowerCase()
      .startsWith("application/json")
  )
    throw failure(415, "Please send your message through the contact form.");
  if (Number(req.headers["content-length"]) > MAX_BODY)
    throw failure(
      413,
      "That message is too long. Please shorten it and try again.",
    );
  if (req.body === null) return null;
  if (req.body && typeof req.body === "object") {
    if (Buffer.byteLength(JSON.stringify(req.body)) > MAX_BODY)
      throw failure(
        413,
        "That message is too long. Please shorten it and try again.",
      );
    return req.body;
  }
  let raw = typeof req.body === "string" ? req.body : "";
  if (!raw)
    for await (const chunk of req) {
      raw += chunk;
      if (Buffer.byteLength(raw) > MAX_BODY)
        throw failure(
          413,
          "That message is too long. Please shorten it and try again.",
        );
    }
  if (Buffer.byteLength(raw) > MAX_BODY)
    throw failure(
      413,
      "That message is too long. Please shorten it and try again.",
    );
  try {
    return JSON.parse(raw);
  } catch {
    throw failure(400, "Please check your message and try again.");
  }
}
export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("X-Robots-Tag", "noindex, nofollow");
  try {
    if (req.method !== "POST") {
      res.setHeader("Allow", "POST");
      throw failure(405, "Please use the contact form to send a message.");
    }
    if (req.headers.origin !== CONTACT_ORIGIN)
      throw failure(403, "Please send your message from rohitravi.com.");
    const now = Date.now();
    collect(now);
    const ip = String(
      req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "unknown",
    )
      .split(",")[0]
      .trim();
    const bucket = attempts.get(ip) || { count: 0, expires: now + WINDOW };
    if (bucket.count >= 8 || (attempts.size >= 5000 && !attempts.has(ip))) {
      res.setHeader("Retry-After", "900");
      throw failure(
        429,
        "A few messages have been sent recently. Please wait a little or email me directly.",
      );
    }
    attempts.set(ip, { ...bucket, count: bucket.count + 1 });
    const submission = validateSubmission(await bodyOf(req));
    const hash = fingerprint(submission),
      existing = submissions.get(submission.submissionId);
    if (existing && existing.hash !== hash)
      throw failure(
        409,
        "This message changed while sending. Please try again.",
      );
    if (existing) return res.status(200).json(await existing.promise);
    if (submissions.size >= 1000)
      throw failure(429, "Please try again shortly, or email me directly.");
    const promise = deliverMessage(submission);
    submissions.set(submission.submissionId, {
      hash,
      expires: now + WINDOW,
      promise,
    });
    try {
      return res.status(200).json(await promise);
    } catch (error) {
      submissions.delete(submission.submissionId);
      throw error;
    }
  } catch (error) {
    return res
      .status(error.status || 500)
      .json({
        error: error.status
          ? error.message
          : "I couldn’t submit your message. Your message is still here—please try again or email me directly.",
      });
  }
}
