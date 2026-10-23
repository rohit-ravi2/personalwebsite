# Website contact delivery

The homepage contact panel posts to `/api/contact`. Messages go only to `rohit.ravi2@gmail.com`, with the visitor’s validated email as Reply-To and a topic-based subject. The endpoint uses Resend’s authenticated email API. The interface only shows its receipt when the API accepts the message; rejection, timeout, or missing configuration keeps the draft and offers direct email.

## Current configuration

The free `personalwebsite-contact` integration is connected to the existing production Vercel project. `rohitravi.com` is verified for sending. Its private key and domain configuration are stored in Vercel, not this repository.

## Activate sending

1. Connect Resend to the existing Vercel `personalwebsite` project. The [Vercel Marketplace integration](https://vercel.com/marketplace/resend) can provision the account and add `RESEND_API_KEY` to the project. Select a free option; no paid plan is required or authorized by this implementation.
2. For a separately created Resend account, use `rohit.ravi2@gmail.com` and add a sending-only API key as the private Vercel production variable `RESEND_API_KEY`.
3. The integration supplies `RESEND_EMAIL_DOMAIN`; the endpoint uses `Rohit Ravi website <contact@rohitravi.com>` when that domain is configured. Optionally override it with private `CONTACT_FROM_EMAIL`. If neither variable supplies a domain sender, the fallback is Resend’s `onboarding@resend.dev`, restricted to the Resend account owner’s inbox; that fallback requires the account’s email to match `rohit.ravi2@gmail.com`.
4. Redeploy after setting environment variables, submit one test message, and confirm it reaches the inbox with the correct Reply-To. A provider acceptance alone does not prove inbox delivery.

The native integration can automatically configure DNS for domains purchased through Vercel. Do not transfer the website domain or change MX records to activate this contact form. If the domain is managed elsewhere, use the owner-only default sender initially or add the verification records separately.

Private credentials must stay in Vercel settings or ignored local environment files, never browser code or Git. Publishing source through the existing GitHub integration works independently of Vercel CLI account access.

## Interaction and privacy

- Name and topic are optional; email and a 10–5,000-character message are required.
- Topic selection supplies a relevant prompt without adding required fields.
- A draft is stored in sessionStorage for the current tab, expires after 24 hours, and is cleared after a successful submission or “Clear draft.” Refreshing or client-side navigation restores it while that tab’s storage remains available.
- Sending locks the fields and submit action. Failure preserves the draft; success shows the reply address and a reference.
- No auto-reply is sent to the visitor. “Send another message” preserves their name and email.
- Messages are sent as plain-text email through Resend. The site does not log message bodies or store them in its public repository.

## Abuse and retry handling

The endpoint checks the canonical request Origin, content type, input bounds, email/header-injection attempts, allowed topics, and a hidden honeypot. Instance-local rate limits cap attempts; these are not a distributed firewall. Add platform-level rate limiting if traffic warrants it.

Each draft has an idempotency reference. Concurrent submissions share one in-flight result within an instance. Resend’s `Idempotency-Key` protects successful retries across serverless instances; changing message content creates a new reference. No automatic delivery retry loop is used.

Run `pnpm test` for endpoint and validation checks. Browser tests should cover valid and invalid input, optional topics, refresh restoration, failure recovery, duplicate submission, confirmation focus, and another-message flow at desktop and mobile sizes. Mocked delivery tests do not confirm real mail delivery.

## Provider selection

A preliminary FormSubmit relay request was blocked by its browser security challenge. That relay is not used in the implementation, and no delivery was confirmed from that check.

References: [Resend email API](https://resend.com/docs/api-reference/emails/send-email), [Resend idempotency](https://resend.com/changelog/idempotency-keys), [Resend–Vercel setup](https://resend.com/changelog/vercel-marketplace-integration).
