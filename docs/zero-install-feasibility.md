# "Type a domain and it works" — what is actually buildable

**Researched 13 Nov 2026 21:20–21:25 CET, during the build freeze.** Felix's product instinct:
*"jen zadám doménu, připojím doménu a mám to, a všechno ostatní si to scrapuje pomocí Apify,
Exa, čehokoliv."* This file answers whether that is buildable, with what, and what it costs.

**Short answer: the "type a domain" half is buildable but thinner than it sounds, and the
"connect a domain" half cannot be one DNS record. Nothing here kills the idea; it sets the
promise we are allowed to make.**

---

## 1. The scraping providers, tested against the actual requirement

The requirement: given a domain, return posts/pages that **link to or discuss** it, each with
**author** and **timestamp**.

| Provider | Verdict | Why |
|---|---|---|
| **Apify** | ❌ not for this | **No actor takes a domain and returns all mentions across platforms.** Comment scrapers are per-platform (Instagram, Reddit, Trustpilot). Backlink actors (`meridianlabs/backlink-checker`, `radeance/backlinks-scraper`) find pages linking to a domain but are SEO tools — per-link author and timestamp unconfirmed. Brand monitors (`quarry/brand-monitor`, ~$8/1000 mentions) search by **keyword, not domain**. |
| **Exa** | ⚠️ partly, and not the part we wanted | Returns `author` and `publishedDate`, supports `startPublishedDate` for a recency window. **But there is no link filter**: `includeDomains` filters the host of the *result*, not pages that link *to* a host. So "who links to bbc.com" degrades to a text search for the string. It is an **index**, not a feed. $4–12 per 1000 searches, $1/1000 contents, **$10 free per account per month**. |
| **Context.dev** | ❌ not for social | A scrape/query API over web pages, news, companies and people — Scrape Anything, Map, Crawl, Search Web, Monitors with signed webhooks. **No social-post coverage, no stream.** |
| **Oxylabs** | ❌ skip | SERP/scraper APIs, from ~$0.50/1000, **card required**. Nothing we need. |
| **Bluesky Jetstream** | ✅ the only real-time source | Keyless public firehose, ~40 posts/sec verified first-hand. Domain filtering is **our own code**, not a provider feature. |

### Two corrections to our own documents

- 🔴 **Apify latency rules it out of the live demo.** Apify's own Standby docs say a request
  needing a new run "may take up to 2 minutes". A cold multi-actor run cannot sit inside a
  five-second moment. If Apify is used at all it is for **pre-fetched** material, labelled as
  such.
- ⚠️ **The "Context.dev: 50,000 credits per team, no card" line in
  [`hackerpack.md`](./hackerpack.md) could not be confirmed from public sources** — the public
  free tier is 1,000 credits/month. Our figure came from the platform hackerpack page, which is
  the authoritative source for a sponsor deal, so this is probably fine. But **confirm the number
  on the platform before planning around it.**
- **Apify puts the legal burden on us.** Their GDPR page: *"you are considered the data
  controller for this personal data"*; their legality article says deciding what is lawful
  "remains your responsibility". Combined with the Art 9 finding in
  [`the-five-seconds.md`](./the-five-seconds.md), that is a reason to keep live scraping narrow.

## 2. "Connect a domain" — there is no one-record path to logs

The question asked directly: *is there any option where the owner changes **one DNS record** and
nothing else?* **For logs, no.**

| Option | Owner effort | Catch |
|---|---|---|
| **Cloudflare Logpush** (`http_requests` dataset) | **~5 min, zero server changes** — create a job, paste our HTTPS endpoint and a secret | Only if the domain is already proxied through Cloudflare. Our endpoint must accept `POST` and a gzipped `test.txt.gz` probe at job creation or it fails. Gzipped ndjson. **No backfill** — logs are lost while a job is off. ⚠️ **Plan conflict unresolved:** Cloudflare's own Logpush page lists Free/Pro/Business/Enterprise; third-party docs say Enterprise only. Resolve before promising it. |
| **One line in nginx** | minutes, but needs root + reload | `access_log syslog:server=host:port,tag=...` (syslog support since nginx 1.7.1) — UNVERIFIED, test it |
| **Log shipper** (Vector/Filebeat/Promtail) | 30–60 min for a non-ops owner | Highest friction, works everywhere |
| **Our reverse proxy in front** | one CNAME — but we must terminate TLS for their hostname | Only for a subdomain they can dedicate; apex is not possible this way |

**Domain *ownership* proof is the easy part:** a DNS TXT token is ~5 minutes and fully
automatable — the same flow Google Search Console uses for a Domain property. That is worth
building because it is what lets us say "this is your site" before showing anything.

### Hostinger specifically

- A **VPS customer can read their own access log over SSH** — it is a normal Linux box. That is
  the connected mode's real install path, and it is the mentor's own customer.
- **Hostinger offers no documented log export, log drain or CDN log streaming** (UNVERIFIED —
  their CDN docs were not fetched). Another question for the booth.
- CyberPanel, where installed on a VPS, exposes access/error logs in its own UI.

## 3. What this means for the promise

**We may say:** type a domain and see, live, who is pushing its links right now, with the
evidence and a calibrated confidence — and connect your server in five minutes to add the
traffic layer.

**We may not say:** that we scrape anyone's traffic from outside, that one DNS record is enough
for logs, or that a single provider call returns "all mentions" of a domain.

**The architecture consequence:** the live path is **Jetstream + our own domain filter**, because
it is the only real-time source and the only one whose terms we are inside. Exa becomes the
*breadth* layer for pages rather than posts, run on demand, not in the five-second moment. Apify
stays out of the live path entirely.

## Open

- [ ] Resolve the Cloudflare Logpush plan question — it decides whether the 5-minute flow is real
      for most owners or only for Enterprise.
- [ ] Confirm the Context.dev credit figure on the platform hackerpack page.
- [ ] Test the nginx syslog one-liner on the VPS once the freeze lifts.
- [ ] Ask the Hostinger booth whether the CDN can export logs at all.
