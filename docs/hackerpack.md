# What we actually get — the hackerpacks

Read first-hand off `hackjunction.app/participate/45/hackerpacks` and all 20 pack pages,
13 Nov 2026 ~23:20 CET. These are **resources**, not prizes — they decide what we can
afford to build this weekend.

**No promo code or key is written in this file. This repo is public.** Each entry names its
pack page (`/participate/45/hackerpacks/<id>`); the code is on that page, one click away.

---

## 🔴 Do this now — the one credit the platform issues itself

**OpenAI $100 Codex credits** (pack `70`) is the only pack with a code pool on the platform.
Checked through the platform's own API: codes are still available and **nobody on Felix's
account has claimed one**. Every other pack is claimed off-platform (a form, a booth, Discord).

Per-person, not per-team — **each of the three of us claims their own on their own account.**

- 2,500 usage credits ≈ $100, **valid 30 days** after redemption, each link redeemable once.
- Needs a **personal ChatGPT account** (Free tier works) in a **desktop browser**.
- Covers ChatGPT/Codex usage only — **not** API credits, not a paid subscription.
  So this does not pay for anything we call from code.
- Campaign owner for support: Juhana Peltomaa, OpenAI.

## Compute and API credits

| Pack | What | Scope | How to claim | Watch out |
|---|---|---|---|---|
| CRACI `64` | **€1000 platform credits** + free CI/CD for the whole event | per team | Junction invite code from `#craci` on Discord or the booth (3rd floor, next to the stage), then `platform.craci.com` | ~10 min setup. *"Running your project's CI on CRACI is required to qualify"* for their tech challenge |
| Verda `71` | **€200** AI cloud credits — GPU compute + inference API | per participant | Sign up to the Verda Console → claim form → redeem/share per their docs | Booth #6, floor 3, mentors all weekend |
| Modal `62` | **$100** for CPU/GPU up to B300, per-second billing, scales to zero | per participant | Modal account, then redeem through their form with the code on the pack page | **Credits valid one year** — the only ones that outlive the weekend |
| OpenAI `70` | **$100** Codex credits | per participant | See above | 30 days, ChatGPT-side only |
| Exa `49` | **$50** credits + full API (search, contents, Websets, Monitors, MCP) | per participant | `dashboard.exa.ai/billing`, code at checkout | Demo gallery at `demos.exa.ai` |
| Aiven `66` | **$50** credits | per participant | Just sign up to Aiven | Mentors on site all days |
| Featherless `51` | **$50** API credits, open-weight model catalog, OpenAI-compatible at `api.featherless.ai/v1` | per participant | Junction coupon link on the pack page → register → API key | |
| Featherless `52` | Bare-metal GPU credits (fine-tuning, big embedding jobs) | **selected teams only** | Talk to them about the project and compute need | Not guaranteed |
| Context.dev `60` | **50,000 API credits, no expiry** — live web → structured data, 50 concurrent requests | per team | `context.dev/signup`, redeem the code on the pack page | **No credit card.** SDKs for TS/Python/Ruby/Go + MCP at `mcp.context.dev/mcp` |
| Oxylabs `61` | `/search` + `/scrape` web API | per team | Free account → pick the **$500 Flexible plan** → code at checkout = first month free | ⚠️ **Credit card required, and you must cancel after the event or you get billed monthly.** Only take this if we actually need it |
| Anthropic `67` | **$20** Claude API credits | per participant | Apply through the offer link on the pack page | Smallest pot here; it is API credits though, unlike Codex |
| Google DeepMind `68` | Gemini API credits | unstated | **Not released yet** — *"instructions will be shared once confirmed by Google DeepMind"* | Amount unknown. Don't plan around it |

## Storage, data and the >100 MB problem

**IronShard `59`** — S3-compatible object storage with an MCP interface, and the practical bit:
challenge input data is preloaded into **read-only buckets** you can **fork zero-copy** into your
own writable bucket. If the submission includes trained models or datasets too big for GitHub's
100 MiB per-file limit, put them in a bucket and **reference it in the submission — judges are
given read access.** Sign up at `console.ironshard.dev/sign-up` with the **same email as the
hackathon account**, event code on the pack page, one team account, invite the others.

**MariaDB `69`** — open to every team whichever challenge we pick; they want the most interesting
use of the database itself. Starter repos: `MariaDB/mariadb-wikipedia-datasets`,
`mariadb-vector-wikipedia-rag`, `mariadb-system-versioning-wikipedia`. Mention MariaDB in the
submission or show them at the booth; work gets published and credited on mariadb.org.

**Cloudinary `63`** — image/video upload, transform, optimize, deliver on their free tier.
⚠️ To be judged you **must** enter the project at `cld.media/hackathon-survey`.

## Hardware — you pick it up on the 5th floor

- **Junction Hardware Lab `57`** — assorted devices, booked through `2026.hackjunction.com/hardware`.
- **Arduino `65`** — UNO Q kits (2 GB RAM / 16 GB eMMC) **to borrow**, with USB-C cable, 8-in-1 hub,
  45 W supply, USB camera, and Modulino buzzer / knob / thermo sensors. Bring a laptop with
  **Arduino App Lab** installed. Mentors at the Arduino Hardware Lab and on Discord.
- **Wendy `56`** — `wendy run` deploys your app to a Jetson, Raspberry Pi, ESP32 or Arduino over
  USB; open source, Python/Swift/Rust/ROS. **Has a simulator, so it works with no hardware at all.**

## Robotics / CV tooling, free for the event

- **Qualia `53`** — curation for robot-learning data: upload SO-101 arm or Rokoko mocap episodes,
  it flags failed takes and timing problems, you keep the good ones as a versioned dataset and
  train (e.g. pi0.5 with LeRobot). Sign up at `junction.qualiastudios.dev/login`.
  ⚠️ **The environment is deleted after 7 days — download anything you want to keep.**
- **Scailab / Scaipipe `50`** — computer-vision pipeline in the browser: import and annotate
  images, generate labelled synthetic data from 3D scenes in a node editor, train and evaluate
  detection/segmentation/tracking. **No GPU needed**, and it has an **MCP server so Claude can
  drive the whole pipeline.** Access by their Google form (`forms.gle/Arq4iyekTGXKXmk3A`) — they
  email you a login.

---

## What this changes for us

- **Compute is not the constraint.** CRACI's €1000 plus Verda's €200 and Modal's $100 per head
  covers anything we can realistically run in 36 hours. Modal is the one to reach for for GPU
  work — Python-native, per-second, public URL for the demo in one `modal deploy`.
- **Two of these are MCP servers we can point Claude at today**: Context.dev and Scaipipe.
- **Codex credits don't pay for API calls** — if a plan assumed $100 of OpenAI API, it was wrong.
- **Four packs have a qualification requirement on top of the normal submission**: CRACI (CI must
  run on their runner), Cloudinary (their survey form), Arduino (hardware + Project Hub),
  IronShard (bucket references in the submission). Same trap list as in the README.
