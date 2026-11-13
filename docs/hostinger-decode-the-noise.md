# Decode the Noise (Hostinger) — the deep read

**Status: this is the challenge we're most likely taking.** Everything in the first two
sections is read first-hand off `hackjunction.app/participate/45/challenges/357` on
13 Nov 2026 ~19:55 CET, including the Insight and Resources tabs. Everything after that is
external research with a source on every claim. Where something is unconfirmed it says
UNVERIFIED.

Short version of the brief is in [`challenges.md`](./challenges.md); this file supersedes it
for anything about this challenge.

---

## 1. What they actually ask for (verbatim structure)

> *"Build an AI-powered tool that detects propaganda, bots and bad actors online – and deploy
> it live on Hostinger's infrastructure."*

Analyse **text, social content or web-traffic patterns** to detect coordinated propaganda,
bot-driven amplification or bad-actor behaviour. The form is open: *"an API, a dashboard, a
classifier with an explainable verdict, or a monitoring service for site owners."*

**The hard constraint:** it *"must be deployed and running live on Hostinger's infrastructure
by the end of the hackathon, with a public URL the judges can open."*

And the sentence that decides how we spend the weekend:

> *"We care more about the strength and honesty of your detection logic – including how you
> handle false positives – than a polished front end."*

> *"We're especially interested in approaches that are transparent and measurable – show your
> precision and false-positive rate, not just a demo that works on the happy path."*

### Judging — the weights are public

| Criterion | Weight |
|---|---|
| **Detection quality** (accuracy, false-positive handling, explainability) | **40 %** |
| Technical execution & live deployment on Hostinger | 25 % |
| Originality & real-world impact | 20 % |
| Demo clarity | 15 % |

### The four directions they suggest (pick one or combine)

1. **Spot the swarm** — coordinated inauthentic behaviour from *posting timing, account
   similarity, text reuse and amplification patterns*, explicitly **not** from any single post.
2. **Name the technique** — classify manipulation techniques in text (loaded language, fear
   appeals, whataboutism, fabricated authority) and **explain the verdict in plain language**,
   so a reader learns *why* something is propaganda.
3. **Clean the traffic** — malicious bots, scrapers and credential-stuffing vs real visitors in
   **web-server logs**, with an actionable view for site owners.
4. **Trust the source** — provenance score for a page or account (domain age, cloned content,
   AI-generated text, link networks).

## 2. What they give us

- **Hostinger VPS (KVM 2) per team**, free for 1 month — **2 vCPU, 8 GB RAM, 100 GB NVMe,
  8 TB bandwidth, full root, Docker**, free domain, one-click templates for Ollama, n8n,
  Node.js and Python.
- **AI Router credits** — one API key, one shared balance, *"150+ leading models"* (Anthropic,
  OpenAI, DeepSeek, Mistral, xAI), OpenAI- and Anthropic-SDK compatible.
- **AI Builder** 1-month plan for a front end.
- **Hostinger Connector, API, CLI and MCP server** — drive the VPS from Claude Code.
  Docs `docs.hostinger.com/hostinger-connector` · API `developers.hostinger.com` ·
  MCP `github.com/hostinger/api-mcp-server`.
- **Mentors at the booth all weekend:** Minijus Savickas (VPS Product), Emilis Strimaitis
  (Innovations, AI agents). Plus a dedicated help channel.
- **Talk Sat 13:00:** *"Vibe Coding Solved Building. Now Comes the Hard Part."*
- Prizes: 2000 / 1000 / 500 €, and all three winners get a year of hosting + domain, swag and
  a social-media spotlight. (Recorded for completeness; not a selection criterion for us.)

### ✅ Externally hosted models are explicitly allowed

The page says: *"Use any language, framework or model – open-source or hosted – as long as the
final product runs on Hostinger."* So calling an API from the VPS does **not** break the
deployment requirement. That settles the obvious worry about where inference runs. Confirm the
exact reading with the mentors anyway, because 25 % of the score hangs on it.

### What the box can and cannot do — this shapes the architecture

2 vCPU / 8 GB / **no GPU**:

- ✅ An **encoder-size classifier** (DistilBERT / MiniLM / DeBERTa-small, ~100–300 MB) runs
  fine on CPU — tens of milliseconds per paragraph. This is the right home for the scoring model.
- ✅ Classic feature models (gradient boosting over timing / similarity / graph features) are
  effectively free.
- ❌ **The Ollama one-click template is a trap for the demo.** A 7–8 B model on 8 GB of CPU RAM
  runs at a few tokens per second. Fine for one offline example, far too slow for a judge
  clicking around. UNVERIFIED on this exact VPS — worth a 10-minute test before committing.
- ✅ Therefore: **deterministic detector on the VPS, AI Router only for the plain-language
  explanation.** That also keeps the detection reproducible, which the 40 % criterion rewards.

⚠️ **Discrepancy:** the challenge page claims *"150+"* models on AI Router; Hostinger's own
product pages say *"20+"* ([hostinger.com/ai-router](https://www.hostinger.com/ai-router),
[support article](https://www.hostinger.com/support/?p=19146)). Nothing depends on which is
right, but don't quote "150+" in our pitch. Base URL is `https://router.hostinger.com/v1`,
keys start with `sk-hstgr-`, and it is OpenAI-compatible — LiteLLM already ships it as a
provider ([docs.litellm.ai](https://docs.litellm.ai/docs/providers/hostinger_router)).

The MCP server is real and maintained: [github.com/hostinger/api-mcp-server](https://github.com/hostinger/api-mcp-server),
needs **Node 24+**, and ships modular binaries including VPS and hosting/deploy tool sets, with
Docker container management on the VPS ([Hostinger support](https://www.hostinger.com/support/?p=342)).
**That means Claude Code can deploy and manage the box directly — the 25 % execution criterion
is largely a configuration problem, not a coding one.**

---

## 3. The real problem: 40 % of the score needs labelled data

They want a **precision and false-positive rate on real labelled data**. A demo without a
number against ground truth cannot score well on the largest criterion. So the binding
question is not "what do we build" but **"what can we measure against by tomorrow morning."**

### Datasets, and which ones are actually reachable this weekend

| Dataset | Labels | Reachable tonight? | Fits direction |
|---|---|---|---|
| **SemEval-2020 Task 11 / Propaganda Techniques Corpus** | 14 propaganda techniques, span-level, news articles | ⚠️ **Gated** — registration at `propaganda.qcri.org/ptc/registration.php`; the HF entry `SemEvalWorkshop/sem_eval_2020_task_11` is a loader that still requires the manual download | 2 |
| **SemEval-2023 Task 3** | 23 persuasion techniques, 6 languages, 1 612 articles / ~26 663 train+dev samples ([KInIT paper](https://arxiv.org/abs/2304.11924)) | ⚠️ Task data was also distributed via registration — **check first thing**, a fine-tuned model is public at [huggingface.co/kinit/...](https://huggingface.co/kinit/semeval2023-task3-persuasion-techniques) even if the raw data is gated | 2 |
| **FiveThirtyEight IRA troll tweets** | 2 973 371 tweets, 2 848 IRA handles, 5 hand-assigned troll types, Feb 2012 – May 2018 ([FiveThirtyEight](https://fivethirtyeight.com/features/why-were-sharing-3-million-russian-troll-tweets/)) | ✅ **Open on GitHub, CC0** — the most usable positive-class set we have | 1, 4 |
| **TwiBot-22** | 1 M accounts, 4 entity types, 14 relation types, NeurIPS 2022 benchmark ([arXiv](https://arxiv.org/abs/2206.04564)) | ❌ **Dead end for us.** Access requires emailing the authors from an *institutional* email, naming your research advisor and use case | 1 |
| **TwiBot-20 / cresci-2017 (Bot Repository)** | labelled bot vs human accounts | ⚠️ Request form, same problem at hackathon speed. UNVERIFIED whether any mirror is openly downloadable | 1 |
| **AUTH web robot detection — server logs** | **Sessions labelled bot vs human**, plus extracted simple and semantic features | ✅ **Open, CC BY 4.0, no key** — [zenodo.org/records/3477932](https://zenodo.org/records/3477932). 4 091 155 requests over March 2018 from a university library search engine, 27 061 unique IPs, 3 441 user-agent strings, IPs anonymised. `simple_features.csv` is 16 MB, the raw JSON is 3.2 GB | **3** |
| **Wikipedia Sockpuppet Investigations (CheckUser-confirmed)** | Coordinated multi-account editing, adjudicated by an uninvolved admin plus CheckUser technical correlation. `Category:Wikipedia sockpuppets of <user>` = confirmed, a separate category = merely suspected, and **every case carries a human-written rationale** | ✅ **Open, CC BY-SA, free MediaWiki API, no key.** Category members → account names → `list=usercontribs` for timestamped edit text | **1** |
| **Verified crawler IP ranges** (Google, Bing, OpenAI, Anthropic) | **Legitimate** automation — self-declared and IP-range-backed by the vendors themselves | ✅ Plain JSON at each vendor's own path, no key, minutes | **3** |
| **EUvsDisinfo** | ~14 500 catalogued disinformation cases, each with an EU East StratCom verdict **and a written debunk** | ✅ Via the Mendeley mirror, **CC BY 4.0, no signup** — [data.mendeley.com/datasets/yhdtkszvgp/3](https://data.mendeley.com/datasets/yhdtkszvgp/3). The official API is email-request only; use the mirror | 2, 4 |
| **RAID / HC3** | Machine-generated vs human text | ✅ Ungated on Hugging Face, minutes | 2, 4 |

### 🔴 Correction, 13 Nov 20:25 — direction 3 is not the dead end this document first claimed

An earlier version of this file said *"no credible public labelled set found"* for web-server-log
bot traffic, and concluded that direction 3 was the weakest of the four. **That was wrong.**
`@claude-timi` found labelled log corpora, and the AUTH dataset above was then verified
independently through the Zenodo API: it is real, open, CC BY 4.0, and its own description says
it contains *"a labelled dataset of log entries grouped into sessions along with their extracted
features"*. So the claim collapses.

Also found by `@claude-timi`, **not independently verified here** — treat as leads, not facts:
**SR-BH 2020** (Univ. Alcalá, CC0, ~907 k WordPress-honeypot requests, 525 k normal / 383 k
anomalous, 13 labels, with a known mislabelling audit and a corrected `SR-BH2020-fix`);
**Biblio-US17** (Seville, labelled production-server requests); **CSIC 2010** (older, generated);
**Iliou et al.** (Bournemouth / m4d.iti.gr — logs *plus mouse movement*, human vs moderate vs
advanced bot); **notesbymuneeb/ai-waf-dataset** (MIT, 12 k requests). Unlabelled real logs useful
as a false-positive test bed: Zenodo 18895701 (Polish e-shop, Dec 2025, 179 k requests) and
17178461 (30 days of ModSecurity-blocked requests — positives only).
He also reports a 2026 state-of-the-art reference of **67.7 % of bots caught at a 3 % false-positive
rate from logs alone** — UNVERIFIED, but if it holds it is the number to beat, and it is far more
quotable than anything on the social side.

**Credential stuffing specifically has no public labelled set** — there, a honeypot is the only
honest route.

**Revised consequence:** all four directions now have a path to a real eval set, and direction 3
has the *best* one — a single CC BY 4.0 file with bot/human session labels somebody else assigned,
plus vendor-published ground truth for the legitimate-automation class. What stays true is the
narrower point: **a precision number we invented by writing the attack ourselves is worthless**, so
self-generated honeypot traffic is a stress test to disclose, never a headline metric.

### The live data source that is actually legal

**Bluesky Jetstream** — a keyless public WebSocket firehose of every public post, like and
follow, no API key, no auth, no fee, no app review, and v2 added replay of historical data
([bsky.network/docs/jetstream](https://bsky.network/docs/jetstream),
[docs.bsky.app/blog/jetstream](https://docs.bsky.app/blog/jetstream)).

This is the single most useful external fact in this document. It means a **live** demo —
judges watching real posts stream in and get scored — without scraping anything, without a
developer agreement, and without a rate limit to apologise for. X, Reddit and Facebook all
cost money, approval or a ToS violation; Bluesky costs nothing and is designed to be consumed.

## 4. What a judge already knows, and where the gap is

| Prior art | What it does | What it does **not** do |
|---|---|---|
| **Botometer** (Indiana University) | Scores a Twitter account's bot likelihood from ~1 200 features | Scores **accounts in isolation**, not coordination. And see below — it has a documented false-positive problem |
| **CooRnet / PyCooRnet / CooRTweet** | Detects **coordinated link-sharing behaviour** — entities repeatedly sharing the same URL inside a time window ([CRAN](https://search.r-project.org/CRAN/packages/CooRTweet/readme/README.html), [PyPI](https://pypi.org/project/pycoornet/)) | R/Python research packages, batch, no live stream, no explanation layer, no deployment |
| **Hoaxy** | Visualises how claims and fact-checks spread | Visualisation, not a verdict |
| **Bot Sentinel, Graphika, Blackbird.AI, Alethea, Logically** | Commercial CIB / narrative intelligence | Closed, enterprise, unauditable. A judge cannot check their precision either |

**The gap is the same in every row: nobody gives a site owner or a reader a live, explained,
measured verdict.** Research tools measure but don't deploy or explain; commercial tools
deploy but won't show their numbers. The brief is almost literally asking for the missing cell.

### 🔴 The false-positive story is the challenge

Peer-reviewed work has repeatedly found that the best-known tool in this space misfires badly
on real people:

- **Nearly half of all US Congress members** were rated as bots in April 2018; likewise
  **17.7 %** of Reuters journalists, **21.9 %** of UN Women staff and **35.9 %** of staff at the
  German news agency *dpa*. The authors conclude that studies claiming to study social bots had
  in fact been studying Botometer's false positives
  (Gallwitz & Kreil, [arXiv:2207.11474](https://arxiv.org/pdf/2207.11474)).
- It fails in **both** directions: across five labelled datasets (n = 4 134, English and German,
  three months), thresholds proved unstable — **36 %** of known bots from New Scientist and
  **60.7 %** from Botwiki came out as human
  (Rauchfleisch & Kaiser, *PLoS ONE* 15(10): e0241045,
  [doi:10.1371/journal.pone.0241045](https://doi.org/10.1371/journal.pone.0241045),
  [preprint](https://papers.ssrn.com/abstract=3565233)).

This is the thing to open the pitch with. The sponsor asked for false-positive handling as
the top criterion because **the incumbent tools in this field have a false-positive crisis**.
Leading with our FP rate, a threshold chosen deliberately, and an "insufficient evidence"
verdict instead of a forced guess, answers the 40 % criterion directly and in their language.

### What counts as a respectable number

SemEval-2020 Task 11, the standard benchmark for direction 2:

- **Span identification:** winner Hitachi **F1 51.55**, then ApplicaAI 49.15, aschern 49.10
- **Technique classification (14 classes):** best **F1 62.07**
  ([task overview](https://arxiv.org/abs/2009.02696), [aschern system paper](https://arxiv.org/pdf/2008.02837))

So on the propaganda-technique task, **~0.5–0.6 F1 is state of the art.** If our number lands
near that, it is a good number and we should say so with the citation. **If a model reports
0.95 on this task, something is wrong** — and a judge who knows the field will assume leakage.
Saying this out loud in the pitch is itself a signal of honesty, which is what they asked for.

## 5. Legal and ethical — what we must not get wrong

- **Scraping X / Reddit / Facebook for the live demo is out.** Each needs a paid tier, an
  approval process, or breaks the ToS. Bluesky Jetstream replaces all of it (above).
- **Social media content is personal data under GDPR**, essentially always — the Dutch DPA's
  position is that scraping social media "almost always involves personal data", and the EDPB
  published dedicated web-scraping guidelines
  ([EDPB guidelines](https://www.edpb.europa.eu/system/files/2026-07/edpb_guidelines_2020603_webscraping_v1_en_0.pdf)).
  Private-sector scraping leans on legitimate interest, Art. 6(1)(f), which requires a
  documented three-part assessment including a balancing test
  ([CNIL 2025 guidance summary](https://www.zyte.com/blog/ai-personal-data-scraping-europe-guidance/)).
- **What that means for a weekend prototype** — not a legal opinion, but the standard
  mitigations, and they're cheap:
  - score **content**, not named people; don't publish a verdict about an identifiable account
  - **don't persist** raw posts or handles — hash or drop identifiers, keep only features
  - use **public research archives** (IRA tweets) and an **open protocol** (Bluesky) rather
    than scraping a walled platform
  - say in the demo and the README what is and isn't stored
- **Label the output as a signal, not an accusation.** Given the Botometer record above, any
  tool that prints "this person is a bot" is one screenshot away from being the story. Verdicts
  should be about *behaviour patterns* with the evidence attached, and should be allowed to say
  "not enough evidence".

## 6. Open questions — ask at the booth

- [ ] Does "runs on Hostinger" allow calling an **external** model API, beyond their own AI
      Router? The page says hosted models are fine; get it from a mentor anyway (25 % of score).
- [ ] Is the AI Router balance **per team or per participant**, and what happens when it runs out?
- [ ] Does a local model via the **Ollama template** actually respond fast enough on KVM 2, or
      is it demo-unusable? (10-minute test.)
- [ ] Is there a Hostinger-side **log dataset** they'd give us for direction 3? They host
      millions of sites; if they can hand over even anonymised labelled traffic, direction 3
      becomes the strongest option instead of the weakest.
- [ ] Do they want the public URL in the submission form, or shown live at the table?

## 7. Provenance of this document

- **§1 and §2** — read first-hand off the platform, 13 Nov 2026 ~19:55 CET. Those are their words.
- **§3–§5** — external sources, each linked inline. The Botometer figures and the SemEval
  baselines come from peer-reviewed papers. Dataset reachability was checked by search rather
  than by downloading, **except** the AUTH log corpus, which was confirmed through the Zenodo
  API directly. The direction-3 claim in the first version of this file was wrong and is
  corrected above — see that box before quoting anything from this document.
- Deeper prior-art and legal research is running on the `reshercher` bot; fold its answer in
  here when it lands rather than starting a second document.
- Nothing here is a decision about what we build. It is the evidence to decide with.
