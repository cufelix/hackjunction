# Decode the Noise at Junction 2026: should we pick it, and what demo wins?

*Research for Oliver, 13 Nov 2026. Public web and public X only, no email or private data. Every claim links to its source. **UNVERIFIED** marks anything I couldn't confirm. Sections A–C below are the full evidence; this top part is the synthesis plus Q4 (demo options).*

## Quick answer

- **Pick the challenge, but not the sockpuppet idea as the core.**
  - **No public dataset joins comment accounts to traffic behaviour with sockpuppet labels** (Section C).
  - **The best-known sockpuppet labels are circular.** Kumar et al. define a sockpuppet by shared IP and timing ([arXiv 1703.07355](https://arxiv.org/abs/1703.07355)).
  - **Venue Wi-Fi makes judges look like one ring.** Shared IPs are a documented false-positive source ([Cloudflare on CGNAT](https://blog.cloudflare.com/detecting-cgn-to-reduce-collateral-damage/)).
  - **Small 2026 open-source projects already pitch actor linking and bot-network clustering** ([StyloBot README](https://github.com/johnspencentu/stylobot), claims UNVERIFIED).
- **"Explainable" alone isn't novel.**
  - CrowdSec bot detection lists per-block `score_reasons` ([docs](https://docs.crowdsec.net/docs/appsec/bot_detection/how_it_works/)).
  - Anubis exposes the matched rule ([policies docs](https://raw.githubusercontent.com/TecharoHQ/anubis/main/docs/docs/admin/policies.mdx)).
  - In the Junction 2025 WithSecure track, teams already shipped confidence scores with rationale (Section A).
- **The defensible gap is narrow:**
  - calibrated probabilities instead of a rank-only score (Cloudflare's 1–99 score is not a calibrated probability: [bot score docs](https://developers.cloudflare.com/bots/concepts/bot-score/));
  - a live false-positive rate measured against *known humans* with an error bar, rather than Cloudflare's challenge-solve-rate proxy ([Challenge Solve Rate](https://developers.cloudflare.com/cloudflare-challenges/reference/challenge-solve-rate/));
  - running on one CPU-only VPS, which Hostinger VPS doesn't offer at app level today ([firewall](https://www.hostinger.com/support/8172641-how-to-use-a-managed-vps-firewall-at-hostinger/), [malware scanner](https://www.hostinger.com/support/8450363-hostinger-vps-malware-scanner/)). AI Audit is a CDN feature for web hosting, not VPS ([Hostinger blog](https://www.hostinger.com/blog/cdn-ai-audit/)).
- **Two separate games:**
  - Hostinger judges its own track "against their own criteria".
  - The overall prize goes via peer review, then a top-5 live jury pitch ([2026.hackjunction.com](https://2026.hackjunction.com/)).
  - Public event-platform data suggests **Gavel voting is OFF for 2026** and face-to-face peer review is on ([platform page](https://hackjunction.app/hackathons/junction-2026)). UNVERIFIED in practice, so **ask an organizer tonight.**
- **The Decode the Noise challenge page isn't public.** The 40% weight and whether Hostinger provides labelled data come only from your brief (UNVERIFIED). **Confirm what labelled data Hostinger supplies.** It decides everything below.

## Recommendation (evidence-based; the call is yours)

1. **Core:** per-visitor verdict, calibrated probability, top-3 reasons, and a live FPR estimate with a 95% CI computed from a "trusted human" stream.
2. **Score on held-out labelled data** (Hostinger's if provided, otherwise Iliou et al. web logs + mouse data ([dataset](https://m4d.iti.gr/web-bot-detection-dataset/), CC BY-NC-SA, tiny) and the Lagopoulos sessions from the earlier brief). Show a confusion matrix and reliability diagram ([Guo et al. on calibration](https://arxiv.org/abs/1706.04599)).
3. **Sockpuppet linking:** stretch slide or tab only, labelled "synthetic, lab-only" (the "lab-only evaluation" pitfall: [Arp et al.](https://arxiv.org/abs/2010.09470)), with a shared-IP guard so the venue Wi-Fi doesn't trip it.

## Q4: three 30-second demo options

Evidence on what lands at Junction (Section A):
- **Recent winners showed something moving that the audience could touch.** BREAKDANCE (2024) had the audience scan a QR code; DroneStalker (2025) turned tracking into a "shoot the drone" game ([repo](https://github.com/takara-ai/DroneStalker)).
- **None of the bot-detection winners I found showed precision or FPR numbers.**
- **The verified wins in this category were mostly sponsor prizes**, and the most credible one had real ground truth ([0xSybil](https://ethglobal.com/showcase/0xsybil-h9e55)).

### Option 1: "Be the bot" (judge vs detector, live)
A judge scans a QR code and browses the demo shop on their phone, getting a green HUMAN 0.97 with reasons. Then you press one button that launches a stealth Playwright bot and a curl-impersonate script ([curl-impersonate](https://github.com/lwthiker/curl-impersonate)). Both turn red with reasons, and the live FPR counter ticks.
- **Pros:** interactive, like the 2024 and 2025 winners; the verdict is instant and readable.
- **Cons:** the judge will try to break it in front of you, and good evasion tools often win. FP-Inconsistent reports commercial bot services evading DataDome 52.93% of the time ([arXiv 2406.07647](https://arxiv.org/abs/2406.07647)). Venue Wi-Fi and privacy browsers can false-flag the judge ([Mozilla bug 1909961](https://bugzilla.mozilla.org/show_bug.cgi?id=1909961), [Anubis #1645](https://github.com/TecharoHQ/anubis/issues/1645)).
- **Mitigation:** scripted bots of rising difficulty, and openly show one you *miss*.

### Option 2: "Scoreboard" (replay held-out traffic)
Replay a held-out labelled log at 50× speed through the guard. A live split screen shows the visitor stream with verdicts on the left, and a running confusion matrix, precision, FPR with 95% CI and a reliability curve on the right. Finish with "at 1% FPR we catch X% of bots".
- **Pros:** directly hits the 40% detection-quality weight with numbers judges can check; you control it, so nothing breaks; no other winner showed this.
- **Cons:** less visceral, and peer reviewers may not read a confusion matrix in 30 seconds.

### Option 3: "Sockpuppet reveal"
A review section on the demo site with 6 reviews. Press "link accounts" and 3 reviews collapse into one actor cluster, with the evidence shown (same device fingerprint, typing cadence, session overlap) and a confidence.
- **Pros:** the most memorable story; easy for non-experts.
- **Cons:** the data is synthetic and the labels are yours, so skeptics will dismiss it (Section C). Similar claims already exist ([StyloBot](https://github.com/johnspencentu/stylobot)). Shared venue IPs risk a false cluster live.

**Combination the evidence supports:** open with Option 1 (10 s, human green and bot red), switch to Option 2's scoreboard (15 s, the numbers), and keep Option 3 as a 5 s teaser or a Q&A answer.

## The case against picking this challenge at all

- **Bot, fake-account and sybil projects rarely win overall prizes.** Of about 20 checked, the wins were mostly sponsor or category prizes (Section A).
- **Detection quality is adversarial.** Cheap evasions such as stealth plugins ([puppeteer-extra-stealth](https://github.com/berstend/puppeteer-extra/tree/master/packages/puppeteer-extra-plugin-stealth)) and [undetected-chromedriver](https://github.com/ultrafunkamsterdam/undetected-chromedriver) let a red-team judge embarrass a 30-hour detector in minutes. Bots also beat humans on CAPTCHAs ([Searles et al., USENIX Sec 2023](https://www.usenix.org/system/files/usenixsecurity23-searles.pdf)).
- **Small labelled human sets make the FP claim weak.** With 0 false positives on 56 humans, a true FPR of up to about 5% is still plausible (Section C).
- **If Hostinger supplies no labelled data, the 40% criterion collapses into self-made labels.**

## Claims to make vs avoid (summary of Section C)

| Make | Avoid |
|---|---|
| Precision and FPR with 95% CI at a pre-registered threshold, on time- or site-held-out data | "99% accuracy" |
| Recall at 1% and 5% FPR | "Zero false positives" on a small sample |
| Reliability diagram plus ECE or Brier on held-out data | Calibration measured on training data |
| Beats user-agent-only and IP-rate-only baselines | "Catches LLM agents / undetectable bots" |
| Separate adversarial test set, misses shown | Sockpuppet results scored on IP- or timing-derived labels |
| "Live FPR is an estimate from the trusted-human stream" | Live FPR presented as ground truth |

---

# Section A: Junction judging, past winners, hackathon bot-detection demos

# A. How Junction judges and what wins, plus hackathon bot-detection demos (A and C)

Researched 13 Nov 2026 (Europe/Prague). Sources: public web and public X only. Event times are given in Helsinki local time (UTC+2).

## TL;DR

1. **Two separate games.** The Hostinger track winner is picked by Hostinger "against their own criteria". The main prize goes through peer review to a top 5, then a live pitch to a 7-person grand jury. ([2026.hackjunction.com](https://2026.hackjunction.com/), see Judging and FAQ.) Your 40% detection-quality weighting only matters to Hostinger's judges. Peers in the main-prize review won't run your eval.
2. **2026 may not use Gavel.** The public event JSON on the Junction Platform shows `"hasGavelVoting":false, "physicalPeerReview":true, "reviewAnyChallenge":false` for Junction 2026. Junction 2025 showed `"hasGavelVoting":true, "physicalPeerReview":false, "reviewAnyChallenge":true` ([2026 page](https://hackjunction.app/hackathons/junction-2026), [2025 page](https://hackjunction.app/hackathons/junction-2025), page source). The 2026 site says "face-to-face peer review to select the top 5". What this means in practice is UNVERIFIED. It could just be config that hasn't been set yet. Ask an organizer on Friday night.
3. **Recent overall winners were visual and interactive, and showed technical depth.**
   - 2025: WAVELET, a real-time Maxwell EM simulation in C++/OpenGL with no AI.
   - 2025 finalist: DroneStalker, a gamified "shoot the drone" web experience on top of a YOLO + LSTM pipeline.
   - 2024: BREAKDANCE. A solo builder pitched with a music video and had the audience scan a QR code to use the app live.
4. **Bot and fake-account detection projects on Devpost and ETHGlobal rarely win overall prizes.** Most of the wins I could verify are sponsor-technology prizes (Worldcoin, Blockscout, Vercel, GoDaddy domain). The exceptions are small-event 3rd places (HopHacks 2019, R.U. Hacking 2019). Many lookalike projects (Reddit and X bot-scoring extensions, "AI CAPTCHA" games) won nothing.
5. **The strongest winning detection demo had ground truth.** 0xSybil (ETHGlobal Paris) checked its clusters against a known sybil attack with 38 manually identified fake accounts.

---

## A1. How Junction picks winners (primary sources)

| Item | Evidence |
|---|---|
| 2026 process: partner tracks pick their own winners; the main event uses face-to-face peer review to choose a top 5; the top 5 pitch live to a grand jury | [2026.hackjunction.com](https://2026.hackjunction.com/) ("How the winner is chosen" and the FAQ: "In three stages…") |
| 2026 schedule: final submissions Sun 15 Nov 10:00, "Review time" 10:30, Final Ceremony 14:00, challenge winners at 15:10, Junction winner at 15:50 | [2026.hackjunction.com](https://2026.hackjunction.com/) schedule. The Junction Platform lists "Review time" at 14:00 Helsinki ([hackjunction.app](https://hackjunction.app/hackathons/junction-2026)), which conflicts with the site. Which one is right is UNVERIFIED. |
| Saturday: preliminary submission Fri 23:59; submissions open Sat 12:00 | [2026.hackjunction.com](https://2026.hackjunction.com/) |
| Max team size 4 in 2026 (5 in 2025) | Platform JSON (`maxTeamSize`) on the [2026](https://hackjunction.app/hackathons/junction-2026) and [2025](https://hackjunction.app/hackathons/junction-2025) pages |
| Hostinger is listed as a 2026 **Challenge partner**. It also gives a Saturday 13:00 keynote: "Vibe coding solved building. Now comes the hard part… attention, timing, and real problems worth solving… test your idea before you start building" | [2026.hackjunction.com](https://2026.hackjunction.com/) (partners block and schedule) |
| The "Decode the Noise" challenge text, its judging weights (40% detection quality) and the data provided | **UNVERIFIED.** The platform's public `challenges` array is empty for 2026 ([hackjunction.app](https://hackjunction.app/hackathons/junction-2026)). Web search and public X search (Hostinger + Junction/hackathon/"Decode the Noise", since Sep 2026) found no public page. The 40% figure comes only from Oliver's brief. |
| 2025: review ran 10:15–14:00 Helsinki, then "Top 5 Pitching" at 14:40 | Platform JSON on the [2025 page](https://hackjunction.app/hackathons/junction-2025): `reviewingStartTime 08:15Z`, `reviewingEndTime 12:00Z`, "Top 5 Pitching" 12:40Z |
| Gavel mechanics: a reviewer sees a project, then a second, and votes which was better. Each reviewer covers about 15 projects. Created by HackMIT in 2015 | [junctionplatform.com/review-options](https://www.junctionplatform.com/review-options), [Athalye 2015](https://www.anishathalye.com/2015/03/07/designing-a-better-judging-system/) |
| Junction's own description (2018): one or two team members stay at the table to demo while the others go and review. "You will always compare the current project with the previous one." Judges whose votes deviate are weighted down | [blog.hackjunction.com (2018)](https://blog.hackjunction.com/forget-the-judges-participants-to-choose-the-winner-at-junction-5af699e1fc19) |
| Recommended criteria (academic case study of Junction): novelty, "techiness", efficacy, "wowness" (pitch, demo and prototype quality), business value | [Crowdsourcing… The Case of Junction, IEEE TEM 2021](https://doi.org/10.1109/tem.2021.3079107) |
| Older infodeck: top 5 chosen "with the Junction Platform and peer-to-peer Gavel Voting"; top 5 pitch on stage; the Grand Jury picks the winner | [hackjunction.com/junction-infodeck.pdf](https://www.hackjunction.com/junction-infodeck.pdf) |

**Time per project (derived estimate, UNVERIFIED):** a ~3.5 h review window divided by ~15 comparisons per reviewer gives roughly 10–14 minutes per project including walking and queueing. Expect 2–5 minutes of real attention. Peers compare you only to the previous table they saw, so the job is to be clearly better than whoever was just before you.

### 2026 grand jury: stated preferences (verbatim themes from the site)
Source: [2026.hackjunction.com, Judges](https://2026.hackjunction.com/).
- **Kia Kanninen (Founders House):** "teams that start from a problem worth solving, not just a feature that's easy to build."
- **Daksh Raghuvanshi (AfterQuery, RL/evals):** "teams that can explain exactly why an agent fails in their world, and whose reward holds when someone tries to break it." This lines up closely with an honest-eval bot detector, and it is also the person most likely to attack your FP claims.
- **Otto Söderlund (Supercell):** "start from a real user problem and ship something people want to come back to."
- **Arturs Polis (Verda):** "willing to tackle the hardest challenges… prototypes with a clear path to production and scale."
- **Hubertus van der Drift (Context.dev):** "move fast, stay pragmatic… one unique insight that solves a problem 10x faster or better."
- Timo Ahopelto (Lifeline Ventures) and Umar Noor (Google DeepMind): no stated criteria.

## A2. Past winners and how they demoed

| Year | Winner / track | Demo format evidence |
|---|---|---|
| 2025 overall | **WAVELET** (Sensofusion Radio City track): real-time EM wave simulation on a 1:1 Hong Kong map, C++/OpenGL/CUDA, "completely AI-free"; €10k | [Rudrrayan Manna, LinkedIn](https://www.linkedin.com/posts/rudrrayan_this-weekend-i-took-part-in-the-junction-activity-7395862667153997824-v2QQ); [Wikipedia](https://en.wikipedia.org/wiki/Junction_(hackathon)). Junction 2025 had 291 projects and 1,500+ participants (same Wikipedia page; [Aalto](https://www.aalto.fi/en/news/aalto-students-and-researchers-shape-together-how-we-age-with-ai-at-junction-hackathon-2025)). How the pitch was delivered is UNVERIFIED (the repo README is a stub: [GitHub](https://github.com/ron0studios/WAVELET)). |
| 2025 top 5 | **DroneStalker** (Sensofusion): YOLO detection, an LSTM claimed to beat the FRED paper baseline by 50.3%, a 200 GB dataset re-hosted. The demo was a **gamified mission**: "follow the commander's instructions and shoot down the drone!" A commenter said the "UI and gamification of this 'dry' challenge was amazing." | [Jordan Legg, LinkedIn](https://www.linkedin.com/posts/404missinglink_we-are-junction-2025-finalists-jacob-kenney-activity-7396593086904086528-M9j6); [README](https://github.com/takara-ai/DroneStalker); [codya.dev](https://codya.dev/project/2511-junction). That page says "Winner", but the team's own post says top-5 finalist. The 50.3% figure is self-reported and UNVERIFIED. |
| 2025 WithSecure track (security) | **Aegis**: a deep-research agent producing vendor trust reports with "transparent trust scores"; it shipped a live URL plus a "Benchmark Harness" | [Albin Jaldevik, LinkedIn](https://www.linkedin.com/posts/albin-jaldevik_we-won-1st-place-in-the-withsecure-track-activity-7396637308550266880-5cl8); [Alex Choi, LinkedIn](https://www.linkedin.com/posts/alexchoi1_insane-weekend-in-finland-with-the-best-team-activity-7396639095944523776-WvkU); [README](https://github.com/alexechoi/aegis). Competing teams in the same track also pushed "confidence level" and "transparent scoring with rationale" ([SISU, n8n 2nd](https://www.linkedin.com/posts/praewpiwiw_junction2025-ai-cybersecurity-activity-7396645368442220546-CjfH); [Shaheryar M.](https://www.linkedin.com/posts/mshaheryarmalik_junction2025-hackathon-withsecure-activity-7395854555827154944-giYv)). So a confidence score plus an explanation was table stakes, not a differentiator. |
| 2025 Wolt track | **Wolt Cravings**: share a social video, get dishes to order. They used a **2-minute video** | [Yehor Tereshchenko, LinkedIn](https://www.linkedin.com/posts/yehor-tere_we-won-1st-place-in-junction-2025-main-event-activity-7395927289315856385-5cZy) |
| 2025 P&G track | CampaignClickr. The non-technical member "handled the pitch prep and presentation so the judges could clearly see the value" | [Jack Garvey, LinkedIn](https://www.linkedin.com/posts/jack-garvey-9a73a5327_hackathons-junction2025-campaignclickr-activity-7396546726683279361-XezD) |
| 2025 Aalto track | ApprenticeCircle+. It "stood out among the almost 30… submissions" for focus plus "thoughtful implementation… into the prototype" | [Aalto news](https://www.aalto.fi/en/news/aalto-students-and-researchers-shape-together-how-we-age-with-ai-at-junction-hackathon-2025) |
| 2024 overall | **BREAKDANCE** (Frank Sandqvist, solo, KONE/Granlund track). The on-stage pitch opened with a **music video**, ended with "Scan the QR code and help us inventory the whole cable factory building", then gave a 1-minute technical walkthrough | [LinkedIn post with transcript](https://www.linkedin.com/posts/frank-sandqvist-9a096683_its-not-every-day-that-you-get-to-bring-activity-7261815465537064960-3ARz) |
| 2023 overall | Driving Change: "In the Blink for Safety" | [Wikipedia](https://en.wikipedia.org/wiki/Junction_(hackathon)). Demo format UNVERIFIED. |
| 2022 overall | **Bitter Sweet**, "Haru" AI picture diary. Their keys to winning: blunt mutual critique, and scrapping an idea after hours "because it wasn't fun" | [Junction Asia interview, Medium](https://medium.com/@junctionasia_2023/we-are-the-winner-of-junction-fd91c3aa475b); [slides](https://speakerdeck.com/codertimo/haru-diary-app-with-stable-diffusion-junction-2022) |
| 2022 track (Web3) | DAO Guard / "Proof of Life". A **2-minute submission video** took 10+ hours. The sponsor's stated reason: "presentation's quality, communications, and their way of taking feedback" | [Lemon King, Medium](https://thelemonking.medium.com/5-tips-to-win-at-worlds-biggest-hackathons-39c28551fa60) |

**Pattern (my interpretation, not a published rubric):** overall winners made something you could *see move*, or that the audience could *use* on the spot. Partner-track winners matched the partner's stated problem and iterated with the partner's mentors. Video was the submission artifact in 2022 and in some 2025 tracks. The finals were live pitches.

---

## C. Hackathon projects that demoed bot, fake-account or sybil detection

| # | Project / event | What was demoed | Result (from the page) |
|---|---|---|---|
| 1 | **0xSybil**, ETHGlobal Paris | Clustered 179 voters of a known sybil-attacked Talent Protocol snapshot by cross-chain tx similarity. "While manually were found 38 fake accounts, we identified more than 50 highly suspicious and connected accounts." Flags were written to EAS, and flagged users could **appeal** with World ID | **Winner:** Worldcoin "Best use of Sign in with Worldcoin" and EAS 🥈 ([showcase](https://ethglobal.com/showcase/0xsybil-h9e55)) |
| 2 | **Airacle**, ETHGlobal Brussels | AI oracle that flags airdrop-farming wallets using graph analysis plus ML on Blockscout data | **Winner:** Blockscout "Best use of Blockscout" ([showcase](https://ethglobal.com/showcase/airacle-4r5o9)) |
| 3 | **DeCartography**, ETHNewYork 2022 | Wallet clustering for sybil resistance. They "pivoted away from… give a demo" and presented as research or data analysis instead | **Winner:** Optimism "Just Deploy!" pool prize ([showcase](https://ethglobal.com/showcase/decartography-u9prm)) |
| 4 | **SybilGuard AI**, ETHGlobal Singapore | Upload wallets, get a graph/ML risk score | No prize listed ([showcase](https://ethglobal.com/showcase/sybilguard-ai-u3rz6)) |
| 5 | **Guard**, uOttaHack 8 (Jan 2026) | Edge bot guard: 8+ heuristics plus an optional PyTorch classifier, actions allow/challenge/throttle/block/reroute, hCaptcha challenge, dashboard with "bot score distributions", Sentry traces with "decision reasoning" breadcrumbs. "Explainable AI" is listed under *What's next* | **Winner:** Vercel sponsor category "Smart Edge Load Balancer with AI Guard" ([Devpost](https://devpost.com/software/guard-z42uyb)). No precision or FP numbers are reported on the page. |
| 6 | **Spam Slayer**, HopHacks Fall 2019 | Paste an Amazon URL; a CNN filters suspected fake reviews and recomputes the rating | **Winner:** 3rd Place Overall, Top 10 Finalists ([Devpost](https://devpost.com/software/spam-slayer)) |
| 7 | **Not My Type**, R.U. Hacking? 2019 | "Mash the keyboard" keystroke-timing bot check. They built a Malduino bot to attack themselves, and admitted "it's fairly easy to write a b[ot]…" | **Winner:** 3rd Place ([Devpost](https://devpost.com/software/not-my-type)) |
| 8 | **Dead Internet Theory**, Cal Hacks 13.0 (Oct 2026) | Extension that scores astroturfing in Reddit/X/TikTok/YouTube threads: 0–100 suspicion index, evidence linked to comments, "not a definitive probability". *What's next*: "evaluate our detection methods against labeled examples to better understand false positives" | **Winner:** only "[MLH] Best Domain Name from GoDaddy Registry" ([Devpost](https://devpost.com/software/dead-internet-theory)) |
| 9 | **CAPTCHA Your Attention**, PickHacks 2024 | Catch-the-apple browser game as a CAPTCHA | **Winner:** only "Best .Tech Domain Name" ([Devpost](https://devpost.com/software/captcha-2)) |
| 10 | Non-winners (lookalikes): **AstroGuard** (HooHacks 2026, Reddit bot-likelihood extension, 20+ heuristics plus Gemini); **FLUX** (Reddit Mod Tools Hackathon 2026, 11-signal weighted "confidence" plus account graph); **Inpuri Anti-Bot** (DubHacks '23, mouse/UA ML); **Playproof** (NexHacks, movement-feature game CAPTCHA); **FraudFortress** (Hack SoDA 2024, e-commerce bot behaviour plus AWS Fraud Detector); **ghost.reviews** (DeveloperWeek NY 2026, explainable fake-review "Ghost Score", bundled sample data "so judges always get a fast, complete report") | No winner badge on the pages: [AstroGuard](https://devpost.com/software/astroguard), [FLUX](https://devpost.com/software/jhgayesgfyjg), [Inpuri](https://devpost.com/software/inpuri-anti-bot), [Playproof](https://devpost.com/software/playproof), [FraudFortress](https://devpost.com/software/teamxhack-fraudfortress), [ghost.reviews](https://devpost.com/software/ghost-reviews-0lm372) |
| 11 | Outside hackathons: **Shepherd**, Columbia SPS Greater Good Challenge (pitch competition, 10 teams) | LLM-based browser extension scoring bot-likeness on X and Reddit | **1st place** ([Columbia SPS, 10 Feb 2025](https://sps.columbia.edu/news/ai-powered-bot-detection-tool-shepherd-wins-years-greater-good-challenge)) |

**What this shows (skeptical read):**
- "Score accounts or visitors with heuristics plus an LLM, and show a confidence number" is a crowded pattern at hackathons, and it mostly doesn't win overall.
- The only projects in this list that touched ground truth (0xSybil's known attack, and Not My Type attacking itself) are also among the more credible winners. No winning page I found reported precision, FPR or calibration.
- That gap is your opening. It is also a warning: judges have seen many "bot score" dashboards.
- Selection caveat: Devpost search surfaces what it indexes, so non-winning entries are under-sampled and winners may be missed. This is not a systematic census (UNVERIFIED completeness).

---

## Lessons for the demo (each tied to evidence above)

- **Optimise for Hostinger's judges first, peers second.** Track winners are chosen by the partner against its own criteria ([2026 site](https://2026.hackjunction.com/)). Get Hostinger's mentors to confirm on Friday night what labelled data they'll score you on. The 2022 Web3 track winner was praised for "their way of taking feedback" ([Lemon King](https://thelemonking.medium.com/5-tips-to-win-at-worlds-biggest-hackathons-39c28551fa60)).
- **Let the reviewer be the attacker.** BREAKDANCE got the audience to scan a QR code and use it ([transcript](https://www.linkedin.com/posts/frank-sandqvist-9a096683_its-not-every-day-that-you-get-to-bring-activity-7261815465537064960-3ARz)). DroneStalker made peers "shoot down the drone" ([Legg](https://www.linkedin.com/posts/404missinglink_we-are-junction-2025-finalists-jacob-kenney-activity-7396593086904086528-M9j6)). For a bot guard, the equivalent is a QR code to your site on the judge's phone: they get "human, 0.9x, because…", then you hit the site with a scripted bot and the verdict flips on screen.
- **Win the side-by-side comparison in the first 10 seconds.** With face-to-face pairwise review, you are compared to the previous table ([Junction blog](https://blog.hackjunction.com/forget-the-judges-participants-to-choose-the-winner-at-junction-5af699e1fc19)). One big live number (current FPR on held-out humans) beats a feature tour.
- **Show ground truth, not just a score.** 0xSybil anchored on a known attack with 38 manually labelled fakes ([showcase](https://ethglobal.com/showcase/0xsybil-h9e55)). Dead Internet Theory admitted it had no labelled eval and won only a domain prize ([Devpost](https://devpost.com/software/dead-internet-theory)). Put a confusion matrix from held-out labelled data on screen.
- **A confidence score plus an explanation alone won't differentiate you.** Several 2025 WithSecure entries already shipped "confidence level" and "rationale" ([SISU](https://www.linkedin.com/posts/praewpiwiw_junction2025-ai-cybersecurity-activity-7396645368442220546-CjfH), [Shaheryar M.](https://www.linkedin.com/posts/mshaheryarmalik_junction2025-hackathon-withsecure-activity-7395854555827154944-giYv)). Guard listed explainability only as future work ([Devpost](https://devpost.com/software/guard-z42uyb)). The differentiator has to be *calibration evidence* (a reliability plot) and an honest live FPR.
- **Expect a judge to try to break it, and pre-empt it.** Daksh Raghuvanshi looks for work "whose reward holds when someone tries to break it" ([2026 site](https://2026.hackjunction.com/)). Not My Type openly said its detector was easy to fool ([Devpost](https://devpost.com/software/not-my-type)). Show a "known evasions" panel with the attacks you already tried and what got through.
- **Have a deterministic fallback.** ghost.reviews shipped bundled sample data so judges "always get a fast, complete report without network failures" ([Devpost](https://devpost.com/software/ghost-reviews-0lm372)). Also keep a recorded 2-minute video; video was the submission artifact for several Junction track winners ([Wolt](https://www.linkedin.com/posts/yehor-tere_we-won-1st-place-in-junction-2025-main-event-activity-7395927289315856385-5cZy), [Lemon King](https://thelemonking.medium.com/5-tips-to-win-at-worlds-biggest-hackathons-39c28551fa60)).
- **Frame it around a problem, not a feature.** This is Kanninen's and Söderlund's stated filter, and Hostinger's keynote theme ([2026 site](https://2026.hackjunction.com/)). "Small site owners can't tell if their reviews are sockpuppets" is a problem. "Behavioural fingerprinting" is a feature.
- **Argument against the sockpuppet-linking angle (judgement, not sourced fact):** it needs data that joins comment accounts to traffic sessions. Unless Hostinger provides that, you would have to synthesise it, and synthetic linkage is exactly what a skeptical judge will discount. Treat it as a stretch layer on top of a solid per-visitor detector, not the core.

## UNVERIFIED / gaps

- The Hostinger "Decode the Noise" challenge page, its criteria (the 40% detection-quality weight), dataset, or prize. Not public: the platform's `challenges` array is empty, and web and X searches found nothing.
- Whether 2026 peer review is Gavel pairwise or another face-to-face format (platform flags differ from 2025), and the exact 2026 review window (the site says 10:30, the platform says 14:00 Helsinki).
- Real minutes per project at Junction expos. Only my derived estimate (~10–14 min including walking).
- Demo format of WAVELET (2025) and the 2023 winner on stage. No transcript or video found. YouTube finals were not checked.
- How Junction 2025 partner judges scored. No partner published a rubric that I found.
- DroneStalker's "50.3% better than the FRED baseline" is self-reported.
- Devpost/ETHGlobal coverage is search-based, not exhaustive. No MLH-specific bot-detection *overall* winner was found.
- Hostinger AI Audit and other prior-art products were not covered (question B, assigned elsewhere).

---

# Section B: Prior art (CrowdSec, Anubis, Cloudflare, DataDome, Wordfence, Hostinger, open source)

# B. Prior art: "explainable bot detection with a live false-positive rate" for website owners

*Research for the Junction 2026 Hostinger "Decode the Noise" decision. Compiled Fri 13 Nov 2026, ~21:30 CET (Europe/Prague). Sources are public web only. Primary docs were preferred (vendor docs, GitHub READMEs, changelogs). Anything I could not confirm from a primary page is marked **UNVERIFIED**. The "does NOT do" statements below mean "not documented in the pages I read". That is absence of evidence, not proof.*

## TL;DR (skeptical read)

- **Per-request explanations are not novel.** CrowdSec's open-source bot detection (alpha, 2026) already lists the signals that fired per request (`score_reasons`, e.g. `cdp=100,utc_timezone=15`) ([CrowdSec signals](https://docs.crowdsec.net/docs/appsec/bot_detection/signals/), [troubleshooting](https://docs.crowdsec.net/u/troubleshooting/bot_detection/)). Several small 2026 open-source projects advertise "full signal trace per request" ([StyloBot](https://github.com/johnspencentu/stylobot)) and per-request inspectors ([GroundShade](https://groundshade.com/)).
- **"Live FP rate" also exists as a proxy.** Cloudflare's Challenge Solve Rate is documented as indicating "the false positive percentage of a rule" ([CF CSR](https://developers.cloudflare.com/cloudflare-challenges/reference/challenge-solve-rate/)). DataDome shows Sankey funnels of challenge outcomes ([DataDome responses](https://docs.datadome.co/docs/analyze-responses.md)). CrowdSec has a Requested→Submitted→Solved/Rejected funnel ([CrowdSec troubleshooting](https://docs.crowdsec.net/u/troubleshooting/bot_detection/)). Two hobby projects measure human FPR against labelled panels: [FCaptcha](https://github.com/WebDecoy/FCaptcha) enforces an FP budget in CI, and [Kitsune](https://github.com/datascry/kitsune) has a `task calibrate` FP gate.
- **No calibrated probability found anywhere.** Cloudflare's 1–99 score is "the predicted probability that a client is human, mapped to the final 1–99 bot score", but heuristics hard-set 1 or 29 ([CF bot score](https://developers.cloudflare.com/bots/concepts/bot-score/)). I found no reliability diagram, ECE or Brier score published by any vendor or project.
- **Sockpuppet / entity linking has prior art too.** StyloBot does "anonymous entity resolution" and Leiden clustering of bot networks ([StyloBot](https://github.com/johnspencentu/stylobot)). Kitsune ships a "coordination/fleet detector" ([Kitsune](https://github.com/datascry/kitsune)).
- **The defensible gap is narrower than the pitch:** a **calibrated** per-visitor probability plus a **ground-truth-anchored** live FPR with an uncertainty interval, in one CPU-only self-hosted box. This is also exactly what the 40% detection-quality criterion rewards.

---

## 1. Commercial / hosted

### Cloudflare Bot Management, Bot Analytics, Turnstile
- **Verdict per request:** yes. Every request gets a bot score 1–99; 0 = not computed ([bot score](https://developers.cloudflare.com/bots/concepts/bot-score/)).
- **Score type:** the ML engine's "Output variable (Y): the predicted probability that a client is human, mapped to the final 1–99 bot score". The Heuristics engine assigns 1 for deterministic detections and "occasionally… 29" ([bot score](https://developers.cloudflare.com/bots/concepts/bot-score/)). So the score is probability-derived, but fixed heuristic values are mixed in and no calibration claim is made. Read it as a **rank**. Cloudflare also warns that "bot scores can shift as the model learns, so periodically review rules" (same page).
- **Access:** granular scores are Enterprise + Bot Management only. Pro and above see only buckets: Automated (1), Likely automated (2–29), Likely human (30–99), Verified bot (same page).
- **Explanation (owner):** **Detection IDs** are static heuristic rules "with no overlap with human traffic", plus detection tags such as `go`. They are visible in Bot/Security Analytics and Logpush, Enterprise only ([detection IDs](https://developers.cloudflare.com/bots/additional-configurations/detection-ids/)). Bot Analytics shows score distribution and "bot score source" (which engine), on **1–10% sampled** data ([Bot Analytics](https://developers.cloudflare.com/bots/bot-analytics/)). JA3/JA4 plus JA4 "Signals" ratios are Enterprise only ([JA3/JA4](https://developers.cloudflare.com/bots/additional-configurations/ja3-ja4-fingerprint/)).
- **Explanation (visitor):** none documented. Turnstile `siteverify` returns `success`, `error-codes`, `action`, `hostname` and similar fields, with no score and no reason ([Turnstile validate](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/)).
- **FP metric / loop:**
  - **CSR** = solved / issued challenges, which "indicates the false positive percentage of a rule". It is shown per rule in Security rules ([CSR](https://developers.cloudflare.com/cloudflare-challenges/reference/challenge-solve-rate/)). Caveat from the same page: abandonments count as failures, so CSR measures who solves a challenge. It is not ground truth.
  - **Bot Feedback Loop:** you report FPs/FNs, and Cloudflare "uses it to train the next machine learning model". Enterprise only ([feedback loop](https://developers.cloudflare.com/bots/concepts/feedback-loop/)).
  - The docs also name a known FP trigger: a missing User-Agent gives score 1, which hits corporate proxies and WARP ([bot score](https://developers.cloudflare.com/bots/concepts/bot-score/)).
- **Verified bots:** an allowlist based on Web Bot Auth signatures, IP lists or reverse DNS, with a behaviour taxonomy ([verified bots](https://developers.cloudflare.com/bots/concepts/bot/verified-bots/)).
- **Self-hosted / CPU:** no. It is an edge SaaS.

### DataDome
- **Verdict per request:** yes. Requests are matched to named **detection models** in four families: Signature, Behavioral, Reputational, Vulnerability Scanner. Each model has a policy: Device Check, Slider or Block ([Threats Detection](https://docs.datadome.co/docs/threat-detection)).
- **Explanation (owner):** the model or rule name is the reason. The Explore view lists the last 100–1000 requests with Date, URL, UA, **Rule Name**, IP and Response ([Explore](https://docs.datadome.co/docs/how-to-explore-your-data.md)). Integrations expose `X-DataDome-isbot` and `X-DataDome-botname` headers ([Cloudflare worker integration](https://docs.datadome.co/docs/cloudflare-worker-wrangler)).
- **Score:** Bot Protect pages don't expose a per-request numeric score. **Account Protect** returns `reasons` (e.g. `brute_force`, `teleportation`) and an integer `score`, described as "the level of confidence when identifying a request as coming from a fraudster". Nothing says it is calibrated ([Account Protect login SDK](https://docs.datadome.co/docs/account-protect-login.md)).
- **FP metric:** no explicit FPR. The **Responses** page shows Sankey flows for Slider and Device Check: JS executed → completed → validated / blocked ([Analyze responses](https://docs.datadome.co/docs/analyze-responses.md)). This works as a de facto FP proxy, similar to CSR. You can deactivate a model and preview it with "Explore" before turning it back on ([Threats Detection](https://docs.datadome.co/docs/threat-detection)).
- **Self-hosted / CPU:** no. Detection runs in the DataDome cloud and the module calls their API (timeout default 300 ms) ([integration](https://docs.datadome.co/docs/cloudflare-worker-dashboard)).

### Wordfence (WordPress plugin)
- **Verdict:** yes. The firewall, rate limiting and IP blocklist decide per request ([WP.org readme](https://plugins.svn.wordpress.org/wordfence/trunk/readme.txt)).
- **Explanation (owner):** Live Traffic shows IP, location, UA and blocked requests. You can filter it by "Blocked by Firewall" ([Live Traffic](https://www.wordfence.com/help/tools/live-traffic/), [Learning Mode](https://www.wordfence.com/help/firewall/learning-mode/)). The WAF names attack types, e.g. "Directory Traversal" (Learning Mode page).
- **Explanation (visitor):** a block message. Logged-in admins get a button to allowlist the blocked action ([Learning Mode](https://www.wordfence.com/help/firewall/learning-mode/)).
- **Score:** none documented. The decision is rule-based.
- **FP loop:** **Learning Mode** (default 7 days) auto-allowlists requests that look like attacks, and blocked params can later be allowlisted. Wordfence explicitly calls these "false positives" ([Learning Mode](https://www.wordfence.com/help/firewall/learning-mode/)). There is no FP *rate*.
- **Self-hosted / CPU:** yes, inside WordPress/PHP. The real-time IP blocklist is Premium (readme).

### Hostinger: "AI Audit", CDN bot controls, VPS security
- **AI Audit** is a CDN feature for **web hosting and cloud hosting** clients. It is found under Performance → CDN → AI Audit. It lets you "see which AI crawlers visit your site", "track crawler activity" and block individual bots ([Hostinger blog](https://www.hostinger.com/blog/cdn-ai-audit/), [changelog](https://www.hostinger.com/changelog/ai-audit)). Each AI crawler has a Block switch, and request counts cover the last 24 h ([Hostinger tutorial](https://www.hostinger.com/tutorials/how-to-rank-in-chatgpt/)). It is a per-crawler allow/block list with counts. It has no score, no explanation and no FP metric.
- **CDN security levels / Under Attack mode:** the CDN has security levels. Medium and Low challenge suspicious visitors while letting verified Googlebot/Bingbot through, and Under Attack challenges everyone ([CDN security levels](https://www.hostinger.com/support/10695636-hostinger-cdn-security-levels/), [Under Attack](https://www.hostinger.com/support/8512979-hostinger-cdn-the-under-attack-mode/), [verification page](https://www.hostinger.com/support/hostinger-cdn-the-browser-verification-page/)). *The exact level semantics are summarised from search snippets; the page body did not render for me (**UNVERIFIED** detail).*
- A third-party investigation claims Hostinger's CDN challenged AI-crawler IP ranges even with AI Audit set to Allow and Security Level "Essentially Off". If true, that would mean some decisions are invisible to the owner ([Stonegate](https://stonegatewebsecurity.com/articles/hostinger-bot-protection-blocking-ai-crawlers/)). **UNVERIFIED:** this is a secondary source and Hostinger has not confirmed it. It is still a useful pitch hook: "owners can't see why traffic was blocked."
- **VPS (the challenge platform):**
  - **Managed cloud firewall:** L3/L4 accept/reject/drop by protocol, port and source, filtered before traffic reaches the VPS. The docs say it is "not intended to block the large volumes of traffic" of DDoS ([VPS firewall](https://www.hostinger.com/support/8172641-how-to-use-a-managed-vps-firewall-at-hostinger/)).
  - **Monarx malware scanner:** file-based only, with no database scanning. Auto-cleanup needs a paid license ([malware scanner](https://www.hostinger.com/support/8450363-hostinger-vps-malware-scanner/)).
  - **Anti-DDoS Events:** filtering or null-route events. For repeated attacks Hostinger recommends putting a CDN, reverse proxy or **WAF** in front ([Anti-DDoS events](https://www.hostinger.com/support/what-are-anti-ddos-events-in-hostinger-vps/)).
  - **Docker Catalog:** the Security category (39 apps) is identity, secrets and VPN tools. I found **no CrowdSec, WAF or bot-detection app** in it when I fetched it ([Docker Catalog](https://www.hostinger.com/support/hostinger-docker-catalog-applications/)). A Docker VPS template exists (Ubuntu 24.04 + docker-compose) ([Docker template](https://www.hostinger.com/support/8306612-how-to-use-the-docker-vps-template-at-hostinger/)).
  - **So:** Hostinger VPS has **no L7 bot detection** built in. This is the gap the challenge targets, and the pitch can say it truthfully.

## 2. Self-hosted / open source

### CrowdSec (Security Engine, AppSec WAF, Console, and the new Bot Detection)
- **Architecture:** the Log Processor matches logs against **scenarios** and HTTP requests against **AppSec rules**, then raises **alerts**. LAPI profiles turn alerts into **decisions** (`ban`, `captcha`), and bouncers enforce them ([Concepts](https://docs.crowdsec.net/docs/next/concepts)).
- **Verdict:** per IP via decisions. AppSec in-band rules also return an action for the current request ([AppSec intro](https://docs.crowdsec.net/docs/next/appsec/intro)).
- **Explanation (owner):** the scenario name, plus configurable **Alert Context** (target_uri, UA, method, status) that is visible in `cscli` and the Console ([Alert Context](https://docs.crowdsec.net/docs/next/log_processor/alert_context/intro)).
- **New Bot Detection (alpha, CrowdSec ≥1.8.2):** a browser PoW plus the fpscanner device fingerprint. "Every verdict is a CrowdSec signal" ([intro](https://docs.crowdsec.net/docs/appsec/bot_detection/intro/), [how it works](https://docs.crowdsec.net/docs/appsec/bot_detection/how_it_works/)).
  - Signals add weighted points to a request score. **Alerts list the signals that fired in `score_reasons`** ([signals](https://docs.crowdsec.net/docs/appsec/bot_detection/signals/)).
  - New signals ship as **experimental, weight 0**, so they are measured before they can block (same page).
  - Owners debug rejections of real visitors with `cscli alerts inspect`. `cscli metrics show bot-detection` gives a challenge funnel (Requested / Submitted / Solved / Rejected / Exempt) ([troubleshooting](https://docs.crowdsec.net/u/troubleshooting/bot_detection/)).
  - Known good bots are matched by IP range or forward-confirmed rDNS, and `cscli bot match` tests a request ([cscli bot match](https://docs.crowdsec.net/docs/cscli/cscli_bot_match/)).
- **Explanation (visitor):** "Verification rejected" or "Cookies are disabled" pages, with no reason shown ([troubleshooting](https://docs.crowdsec.net/u/troubleshooting/bot_detection/)).
- **Score / calibration:** additive weights against a bundle threshold (strict / default / permissive). The docs advise switching bundle when "several weak signals barely reach the threshold" (same page). It is **not a probability**. CTI reputation scores run 0–5 (aggressiveness, threat, trust, anomaly) ([CTI object](https://docs.crowdsec.net/u/cti_api/taxonomy/cti_object)).
- **FP handling:** allowlists, `false_positives` tags on CTI IPs ("shouldn't be considered as malicious"), community-blocklist entries "manually purged (ie. false positive)" ([CTI object](https://docs.crowdsec.net/u/cti_api/taxonomy/cti_object)). Console Remediation Metrics show blocked volume and "Malicious Intents", but CrowdSec itself says intent "is an estimate inferred from broader network observations" ([Remediation Metrics](https://docs.crowdsec.net/u/console/remediation_metrics)). There is **no FP rate**.
- **CPU:** yes, self-hosted. Bot detection needs amd64 with SSE4.1 (or arm64) plus executable-memory mapping for the WASM runtime ([intro](https://docs.crowdsec.net/docs/appsec/bot_detection/intro/)). Whether Hostinger VPS CPUs expose SSE4.1 is **UNVERIFIED**, though it is likely on any modern x86.

### Anubis (TecharoHQ)
- **What it is:** a "Web AI Firewall Utility". Its README says it "is a bit of a nuclear response" that may block "good bots" like the Internet Archive ([README](https://github.com/TecharoHQ/anubis)).
- **Verdict:** ALLOW / CHALLENGE / DENY / WEIGH per request ([policies](https://raw.githubusercontent.com/TecharoHQ/anubis/main/docs/docs/admin/policies.mdx)).
- **Score:** **request weight**, an additive "custom level of suspicion" mapped to actions by CEL thresholds ([thresholds](https://raw.githubusercontent.com/TecharoHQ/anubis/main/docs/docs/admin/configuration/thresholds.mdx)). It is not a probability.
- **Explanation (owner/downstream):** `X-Anubis-Rule`, `X-Anubis-Action` and `X-Anubis-Status` headers, rule names in Prometheus metrics, verbose JSON logs, and "Log weight when issuing challenge" ([policies](https://raw.githubusercontent.com/TecharoHQ/anubis/main/docs/docs/admin/policies.mdx), [CHANGELOG](https://raw.githubusercontent.com/TecharoHQ/anubis/main/docs/docs/CHANGELOG.md)).
- **Explanation (visitor):** DENY deliberately "send[s] back an error message that scrapers think is a success" (policies). This is the opposite of explaining.
- **FP:** the docs flag one sample rule as "known to have a high false positive rate in testing" (policies). There is **no FP metric**.
- **CPU:** yes. It is written in Go, and v1.28.x is current per the changelog.

### go-away
- Self-hosted rule engine with CEL rules over IP/network, headers, UA, path and **JA3N/JA4** fingerprints. It can PASS / CHALLENGE / DENY and run transparent, non-JS or JS/WASM challenges. It describes itself as using "conventional non-nuclear options" ([README](https://git.gammaspectra.live/git/go-away/raw/branch/master/README.md)). The rule name works as the explanation. It has no score and no FP metric.

### iocaine and Nepenthes (tarpits / poisoners)
- **iocaine:** puts unwanted crawlers in "an infinite maze of garbage". Its scripting engine decides per request, and poisoned URLs identify crawlers that come back through real browsers ([README](https://git.madhouse-project.org/iocaine/iocaine/raw/branch/main/README.md)).
- **Nepenthes:** "DELIBERATELY MALICIOUS SOFTWARE". It warns that "ANY SITE THIS SOFTWARE IS APPLIED TO WILL LIKELY DISAPPEAR FROM ALL SEARCH RESULTS" and that it can cause high CPU ([zadzmo.org](https://zadzmo.org/code/nepenthes/)).
- Neither offers confidence or FP measurement.

### BotD (FingerprintJS)
- **Output:** client-side, returns `{bot: true, botKind}` or `{bot: false}`, with no score ([API](https://raw.githubusercontent.com/fingerprintjs/BotD/main/docs/api.md)).
- **Status:** the roadmap is "stability-only". The README pitches the paid Fingerprint API "to detect sophisticated bots with confidence" ([README](https://raw.githubusercontent.com/fingerprintjs/BotD/main/README.md)).
- **Weakness:** the client decides, so the result is spoofable. **UNVERIFIED** as a documented claim, but inherent to the design.

### open-appsec
- **How it decides:** a supervised ML model, then an unsupervised per-site model that creates "a final confidence score that determines whether the request should be allowed or blocked". The basic model is Apache-2.0; the advanced model sits under an ML Model license ([README](https://raw.githubusercontent.com/openappsec/openappsec/main/README.md)).
- **Scope:** an attack WAF (OWASP Top 10), not a bot classifier. The confidence score is not documented as calibrated.

### SafeLine (Chaitin) and Coraza + OWASP CRS
- **SafeLine:** a self-hosted WAF with an anti-bot challenge. Its README publishes a vendor benchmark of 33,669 samples, with Detection 71.65% / **False Positive 0.07%** in "Balance" mode, against ModSecurity L1 at 17.58% FP ([README](https://raw.githubusercontent.com/chaitin/SafeLine/main/README.md)). This is the **only "FP rate" number I found in an OSS README**. It is a static, vendor-run benchmark of attack payloads, not bots, and not a live metric.
- **CRS (used by Coraza):** **anomaly scoring**. Matched rules log rule ID, data and URI, and add to a running score that is compared to a threshold ([CRS anomaly scoring](https://coreruleset.org/docs/2-how-crs-works/2-1-anomaly_scoring/), [Coraza](https://raw.githubusercontent.com/corazawaf/coraza/main/README.md)). Each matched rule ID is an explanation. The score is a sum, not a probability.

### 2025–2026 small projects (direct competitors to a hackathon build)
| Project | What it claims (from its README) |
|---|---|
| [StyloBot](https://github.com/johnspencentu/stylobot) (2026) | 49 detectors, "~150µs per request", "full signal trace per request", probability badges (Bot/Suspicious/Uncertain/Human), entity resolution across IP/UA/TLS rotation, Leiden clustering of bot networks, and an observe-first mode that shows "what it *would* have acted on and why". |
| [Kitsune](https://github.com/datascry/kitsune) (2026) | Explainable noisy-or verdict where "every point of bot-likelihood traces back" to evidence, a red-team evader fleet, `task calibrate` to "measure the false-positive rate against real browser fingerprints", and a coordination/fleet detector. |
| [FCaptcha](https://github.com/WebDecoy/FCaptcha) | Per-signal **FP budget enforced in CI** across 14 human personas. It warns that "0% false-positive rate… means no sample in this corpus crossed the threshold — not 0% of real users would." |
| [Thunderhead](https://github.com/trialogue/thunderhead) | Passive Go proxy that scores 0–100 from 5 named weighted signals, with a dry-run mode and JSON-logged decisions. It admits that "scoring is per-IP" and that patient bots evade it. |
| [AntiBotGW](https://github.com/tarrinho/AntiBotGW) | 36 weighted signals, a block-reason breakdown, a per-signal risk breakdown and a "Stealth Agent Hunter". |
| [GroundShade](https://groundshade.com/) | Rust proxy with a "per-request inspector that shows which gates fired, and why". |
| [humanymous](https://github.com/modootoday/humanymous) | Risk score 0–100 mapped to ALLOW/CHALLENGE/DENY. It explicitly says its synthetic baseline "is not evidence of a population-wide false-positive rate". |
| [bot-signal](https://github.com/okasi/bot-signal) (2026, MIT) | "Every flag has a name, weight, and confidence level". |

*Star counts and creation dates come from repository metadata at fetch time. I did not run or audit any of these projects; their claims are **UNVERIFIED** beyond their own READMEs.*

---

## 3. Summary table

| Product | (1) Per-request verdict | (2) Reason to owner / visitor | (3) Score: calibrated? | (4) FP metric / feedback | (5) Self-hosted CPU |
|---|---|---|---|---|---|
| Cloudflare BM | Yes | Owner: detection IDs, engine source (Ent.); visitor: none | 1–99, ML output "mapped" from P(human); heuristics fixed 1/29 → **rank** | CSR (proxy); Feedback Loop (Ent.) | No |
| Turnstile | Pass/fail token | Error codes only | None | No | No |
| DataDome | Yes | Owner: model/rule name; visitor: none | Integer "confidence" (Account Protect only); not calibrated | Challenge Sankey (proxy); model deactivation | No |
| Wordfence | Yes | Owner: Live Traffic, attack type; visitor: block msg + admin allowlist button | None | Learning Mode, allowlist; no rate | Yes (WP/PHP) |
| Hostinger AI Audit / CDN | Per crawler (UA) / security level | Counts per crawler; no reason | None | No | No (CDN; web/cloud plans, not VPS) |
| Hostinger VPS | L3/L4 firewall only | n/a | n/a | n/a | n/a (no L7 bot tool) |
| CrowdSec (+AppSec, Bot Det.) | Yes (decision + in-band) | Owner: scenario, alert context, **score_reasons**; visitor: generic | Additive weights vs threshold; CTI 0–5 | Allowlists, CTI FP tags, challenge funnel, experimental weight-0 signals; no rate | Yes |
| Anubis | Yes | Owner: rule headers, metrics, logs; visitor: fake success on DENY | Additive weight | None | Yes |
| go-away | Yes | Rule name | None | None | Yes |
| iocaine / Nepenthes | Route to maze | Minimal | None | None | Yes (Nepenthes CPU-heavy) |
| BotD | Client-side bool + kind | botKind | None | None | Client JS |
| open-appsec | Yes | Not documented | "Confidence score", not calibrated | Learning period | Yes |
| SafeLine | Yes | Attack type | Not documented | Static vendor benchmark (0.07% FP) | Yes |
| Coraza + CRS | Yes | Rule IDs + msg | Anomaly sum | Paranoia levels / exclusions | Yes |
| StyloBot / Kitsune / FCaptcha etc. | Yes | Signal traces | Weighted / noisy-or; no calibration evidence | Observe mode; offline FP harness / CI budget | Yes |

## 4. The gap: what none of them document doing

1. **Calibrated probability.** Nobody publishes reliability curves, ECE or Brier scores, or says "a 0.8 means 80% of such visitors are bots on your site". Cloudflare comes closest by mapping ML probability to 1–99, but mixes in fixed heuristic values and only exposes buckets below Enterprise ([bot score](https://developers.cloudflare.com/bots/concepts/bot-score/)). Everything else is additive weights or rule matches.
2. **A live FPR anchored to ground truth, with uncertainty.** Live FP signals are all **proxies**: CSR, challenge funnels, owner-reported FPs. Measured FPRs (SafeLine, FCaptcha, Kitsune) are **offline**, on fixed corpora, and the authors themselves warn against generalising. No tool computes a running FPR on the live stream against a trusted human set with a confidence interval. Examples of such a set: authenticated or returning users, canary/known-good sessions, a held-out labelled replay.
3. **Explanation to the visitor plus appeal.** Visitors get generic pages ("Verification rejected"), deliberately fake success (Anubis DENY), or nothing. Only Wordfence gives an allowlist button, and only to logged-in admins.
4. **All of it in one CPU-only box for a Hostinger VPS,** which today has no L7 bot tool in its catalog.

**Where the novelty argument is weak:**
- Per-request explanations: CrowdSec `score_reasons`, StyloBot, GroundShade.
- Observe-before-block: StyloBot, JA4proxy, CrowdSec weight-0 experimental signals.
- Cross-IP identity linking / fleets: StyloBot, Kitsune.
- Sockpuppet linking of comment accounts to traffic fingerprints is close to StyloBot's entity resolution. It only becomes distinct if you show it on **labelled** data.

A judge who knows CrowdSec or Cloudflare can fairly say "this exists". The honest differentiator is **measured quality**: calibration plus a ground-truthed live FPR. That maps directly onto the 40% detection-quality weight, more than "explainability" does.

## UNVERIFIED list

- Hostinger CDN security-level semantics: summarised from search snippets because the page body didn't render.
- Stonegate's claim of hidden AI-crawler challenges on Hostinger CDN: secondary source, not confirmed by Hostinger.
- Whether Hostinger VPS CPUs expose SSE4.1 for CrowdSec bot detection.
- BotD spoofability: inferred from its client-side design, not stated in its docs.
- All claims by small 2025–2026 projects (StyloBot, Kitsune, FCaptcha, Thunderhead, AntiBotGW, GroundShade, humanymous, bot-signal): README-only, not run or audited.
- "Not documented" findings (e.g. no calibration at DataDome, no FP rate at CrowdSec): based on the doc pages read on 13 Nov 2026. Enterprise or private docs may differ.
- The SafeLine 0.07% FP figure: a vendor self-benchmark.
- Whether the Hostinger Docker Catalog adds a WAF or CrowdSec later: the catalog changes. This finding reflects the fetch on 13 Nov 2026.

---

# Section C: Datasets linking accounts and traffic, plus red-team risks and honest claims

# Decode the Noise: Q5 (joined datasets) and Q6 (red-team risks and honest eval claims)

*Researched 13 Nov 2026 (Europe/Prague), from public web sources only. Every claim links its source. Anything I couldn't confirm is marked **UNVERIFIED**. Builds on `/workspace/decode-the-noise-research.md` (Lagopoulos Zenodo 3477932, cresci, Cima) and doesn't repeat it.*

## TL;DR

- **I found no public dataset that joins account- or comment-level behaviour with web-traffic behaviour and carries sockpuppet labels.** The one corpus that had both, Kumar et al.'s Disqus trace (62.7M posts with anonymized IPs), was supplied privately by Disqus. I found no public download for it.
- **Its "ground truth" is itself an IP-and-time rule.** If you build a detector on IP co-occurrence and score it against those labels, you get back the labelling rule, not a real detection result.
- **The closest openly downloadable traffic-plus-interaction set is Iliou et al.'s Web Bot Detection Dataset.** Web logs and mouse movements are joined on `session_id`. It is CC BY-NC-SA and very small: 50 + 56 human sessions.
- **For content-side sockpuppet labels without traffic, the options are:** Wiki-Socks (GPL-3.0), Wikipedia Ban Evasion (8,551 pairs; licence not stated) and the Reddit IRA list (944 accounts; the CC0 archive is a third party's).
- **The sockpuppet-linking idea can't be measured on public data.** Its labels would have to be synthetic, made by construction on your own site, and judges can rightly discount that. It also has a built-in false-positive engine: shared IPs from CGNAT, VPN, Tor and campus Wi-Fi. Cloudflare reports that CGNAT IPs get rate-limited about 3× more often even though they carry less bot traffic.
- **What survives a red-team:** precision and FPR at a pre-declared threshold, with confidence intervals, on a held-out set split by time or site; a separate adversarial set; and a calibration plot. The state of the art loses to commercial evasive bots more than half the time (FP-Inconsistent: 52.93% evasion vs DataDome). Never claim "undetectable bots caught".

---

## Q5: Public datasets that link account or content behaviour with traffic behaviour

| Dataset | Content or account side | Traffic or interaction side | Labels | Size | Licence | Download without signup? |
|---|---|---|---|---|---|---|
| **Kumar et al. 2017, "An Army of Me" (Disqus)** | Posts and discussions across 9 communities | Anonymized IP per post, timing | 3,656 sockpuppets in 1,623 groups, **derived heuristically** | 2,897,847 users; 62,744,175 posts ([arXiv:1703.07355](https://arxiv.org/abs/1703.07355)) | The paper is CC BY 4.0. I found no data licence. | **No.** "Disqus… provided us with a complete trace" ([arXiv](https://arxiv.org/abs/1703.07355)). `snap.stanford.edu/sockpuppets/` returned 500 here. I found no public data link. **Public unavailability is UNVERIFIED** (absence of evidence). |
| **Iliou et al., Web Bot Detection Dataset** | None (no accounts or comments) | Apache-style web logs plus mouse trajectories, joined on `session_id` | human / moderate_bot / advanced_bot | Phase 1: 50 human sessions plus matched bots (train 70, test 30). Phase 2: 56 human, 28 moderate, 56 advanced. The zip is 32.2 MB (369 MB unpacked, 485 files). Counts are from the annotation files, checked here. ([m4d.iti.gr](https://m4d.iti.gr/web-bot-detection-dataset/)) | **CC BY-NC-SA** (the page links the 2.0 deed) ([m4d.iti.gr](https://m4d.iti.gr/web-bot-detection-dataset/)) | **Yes.** [Direct zip](https://m4d.iti.gr/wp-content/uploads/2024/04/web_bot_detection_dataset.zip) returned HTTP 200. |
| **Wiki-Socks** (Raszewski, ACL 2025) | Wikipedia edits: timestamp, page, edit summary | Timing only, no IP | `sock` 0/1 per contribution, grouped per investigation, with matched non-socks | Not stated in the README. 984 empty investigations are listed ([GitHub](https://github.com/lraszewski/wiki-socks)). | GPL-3.0 ([GitHub](https://github.com/lraszewski/wiki-socks)) | Yes (GitHub) |
| **Wikipedia Ban Evasion** (Niverthi, Verma, Kumar, WWW 2022) | Revisions with added and deleted text, account metadata | Timing only | 8,551 parent→child evader pairs, plus matched control pairs | About 33 GiB compressed ([GitHub](https://github.com/claws-lab/ban_evasion)) | **No licence found (UNVERIFIED)** | Yes, through the Dropbox link in the README |
| **Solorio et al. 2014, Wikipedia SPI corpus** | Talk-page text | None | 305 confirmed sock cases, 318 non-sock cases ([arXiv:1310.6772](https://arxiv.org/pdf/1310.6772)) | 623 cases | CC BY 3.0 per the paper | The original host `docsig.cis.uab.edu` returned 500 here. A derived 376-case AV corpus is on [Zenodo 20128904](https://doi.org/10.5281/zenodo.20128904). |
| **Reddit IRA accounts (2018)** | Account list. The archive adds comments and submissions | None | 944 accounts that Reddit attributed to the IRA ([Reddit post](https://www.reddit.com/r/announcements/comments/8bb85p/reddits_2017_transparency_report_and_suspect/)) | 944 | The archive is CC0 "where applicable" ([ALCC01](https://github.com/ALCC01/reddit-suspicious-accounts)). There is also a username list on [Harvard Dataverse](https://doi.org/10.7910/dvn/pmy0kf/zf1skf). | Yes. **Positive-only**, so FPR is undefined without controls. |
| **YelpChi / Amazon (CARE-GNN packaging)** | Review and user graphs with handcrafted features | None | Yelp filter flag (spam) / helpful-vote proxy | YelpChi 45,954 reviews (14.5% fraud). Amazon 11,944 users (9.5%) ([arXiv:2008.08692](https://arxiv.org/pdf/2008.08692)) | The repo is Apache-2.0. **The underlying Yelp/Amazon data rights are UNVERIFIED.** | Yes: `data/YelpChi.zip` 17,980,652 B and `Amazon.zip` 26,122,297 B ([GitHub](https://github.com/YingtongDou/CARE-GNN)) |
| **YelpNYC / YelpZip** (Rayana & Akoglu) | Reviews | None | Yelp filter | **UNVERIFIED.** I found no public link. Akoglu's [pubs page](https://www.andrew.cmu.edu/user/lakoglu/pubs.html) gives none. | UNVERIFIED | Likely on request (UNVERIFIED) |
| **Balabit Mouse Dynamics Challenge** | None | Mouse events from RDP sessions | User identity / legal vs illegal session (account takeover) | 10 users ([GitHub](https://github.com/balabit/Mouse-Dynamics-Challenge)) | None found on the repo (UNVERIFIED) | Yes |
| **BeCAPTCHA-Mouse** | None | Synthetic bot mouse trajectories | human / bot | Not checked | Licence agreement required ([GitHub](https://github.com/BiDAlab/BeCAPTCHA-Mouse)) | **No.** You email a signed agreement. |
| **Aalto 136M keystrokes** | Typed sentences | Keystroke timing | User identity, not bot or sock | 1.4 GB zip ([Aalto](https://userinterfaces.aalto.fi/136Mkeystrokes/)) | Research and non-commercial, per the search summary (**UNVERIFIED** wording) | Yes |
| **Twitter bot sets with a `source` (client) field** | Tweets and profiles | Posting-client string only, not traffic | Bot / human | See the prior file (cresci-2017 etc.) | See the prior file | Yes. **That cresci tweet CSVs contain `source` is UNVERIFIED here** (not unpacked). |

**Why no such dataset is public.** Platforms hold the traffic data, and IPs plus timing are personal data (CJEU *Breyer*, see the prior file). Wikipedia's IP-to-account linkage is restricted to CheckUsers ([CheckUser policy](https://meta.wikimedia.org/wiki/CheckUser_policy)). On wikis with temporary accounts, unregistered editors' IPs "are not visible publicly" ([MediaWiki Temporary Accounts](https://www.mediawiki.org/wiki/Trust_and_Safety_Product/Temporary_Accounts)). So the public data that once linked IPs to edits is disappearing.

**Circular labels (skeptic's note).** Kumar et al. define sockpuppets as accounts that "post from the same IP address in the same discussion in close temporal proximity at least 3 times" (T = 15 min). Before that, they remove the top 5% most-shared IPs to avoid proxies and intranets ([arXiv:1703.07355](https://arxiv.org/abs/1703.07355)). Even with the private data in hand, an "IP plus timing fingerprint" detector would be graded against its own rule. The paper also shows the rule needs heavy filtering to stay precise. That is the CGNAT problem in miniature.

**Closest usable pair (not joined).** Use (a) Iliou for traffic and mouse, (b) Lagopoulos for log-session scale (prior file), and (c) Wiki-Socks or Ban Evasion for content-side linking. Nothing links rows across them. Any join is synthetic.

### Making labelled joined data in hours (labels by construction)

1. **Run a demo site you control** (a blog with comments and reviews on the VPS) that logs request lines, a cookie/session ID, a JS fingerprint hash, and mouse and keystroke timing. The Iliou log format, with `session_id` in the log line, is a good template ([dataset zip](https://m4d.iti.gr/wp-content/uploads/2024/04/web_bot_detection_dataset.zip)).
2. **Humans:** teammates and consenting hackathon attendees, each with one account, posting naturally. Get consent and truncate IPs (see the prior file).
3. **Puppetmasters:** one person driving 3–5 accounts. Vary it across conditions: same browser, separate profiles, a VPN switch, a phone on mobile data.
4. **Bots, in rising sophistication:** curl; Playwright with default settings; Playwright with stealth ([puppeteer-extra-plugin-stealth](https://github.com/berstend/puppeteer-extra/tree/master/packages/puppeteer-extra-plugin-stealth)); undetected-chromedriver ([repo](https://github.com/ultrafunkamsterdam/undetected-chromedriver)); and an LLM agent writing comments.
5. **Split by person and time, not by row.** Hold out whole puppetmasters and whole bot tools.

**How judges will see it.** Synthetic or lab labels are a named pitfall: "lab-only evaluation", together with sampling bias and label inaccuracy, in Arp et al.'s study of 30 top-venue security ML papers ([arXiv:2010.09470](https://arxiv.org/abs/2010.09470), USENIX Security 2022). Expect "you tested on bots you wrote". Defend it by disclosing it on the slide, holding out generators the model never saw, and reporting the public Iliou or Lagopoulos numbers next to it. Note that FP-Inconsistent got its realistic evasive traffic only by *buying* it from 20 bot services ([arXiv:2406.07647](https://arxiv.org/abs/2406.07647)). A 48 h team can't replicate that.

**Sample-size reality.** With 56 human sessions (Iliou phase 2) and 0 false positives, the 95% upper bound on FPR is still about 3/56 ≈ 5.4% (rule of three, my calculation). Tiny sets can't support a "<1% FPR" claim.

---

## Q6: How judges break bot detectors in 15 minutes, and which claims survive

### Likely attacks, cheapest first

| Attack | Evidence that it works or causes FPs | What it exposes |
|---|---|---|
| **curl with copied headers**, then **curl-impersonate** | curl-impersonate performs "TLS and HTTP handshakes that are identical to that of a real browser" ([repo](https://github.com/lwthiker/curl-impersonate)) | Header-only or UA-only rules. A JA3/JA4 check beats plain curl but not curl-impersonate. |
| **Changing the UA string** | Anubis by default passes clients without "Mozilla" in the UA. Users note "it would be trivial to… change your UA" ([Anubis #95](https://github.com/TecharoHQ/anubis/issues/95)) | UA-keyed policy and **UA-derived labels** |
| **Headless or stealth Playwright/Puppeteer, undetected-chromedriver** | Stealth plugin "to prevent detection" ([repo](https://github.com/berstend/puppeteer-extra/tree/master/packages/puppeteer-extra-plugin-stealth)). Undetected-chromedriver claims it "does not trigger… DataDome / Imperva" ([README](https://github.com/ultrafunkamsterdam/undetected-chromedriver)). In FP-Crawlers, sites detecting a heavily modified crawler fell to 2.5% ([Vastel et al., MADWeb 2020](https://www.ndss-symposium.org/wp-content/uploads/2020/02/23010-paper.pdf)). | `navigator.webdriver` / headless checks |
| **Commercial "undetectable" bot traffic, residential proxies** | 20 bot services: average evasion 52.93% vs DataDome and 44.56% vs BotD. Inconsistency rules cut evasion by 48.11% and 44.95% ([FP-Inconsistent, IMC 2025](https://arxiv.org/abs/2406.07647)). | Even production vendors lose about half the time |
| **LLM browser agents (browser-use etc.)** | Browser Use says its Chromium fork plus residential IPs plus a behaviour layer "bypass all major antibot systems". It also says vendors "can detect far more than they actually block" because of false-positive cost ([blog, 2 Feb 2026](https://browser-use.com/posts/bot-detection)). **Vendor claim, UNVERIFIED.** | Fingerprint and behaviour features together |
| **Human-like or replayed mouse** | Iliou's "advanced" bots use humanlike mouse movement ([dataset page](https://m4d.iti.gr/web-bot-detection-dataset/)). Follow-up work evades detectors with GANs and deep RL ([Zenodo 5549387](https://zenodo.org/records/5549387), [7032122](https://zenodo.org/records/7032122)). | Mouse-only models |
| **Solving the challenge** | Bots' CAPTCHA accuracy is 85–100% vs 50–85% for humans, and bots are faster everywhere except image reCAPTCHA ([Searles et al., USENIX Sec 2023](https://www.usenix.org/system/files/usenixsecurity23-searles.pdf)) | "We'll just CAPTCHA the uncertain ones" |
| **Slow timing** | Rate and session features fall to bots that pause between requests. I found no single source to cite; this follows from the design (**UNVERIFIED as a quantified claim**). | Rate thresholds |

### The false-positive traps judges will spring (real humans)

- **Tor Browser:** Anubis users report a cookie error on "any site running Anubis" over Tor ([#1645](https://github.com/TecharoHQ/anubis/issues/1645), open, May 2026). Cloudflare once said 94% of Tor requests it saw were "per se malicious" ([Cloudflare 2016](https://blog.cloudflare.com/the-trouble-with-tor/)). The Tor Project disputed that IP reputation is fair to individual users ([Tor blog](https://blog.torproject.org/trouble-cloudflare/)).
- **Privacy settings:** with `privacy.resistFingerprinting`, Canvas/WebGL blockers or uBlock options, Cloudflare verification "infinitely loop[s]". That report was later fixed on Cloudflare's side ([Mozilla bug 1909961](https://bugzilla.mozilla.org/show_bug.cgi?id=1909961)). Brave users report loops as well ([Brave community](https://community.brave.app/t/brave-does-not-work-with-cloudflare/620665), not opened here, **UNVERIFIED**).
- **No-JS and accessibility:** the long Anubis no-JS thread ends with "Please just enable JavaScript" ([#95](https://github.com/TecharoHQ/anubis/issues/95)). A blind user with uBlock and strict privacy hit a click-in-circle challenge ([#1295](https://github.com/TecharoHQ/anubis/issues/1295)). The maintainer said that challenge came from the site, not Anubis.
- **CGNAT and shared IPs:** CGNAT IPs have a lower mean bot rate (7% vs 13.1%) but are "subject to rate limiting three times more often" ([Cloudflare, Oct 2025](https://blog.cloudflare.com/detecting-cgn-to-reduce-collateral-damage/)). This bears **directly on sockpuppet linking.** Several judges sharing the venue Wi-Fi and posting comments will look like one puppetmaster. That is the most likely live failure of the idea.

### Claims to make vs claims to avoid

**Make these (they survive scrutiny):**
1. "At a threshold we fixed before testing, on a **held-out** set split by time and by site or generator, precision is X [95% CI] and FPR is Y [95% CI] on N humans." Use Wilson or bootstrap intervals and show N.
2. "Recall at a fixed FPR" (for example TPR at 1% and 5% FPR), in the RAID style noted in the prior file. Give a PR curve rather than accuracy or ROC alone. Under heavy class imbalance, the base rate dominates (the base-rate fallacy, [Arp et al.](https://arxiv.org/abs/2010.09470)).
3. "Confidence is calibrated": show a reliability diagram plus ECE or Brier on the held-out set, and say which recalibration you used (temperature/Platt/isotonic). Modern models are often miscalibrated without it ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)).
4. "A separate adversarial set (stealth Playwright, curl-impersonate, an LLM agent) **not used in training**. Here is where we fail." Show the misses.
5. "We ran a UA-only and an IP-rate-only baseline. We beat them by Δ." The inappropriate-baseline pitfall is in [Arp et al.](https://arxiv.org/abs/2010.09470).
6. "The live FPR counter is an **estimate**." Say what defines the FP: humans who passed a step-up check, or a known-human canary cohort. It isn't ground truth.
7. Public-data result next to own-data result: Iliou (tiny, NC licence) and/or Lagopoulos (UA-label caveat).

**Avoid these:**
1. "99% accuracy." It's meaningless under imbalance, and labels drawn from UA lists leak into UA-adjacent features (prior file; [Lagopoulos arXiv:1711.05098](https://arxiv.org/pdf/1711.05098)).
2. "Catches undetectable or LLM bots." Commercial vendors are evaded at around 45–53% ([FP-Inconsistent](https://arxiv.org/abs/2406.07647)).
3. "Zero false positives" on fewer than ~300 humans. By the rule of three, 0/56 still allows ~5% FPR.
4. "Sockpuppets detected" scored against labels built from the same IP or timing rule (circular, as Kumar shows), or with no CGNAT/VPN handling ([Cloudflare](https://blog.cloudflare.com/detecting-cgn-to-reduce-collateral-damage/)).
5. "Calibrated" when the plot is computed on the training set or on synthetic bots alone.
6. Random row-level splits of sessions from the same person, tool or day. That's data snooping and sampling bias ([Arp et al.](https://arxiv.org/abs/2010.09470)).
7. "Privacy-safe fingerprinting" without IP truncation and consent (CJEU *Breyer*, prior file).

**Argument against the sockpuppet idea.** It adds a task that public data can't label, and synthetic data can only label weakly. A judge on shared Wi-Fi can make it fail in one move. Its useful metric, pairwise precision, rests on a handful of puppetmasters. Meanwhile 40% of the score is detection quality "on labelled data". **Whether Hostinger provides that labelled set is UNVERIFIED.** Check the challenge brief. Recommendation, for Oliver to decide: keep sockpuppet linking as a clearly labelled stretch feature ("same-device links, with shared-IP links suppressed") and stake the score on visitor-level verdicts.

---

## UNVERIFIED list

- Whether Kumar et al.'s Disqus data is publicly available anywhere (none found; the SNAP page returned 500).
- Wikipedia Ban Evasion licence; Balabit licence; Aalto 136M licence wording; YelpNYC/YelpZip access route and size.
- Underlying data rights for the CARE-GNN YelpChi/Amazon zips.
- That cresci tweet CSVs include a `source` (client) column.
- Browser Use's "bypass all major antibot systems" (vendor claim).
- Brave community loop reports (not opened).
- A quantified source for slow-timing evasion.
- Whether the Hostinger challenge supplies its own labelled dataset, and at what grain.
- Solorio corpus availability today (host 500; Zenodo derivative exists).
