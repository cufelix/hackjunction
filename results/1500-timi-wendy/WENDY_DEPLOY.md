# 42 — Wendy deploy (organizátoři: „deploy přes wendy agent")

Stav: research z oficiálních docs (WebFetch, 14. 11.). Nic nainstalováno, nic naklonováno.
Značení: **[?]** = nepotvrzeno v docs nebo v rozporu, ověřit přímo s organizátory / na zařízení.

## 1. Co Wendy je

- Wendy Labs: CLI + agent pro nasazování kontejnerizovaných aplikací na edge zařízení (Jetson, Raspberry Pi, běžný Linux, robot).
- **wendy CLI** běží na notebooku (macOS/Linux/Windows přes WSL2). **wendy-agent** běží na cílovém zařízení, spravuje aplikace a vrací logy. Apps běží v kontejnerech (Docker/Compose projekty jsou podporované).
- Lokální vývoj nevyžaduje cloud účet. Cloud je volitelný (remote access, fleet). Offline protokol: zařízení na stejné LAN se najdou přes mDNS a komunikují šifrovaně; relay je volitelný.
- WendyOS = image s předinstalovaným agentem (RPi, Jetson). **G1 PC2 běží na Unitree Ubuntu image, ne na WendyOS** — install guide pro WendyOS se tedy nevztahuje.

Zdroje:
- https://docs.wendy.dev/latest (get started; wendy.sh/docs přesměrovává na wendy.dev → docs.wendy.dev)
- https://github.com/wendylabsinc/wendy-agent (README)
- https://github.com/wendylabsinc/wendyos (CLI + agent repo, dle search snippetu)

## 2. Jak se balí app: `wendy.json`

Povinné je jen `appId`. Relevantní pole pro nás:

| Pole | Význam pro nás |
|---|---|
| `appId` | string, povinné. Např. `hj-policy` |
| `platform` | `"linux"` (arch zařízení se odvodí), nebo explicitně `"linux/amd64"` / `"linux/arm64"` |
| `language` | volba build toolchainu (`swift`, `python`, …). U Dockerfile projektu nejisté **[?]** |
| `env` | mapa; `${VAR}` se expanduje z deploy stroje. `WENDY_*`, `LD_*`, `DYLD_*` rezervované |
| `resources` | `memory` ("32Gi" …), `cpus`, `pids` (default 4096) |
| `readiness` | `tcpSocket.port`, `timeoutSeconds` (**default 30 s**) |
| `entitlements` | vše, co není vypsáno, je pro app nedostupné; entitlementy jsou code-signed |

Relevantní entitlementy: `network` (`mode`: `host` / `bridge` / `none` / `mesh`; omitted = dnes `host`, deprecated), `http` (`port`, primární HTTP port), `gpu` (bez polí), `camera` (`allowlist` např. `["/dev/video4"]`), `persist` (`name`, `path`), `admin`, `build` (privilegované, max 1).

Zdroj: https://docs.wendy.dev/latest/reference/app-configuration/

## 3. Architektura: arm64 vs x86

- **G1 EDU PC2** = NVIDIA Jetson (model v docs nenapsán, Orin **[?]**), **arm64**, Ubuntu 20.04/22.04 arm64 image. Agent instalace: `wget -O /tmp/wendy-agent.sh https://install.wendy.dev/agent.sh`, zkontrolovat, pak `bash /tmp/wendy-agent.sh --network-access network`; ověření `sudo systemctl is-active wendy-agent`. Pozor: skript nesmí odstranit Docker nebo nahradit `containerd.io`.
- **RTX 5090 PC** = x86_64. Docs říkají, že GPU stack je NVIDIA **na Jetson i x86** (CUDA), ale konkrétní x86 agent install guide a Ubuntu verze **[?]** (dokumentace jen říká „existing Mac or Linux target"). Konkrétní GPU modely (RTX/Blackwell) nejsou uvedeny **[?]**.
- **Náš image** (`aaltonomous/docker/submission.Dockerfile`): hlavička říká `x86_64, CUDA >= 12.8 (RTX 5090 = Blackwell)`, base `nvidia/cuda:12.8.0-base-ubuntu22.04`. **Na G1 PC2 (arm64) by běžet nemusel bez arm64 buildu** — ověřit, zda base image má arm64 variantu **[?]**.
- **Rozpor v dokumentaci/handoffu:** `HANDOFF-timi-main.md` (10. 07:35) píše „base multi-arch, riziko jen wheel pin==3.3.1 aarch64, build čeká na souhlas (`binfmt`)". Dockerfile ale říká x86_64. Ověřit, co přesně je v base.
- **Rozpor v entrypointu:** `CMD ["python3", "/app/policy.py"]` (kopíruje `example_policy.py`), zadání mluví o `python3 -m aaltonomous.baselines.classical.policy_server`. Ověřit, která varianta se má nasazovat — nedělat deploy s example policy.
- Build host potřebuje **Docker Buildx** (docs: Windows/WSL2 „Docker with Buildx"); pro arm64 na x86 pravděpodobně qemu/binfmt **[?]**.

## 4. Kroky (návrh, dle docs)

1. Na notebooku: `curl -fsSL https://install.wendy.dev/cli.sh | bash` → `wendy --version`. (Search snippet uváděl `install.wendy.sh` — rozpor, použít to, co vrátí oficiální docs v den instalace.)
2. Na cílovém PC (jen pokud to organizátoři potvrdí): `curl -fsSL https://install.wendy.dev/agent.sh | bash` → `systemctl is-active wendy-agent`. Agent port nebyl v docs uveden; docs varují nepředpokládat 50051 (u „Only this computer" je 50051 jediný zmíněný port).
3. `wendy.json` do rootu projektu vedle Dockerfile (viz §5). Dockerfile ve složce projektu — **[?]** přesný název, docs říkají jen „creates a Dockerfile".
4. `wendy discover` → zapsat adresu zařízení. Pak `wendy run --device <ip>` (bez `--device` se otevře picker). `--watch` pro redeploy při změně.
5. Logy se streamují během `wendy run`; samostatný `wendy logs` v docs není **[?]**.
6. Test: z LAN `python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://<ip>:8000` (příkaz z Dockerfile hlavičky).
7. Stop: `wendy --device <ip> device apps stop hj-policy`.

## 5. Návrh `wendy.json` pro náš policy server (neověřeno na zařízení)

```json
{
  "$schema": "https://wendy.dev/schemas/wendy.json",
  "appId": "hj-policy",
  "platform": "linux/amd64",
  "version": "1.0.0",
  "readiness": { "tcpSocket": { "port": 8000 }, "timeoutSeconds": 120 },
  "resources": { "pids": 512 },
  "entitlements": [
    { "type": "network", "mode": "host" },
    { "type": "http", "port": 8000 },
    { "type": "gpu" }
  ]
}
```

Poznámky:
- `readiness.timeoutSeconds: 120` — policy musí naslouchat do 120 s (spec §8); default 30 s by to shodil.
- `platform`: `linux/amd64` pro 5090 PC; pro G1 PC2 `linux/arm64` (vyžaduje arm64 image — viz §3).
- `network: host` — policy musí číst robot DDS / kamery na LAN; docs u G1 přímo říkají, že DDS vyžaduje host network, a ten zruší izolaci kontejneru. Port 8000 pak je přímo dostupný.
- `camera` entitlement jen pokud policy čte kamery přímo (G1 `g1-rc` šablona čte `/dev/video4`, 640×480); jinak nechat být **[?]** zda má náš stack přístup k fyzickým kamerám.
- `http` s portem 8000: docs ho popisují jako „primary HTTP port" — zda wendy routuje websocket upgrade, **[?]**. Bez ověření spoléhat na `host` mode, ne na `http` routing.
- Bez `language` pole (Dockerfile projekt) **[?]**.
- Entitlement `gpu` na x86 NVIDIA: docs jen říkají, že WendyOS neexponuje GPU bez deklarace. Jak je to na běžném Linuxu s NVIDIA driverem a nvidia-container-toolkit **[?]**.

## 6. Offline a LAN

- Docs netvrdí, že cloud je povinný pro deploy. Deploy přes `--device <ip>` jde po LAN. Offline provoz lze odvodit, ale výslovné potvrzení chybí **[?]**.
- **Velký otevřený bod:** jak se image dostane na zařízení. Docs zmiňují tahání image z ghcr.io, což na venue bez internetu nejde. Jestli agent přijímá image push z CLI, nebo potřebuje registry **[?]** — zásadní otázka.
- Offline protocol (volitelný) používá mDNS (`OP_LAN=1`), port 7878 pro peer stream, HTTP front na 127.0.0.1:8080. Pro nás není nutný.

## 7. Otázky pro organizátory

1. Který stroj hostuje agent — RTX 5090 PC (x86), nebo G1 PC2 (arm64 Jetson)? Jaká Ubuntu verze, Docker verze, root přístup?
2. Mají mít týmy image předem, nebo ji nahrajeme přes wendy CLI z notebooku? Je na venue registry / internet?
3. Který port agenta a jak se dá deploy ověřit na jejich síti (wendy discover funguje na jejich switchi?).
4. Potřebuje policy přístup k DDS (host network) a kamerám? Jaké dostaneme vzorové `wendy.json`?
5. Limit velikosti image a časový limit startu (spec 120 s)?
6. Mají připravený fallback (plain `docker run --gpus all -p 8000:8000`, jak v našem Dockerfile)?
7. Kdo vlastní fyzický E-stop a gantry — deploy app, která hýbe robotem, bez fyzické bezpečnosti není možný (G1 docs).

Dotaz posílat přes Olivera (viz memory `discord-questions-via-oliver`), agent připraví text.

## 8. Rizika

| Riziko | Dopad | Mitigace |
|---|---|---|
| Arch mismatch (x86 image vs arm64 G1 PC2) | deploy padne na robotu | Pokud je cíl G1, připravit arm64 build; jinak držet `linux/amd64` |
| Start > readiness timeout (default 30 s) | app označena jako failed | `timeoutSeconds: 120` + měřit cold start s vahami |
| Image přenos bez internetu | deploy nejde vůbec | Otázka 2; připravit lokální `docker save`/load fallback **[?]** |
| CMD mismatch (example_policy vs policy_server) | špatná policy běží | Opravit CMD před buildem |
| `network: host` = žádná izolace, DDS nešifrované | bezpečnost sítě, kolize DDS | Testovat jen na izolovaném switchi |
| Docs se mění (wendy.sh → wendy.dev, install domény) | špatný install skript | Stáhnout a zkontrolovat skript v den použití |
| `wendy logs` / `deploy` nejsou v docs | debug na venue je těžký | Logovat do stdout + `wendy run` stream |
| Bez robota nelze testovat G1 část | nezachycená chyba | Simulace: `wendy tour --simulator` (Go2, ne G1) — jen pro CLI tok, ne pro G1 |
| Bezpečnost: app řídí robot bez E-stopu | úraz / poškození | Žádný motion test bez fyzického E-stopu a gantry |

## 9. Co připravit a otestovat bez robota (teď)

1. Opravit Dockerfile CMD na skutečnou policy (policy_server) a ověřit lokálně `docker run --gpus all -p 8000:8000` + `replay --synthetic 100`.
2. Změřit cold start (váhy v image) a porovnat s 120 s.
3. Buildnout `linux/amd64` (ověřit Buildx) a případně `linux/arm64` zkusit jen po schválení (viz handoff: binfmt jen na souhlas).
4. Napsat `wendy.json` výše a být připraven na `wendy run --device` (CLI nainstalovat až na pokyn, nic nespouštět bez souhlasu).

## Top 3 akce

1. Poslat organizátorům otázky 1, 2, 4 (stroj, image transfer, DDS/kamery) přes Olivera.
2. Opravit CMD na policy_server a změřit cold start do 120 s lokálně.
3. Připravit `wendy.json` (§5) a arm64 otázku rozhodnout podle odpovědi na otázku 1.

Zdroje:
- https://docs.wendy.dev/latest
- https://docs.wendy.dev/latest/reference/app-configuration/
- https://docs.wendy.dev/latest/getting-started/app/
- https://docs.wendy.dev/latest/installation/
- https://docs.wendy.dev/latest/installation/wendy-agent-unitree-g1/
- https://docs.wendy.dev/latest/hardware/gpu/
- https://docs.wendy.dev/latest/guides/offline-protocol-services/
- https://github.com/wendylabsinc/wendy-agent
