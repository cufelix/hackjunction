"""Measure: if a user types a domain, does the live Bluesky firehose yield anything usable?
Counts link mentions per registrable domain, distinct accounts, and near-duplicate text.
Privacy: DIDs are hashed and never printed; no handles collected. Throwaway probe, not product code."""
import asyncio, json, re, hashlib, collections, time, sys
import websockets

URL = ("wss://jetstream2.us-east.bsky.network/subscribe"
       "?wantedCollections=app.bsky.feed.post")
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 120

URL_RE = re.compile(r'https?://([^/\s)">]+)')
# crude eTLD+1: good enough for a yield measurement
def reg_domain(host):
    host = host.lower().lstrip('.').split(':')[0]
    parts = host.split('.')
    if len(parts) < 2: return host
    two = {'co.uk','org.uk','ac.uk','com.au','co.jp','com.br','co.nz','com.tr','co.za','gov.uk'}
    if len(parts) >= 3 and '.'.join(parts[-2:]) in two:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])

def norm(t):
    t = re.sub(r'https?://\S+', '', t.lower())
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return ' '.join(t.split())

async def main():
    posts = 0
    with_link = 0
    dom_posts = collections.Counter()           # domain -> post count
    dom_accounts = collections.defaultdict(set) # domain -> hashed dids
    dom_texts = collections.defaultdict(list)   # domain -> (normtext, didhash)
    t0 = time.time()
    async with websockets.connect(URL, max_size=None) as ws:
        while time.time() - t0 < DURATION:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                break
            try: ev = json.loads(raw)
            except Exception: continue
            c = ev.get('commit') or {}
            if c.get('operation') != 'create': continue
            rec = c.get('record') or {}
            if rec.get('$type') != 'app.bsky.feed.post': continue
            posts += 1
            did = ev.get('did','')
            dh = hashlib.sha256(did.encode()).hexdigest()[:12]
            text = rec.get('text','') or ''
            hosts = set()
            for m in URL_RE.finditer(text):
                hosts.add(reg_domain(m.group(1)))
            for f in (rec.get('facets') or []):
                for feat in (f.get('features') or []):
                    u = feat.get('uri')
                    if u:
                        mm = URL_RE.match(u)
                        if mm: hosts.add(reg_domain(mm.group(1)))
            for e in (rec.get('embed') or {}).get('external', {}) and [ (rec.get('embed') or {}).get('external',{}) ] or []:
                u = e.get('uri')
                if u:
                    mm = URL_RE.match(u)
                    if mm: hosts.add(reg_domain(mm.group(1)))
            if hosts: with_link += 1
            for h in hosts:
                dom_posts[h] += 1
                dom_accounts[h].add(dh)
                dom_texts[h].append((norm(text), dh))

    el = time.time() - t0
    print(f"=== {posts} posts in {el:.0f}s ({posts/el:.1f}/s); {with_link} carried a link "
          f"({100*with_link/max(posts,1):.1f}%); {len(dom_posts)} distinct domains ===\n")
    print(f"{'domain':32} {'posts':>6} {'accts':>6}  dup-text clusters")
    for d, n in dom_posts.most_common(25):
        accts = len(dom_accounts[d])
        # near-duplicate: identical normalised text from >=2 distinct accounts
        by_text = collections.defaultdict(set)
        for t, dh in dom_texts[d]:
            if len(t) > 15: by_text[t].add(dh)
        clusters = {t: a for t, a in by_text.items() if len(a) >= 2}
        mark = ''
        if clusters:
            biggest = max(len(a) for a in clusters.values())
            mark = f"<-- {len(clusters)} cluster(s), biggest spans {biggest} accounts"
        print(f"{d:32} {n:>6} {accts:>6}  {mark}")
    # how many domains would give a user ANY result
    for k in (1,2,3,5,10):
        print(f"domains with >= {k:2} mentions in this window: {sum(1 for v in dom_posts.values() if v>=k)}")
asyncio.run(main())
