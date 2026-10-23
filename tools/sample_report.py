import json,sys,collections,re
samples=[json.loads(l) for l in open(sys.argv[1])]
if len(sys.argv) > 2:
    hitches=[(float(e)-float(ms)/1000, float(e)) for ms, e in re.findall(r"hitch: frame \d+ took (\d+) ms, ending at ([\d.]+)", open(sys.argv[2], errors='replace').read())]
    inside=lambda t: any(a <= t <= b for a, b in hitches)
    print(len(hitches), 'hitches;', sum(inside(s['t']) for s in samples), 'samples during them')
    if '--hitches' in sys.argv: samples=[s for s in samples if inside(s['t'])]
vk=collections.Counter(s['th'] for s in samples if any('gpu::vulkan' in f for f in s['f']))
cmd=vk.most_common(1)[0][0]
busy=[s for s in samples if s['th']==cmd and s['f']!=['<wait>']]
n=len(busy); print('busy',n)
def fn(x): return re.sub(r'<[^<>]*>','',re.sub(r'<[^<>]*>','',x.split('@')[0]))[:70]
def ln(x): return x.split('@')[-1] if '@' in x else ''
selfc=collections.Counter(fn(s['f'][0]) for s in busy)
print('--- self by function (inline-aware)')
for k,v in selfc.most_common(45): print(f'{100*v/n:5.1f}% {k}')
print('--- self by line')
c=collections.Counter(fn(s['f'][0])+' '+ln(s['f'][0]) for s in busy)
for k,v in c.most_common(40): print(f'{100*v/n:5.1f}% {k}')
inc=collections.Counter()
for s in busy:
    f=s['f']
    for i,x in enumerate(f):
        if fn(x)=='kirikae_rt::gpu::vulkan::Renderer::try_draw':
            inc[(ln(x), fn(f[i-1]) if i>0 else 'self')]+=1; break
print('--- inside try_draw by callsite')
for (l,callee),v in inc.most_common(30): print(f'{100*v/n:5.1f}% {l:24} {callee}')
