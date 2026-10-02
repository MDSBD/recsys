#!/usr/bin/env python3
"""Chapitre 7 : hybridation auditable, bibliothèque standard uniquement.
Données synthétiques, catalogue connu à la coupure train ; historique gelé.
Usage : python TP_Chapitre_7.py [--test] [--user U01] [--interests ia]
"""
import argparse, csv, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

def load():
    with (ROOT/'catalogue_chapitre7.csv').open(encoding='utf-8', newline='') as f:
        cat = {r['item']: r for r in csv.DictReader(f)}
    with (ROOT/'interactions_chapitre7.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len({(r['user'], r['item']) for r in rows}) == len(rows)
    assert all(r['item'] in cat for r in rows)
    return cat, rows

def normalize(scores):
    """Min-max sur les valeurs disponibles ; une source constante est désactivée."""
    vals = [s for s in scores.values() if s is not None]
    if not vals or max(vals) == min(vals):
        return {i: None for i in scores}
    lo, hi = min(vals), max(vals)
    return {i: None if s is None else (s-lo)/(hi-lo) for i,s in scores.items()}

def fuse(parts, weights):
    out = {}
    for i in parts[0]:
        denom = sum(w for p,w in zip(parts,weights) if p[i] is not None)
        out[i] = (sum(w*p[i] for p,w in zip(parts,weights) if p[i] is not None)/denom
                  if denom else 0.0)
    return out

def rrf(lists, k=60):
    if k < 0: raise ValueError('k doit être positif ou nul')
    out = defaultdict(float)
    for ranking in lists:
        if len(set(ranking)) != len(ranking): raise ValueError('doublon dans une liste')
        for rank, item in enumerate(ranking,1): out[item] += 1/(k+rank)
    return dict(out)

def ranked(scores):
    return sorted(scores, key=lambda i: (-scores[i],i))

class Hybrid:
    def __init__(self, cat, train):
        self.cat = cat
        self.hist, self.users = defaultdict(set), defaultdict(set)
        for r in train:
            self.hist[r['user']].add(r['item'])
            self.users[r['item']].add(r['user'])
        self.tags = {i:set(r['tags'].split('|')) for i,r in cat.items()}

    def components(self, user, interests=()):
        hist = self.hist.get(user,set())
        candidates = sorted(i for i,r in self.cat.items() if r['available']=='1' and i not in hist)
        profile = defaultdict(float)
        for j in hist:
            # Each item has unit-norm binary content vector.
            for t in self.tags[j]: profile[t] += 1/math.sqrt(len(self.tags[j]))
        if not hist:
            for t in interests: profile[t] += 1
        norm = math.sqrt(sum(x*x for x in profile.values()))
        content, cf, pop = {}, {}, {}
        for i in candidates:
            content[i] = (sum(profile.get(t,0) for t in self.tags[i]) /
                          (norm*math.sqrt(len(self.tags[i]))) if norm else None)
            # No CF support for an item with no train interactions: missing, not dislike.
            cf[i] = (sum(len(self.users[i]&self.users[j]) /
                         math.sqrt(len(self.users[i])*len(self.users[j])) for j in hist)/len(hist)
                     if hist and self.users.get(i) else None)
            pop[i] = len(self.users.get(i,set()))
        return content, cf, pop

    def recommend(self, user, mode='hybrid', alpha=.5, interests=(), depth=5):
        raw = self.components(user,interests)
        parts = [normalize(p) for p in raw]
        n = len(self.hist.get(user,set()))
        if mode == 'pop': scores = raw[2]
        elif mode in ('content','cf'):
            src = parts[0 if mode=='content' else 1]
            # Explicit fallback for a complete, all-user comparison.
            scores = {i:src[i] if src[i] is not None else (parts[2][i] or 0) for i in src}
        elif mode == 'rrf':
            lists = [ranked({i:s for i,s in p.items() if s is not None})[:depth] for p in parts]
            scores = rrf(lists)
            # Items outside retrieval union are deliberately not returned.
        else:
            a = n/(n+5) if mode=='adaptive' else alpha
            scores = fuse(parts, [.9*(1-a), .9*a, .1])
        return ranked(scores), scores

def evaluate(model, rows, mode, alpha=.5, k=5):
    truth = defaultdict(set)
    for r in rows:
        if model.cat[r['item']]['available']=='1' and r['item'] not in model.hist.get(r['user'],set()):
            truth[r['user']].add(r['item'])
    buckets = defaultdict(list)
    exposed = set()
    for u, relevant in sorted(truth.items()):
        rec,_ = model.recommend(u,mode,alpha)
        top = rec[:k]; exposed.update(top)
        hits = len(set(top)&relevant)
        dcg = sum(1/math.log2(p+2) for p,i in enumerate(top) if i in relevant)
        idcg = sum(1/math.log2(p+2) for p in range(min(k,len(relevant))))
        cold = {i for i in relevant if not model.users.get(i)}
        row = (dcg/idcg, hits/len(relevant), hits/k, len(top)/k)
        buckets['all'].append(row)
        buckets['warm_user' if model.hist.get(u) else 'cold_user'].append(row)
        if cold: buckets['cold_item_recall'].append(len(set(top)&cold)/len(cold))
    result = {}
    for name,vals in buckets.items():
        if name=='cold_item_recall': result[name]={'users':len(vals),'recall':sum(vals)/len(vals)}
        else: result[name]={'users':len(vals), **{key:sum(v[j] for v in vals)/len(vals)
                    for j,key in enumerate(['ndcg','recall','precision','fill'])}}
    eligible = sum(r['available']=='1' for r in model.cat.values())
    result['catalogue_coverage'] = len(exposed)/eligible
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--test',action='store_true',help='Évaluer une fois les choix figés')
    ap.add_argument('--user',default='U01')
    ap.add_argument('--interests',default='',help='Tags déclarés pour un nouvel utilisateur, séparés par des virgules')
    args=ap.parse_args()
    cat,rows=load()
    sets={name:[r for r in rows if r['split']==name] for name in ['train','validation','test']}
    assert max(r['timestamp'] for r in sets['train']) < min(r['timestamp'] for r in sets['validation'])
    assert max(r['timestamp'] for r in sets['validation']) < min(r['timestamp'] for r in sets['test'])
    model=Hybrid(cat,sets['train'])
    grid=[0,.25,.5,.75,1]
    results={a:evaluate(model,sets['validation'],'hybrid',a) for a in grid}
    best=max(grid,key=lambda a:results[a]['all']['ndcg']) # stable tie: smallest alpha
    print('Validation : alpha = poids relatif CF parmi contenu + CF')
    for a,r in results.items(): print(a,r['all'])
    print('Alpha choisi sur validation uniquement :',best)
    target=sets['test'] if args.test else sets['validation']
    print('Évaluation :', 'TEST' if args.test else 'VALIDATION')
    for mode in ['pop','content','cf','hybrid','adaptive','rrf']:
        print(mode,evaluate(model,target,mode,best))
    ranking,scores=model.recommend(args.user,'hybrid',best,args.interests.split(',') if args.interests else ())
    print('Top-5 pour',args.user)
    for i in ranking[:5]: print(i,cat[i]['title'],round(scores[i],6))
    print('Données pédagogiques synthétiques : aucun gain généralisable ne peut en être déduit.')

if __name__=='__main__': main()
