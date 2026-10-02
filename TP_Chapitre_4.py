"""Filtrage collaboratif pédagogique, Python >= 3.9, bibliothèque standard.
Données: CSV userId,movieId,rating,timestamp ; échelle [1,5] ou [0.5,5].
Ne convient pas à un grand catalogue : calcul exact à la demande, sans index.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EPS = 1e-12


def load(path, rating_min=1.):
    events, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            u, i = row['userId'].strip(), row['movieId'].strip()
            r, t = float(row['rating']), int(row['timestamp'])
            if not u or not i or not math.isfinite(r) or not rating_min <= r <= 5:
                raise ValueError(f'Identifiants requis et notes finies entre {rating_min} et 5.')
            if (u, i) in seen:
                raise ValueError('Couple répété : définir une politique temporelle avant utilisation.')
            seen.add((u, i)); events.append((u, i, r, t))
    if not events:
        raise ValueError('Fichier vide.')
    return sorted(events, key=lambda x: (x[3], x[0], x[1]))


def similarity(a, b, metric='pearson', min_support=2, shrinkage=2.0):
    common = sorted(a.keys() & b.keys())
    n = len(common)
    if n < min_support:
        return None, n, 'support insuffisant'
    x, y = [a[i] for i in common], [b[i] for i in common]
    if metric == 'pearson':
        mx, my = sum(x)/n, sum(y)/n
        x, y = [v-mx for v in x], [v-my for v in y]
    den = math.sqrt(sum(v*v for v in x)*sum(v*v for v in y))
    if den <= EPS:
        return None, n, 'norme ou variance nulle'
    raw = max(-1., min(1., sum(v*w for v,w in zip(x,y))/den))
    return raw*n/(n+shrinkage), n, 'ok'


class KNN:
    def __init__(self, events, method='user', metric='pearson', k=3, min_support=2, shrinkage=2., rating_min=1.):
        if method not in ('user','item') or metric not in ('pearson','cosine'):
            raise ValueError('Méthode ou similarité inconnue.')
        if k < 1 or min_support < 1 or not math.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError('k et support positifs ; shrinkage fini et positif ou nul.')
        self.method, self.metric = method, metric
        self.rating_min = rating_min
        self.k, self.min_support, self.shrinkage = k, min_support, shrinkage
        self.rows, self.cols = defaultdict(dict), defaultdict(dict)
        for u,i,r,_ in events:
            self.rows[u][i] = r; self.cols[i][u] = r
        if not self.rows:
            raise ValueError('Entraînement vide.')
        self.means = {u:sum(v.values())/len(v) for u,v in self.rows.items()}
        self.item_means = {i:sum(v.values())/len(v) for i,v in self.cols.items()}
        self.global_mean = sum(sum(v.values()) for v in self.rows.values())/sum(map(len,self.rows.values()))
        # Pour Item-KNN, résidus centrés par utilisateur, y compris pour la similarité.
        self.residuals = {i:{u:r-self.means[u] for u,r in col.items()} for i,col in self.cols.items()}

    def fallback(self, u, i):
        if u in self.means: return self.means[u], 'moyenne utilisateur'
        if i in self.item_means: return self.item_means[i], 'moyenne item'
        return self.global_mean, 'moyenne globale'

    def predict(self, u, i):
        base, reason = self.fallback(u,i)
        if u not in self.rows or i not in self.cols:
            return {'score':base,'raw':base,'fallback':reason,'neighbors':[],'audit':[]}
        candidates = sorted(self.cols[i]) if self.method == 'user' else sorted(self.rows[u])
        audit, neighbors = [], []
        for v in candidates:
            if (self.method == 'user' and v == u) or (self.method == 'item' and v == i):
                continue
            if self.method == 'user':
                a,b,metric = self.rows[u], self.rows[v], self.metric
                residual = self.rows[v][i]-self.means[v]
            else:
                a,b,metric = self.residuals[i], self.residuals[v], 'cosine'
                residual = self.rows[u][v]-self.means[u]
            weight,n,status = similarity(a,b,metric,self.min_support,self.shrinkage)
            row = {'id':v,'support':n,'weight':weight,'residual':residual,
                   'status':status if weight is None else ('admissible' if weight>EPS else 'similarité non positive')}
            audit.append(row)
            if weight is not None and weight>EPS: neighbors.append(row)
        neighbors.sort(key=lambda x:(-x['weight'],x['id']))
        selected=neighbors[:self.k]
        if not selected:
            return {'score':base,'raw':base,'fallback':reason,'neighbors':[],'audit':audit}
        den=sum(x['weight'] for x in selected)
        raw=self.means[u]+sum(x['weight']*x['residual'] for x in selected)/den
        for x in selected: x['contribution']=x['weight']*x['residual']/den
        return {'score':max(self.rating_min,min(5.,raw)),'raw':raw,'fallback':None,'neighbors':selected,'audit':audit}

    def recommend(self, u, top=3, popular=False):
        candidates = sorted(self.cols.keys()-self.rows.get(u,{}).keys())
        if popular:
            candidates.sort(key=lambda i:(-len(self.cols[i]),i))
            return [{'item':i,'count':len(self.cols[i])} for i in candidates[:top]]
        out=[{'item':i,**self.predict(u,i)} for i in candidates]
        return sorted(out,key=lambda x:(-x['score'],x['item']))[:top]


def temporal_split(events):
    """60/20/20 approximatif : égalités de timestamp gardées dans un seul bloc."""
    if len(events)<5: raise ValueError('Au moins 5 événements requis pour le découpage.')
    t1=events[max(0,int(.6*len(events))-1)][3]
    t2=events[max(0,int(.8*len(events))-1)][3]
    train=[e for e in events if e[3]<=t1]
    val=[e for e in events if t1<e[3]<=t2]
    test=[e for e in events if e[3]>t2]
    if not all((train,val,test)): raise ValueError('Blocs temporels vides : vérifier les dates.')
    return train,val,test


def evaluate(model, heldout, top):
    errors=[model.predict(u,i)['score']-r for u,i,r,_ in heldout]
    cold_user=sum(u not in model.rows for u,_,_,_ in heldout)
    cold_item=sum(i not in model.cols for _,i,_,_ in heldout)
    fallback=sum(model.predict(u,i)['fallback'] is not None for u,i,_,_ in heldout)
    rel=defaultdict(set); users={u for u,_,_,_ in heldout}
    for u,i,r,_ in heldout:
        if r>=4 and i in model.cols and i not in model.rows.get(u,{}):rel[u].add(i)
    results={}
    # Même population admissible pour KNN et popularité ; catalogue = items d'entraînement.
    eligible=[u for u in sorted(users) if u in model.rows and rel[u] and
              len(model.cols.keys()-model.rows[u].keys())>=top]
    for popular in (False,True):
        precisions,recalls,ndcgs,covered=[],[],[],set()
        for u in eligible:
            rec=[x['item'] for x in model.recommend(u,top,popular)]
            hits=[int(i in rel[u]) for i in rec]
            precisions.append(sum(hits)/top);recalls.append(sum(hits)/len(rel[u]))
            dcg=sum(h/math.log2(j+2) for j,h in enumerate(hits))
            ideal=sum(1/math.log2(j+2) for j in range(min(top,len(rel[u]))))
            ndcgs.append(dcg/ideal);covered.update(rec)
        avg=lambda xs:sum(xs)/len(xs) if xs else None
        results['popularity' if popular else 'knn']={'precision':avg(precisions),'recall':avg(recalls),
             'ndcg':avg(ndcgs),'coverage':len(covered)/len(model.cols) if eligible else None}
    return {'events':len(heldout),'mae':sum(map(abs,errors))/len(errors),
       'rmse':math.sqrt(sum(e*e for e in errors)/len(errors)),
       'cold_user_events':cold_user,'cold_item_events':cold_item,'fallback_events':fallback,
       'ranking_users_total':len(users),'ranking_users_eligible':len(eligible),
       'ranking_users_excluded':len(users)-len(eligible),'ranking':results}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ratings',type=Path,default=Path(__file__).with_name('notes_chapitre4.csv'))
    ap.add_argument('--user',default='Yasmine');ap.add_argument('--item',default='Parasite')
    ap.add_argument('--method',choices=['user','item'],default='user')
    ap.add_argument('--metric',choices=['pearson','cosine'],default='pearson',help='User-KNN seulement ; Item-KNN utilise le cosinus ajusté.')
    ap.add_argument('--k',type=int,default=3);ap.add_argument('--min-support',type=int,default=2)
    ap.add_argument('--shrinkage',type=float,default=2.);ap.add_argument('--top',type=int,default=3)
    ap.add_argument('--evaluate',action='store_true')
    ap.add_argument('--validation-only',action='store_true',help='Ne pas consulter le test pendant les réglages.')
    ap.add_argument('--rating-min',type=float,choices=[0.5,1.0],default=1.)
    args=ap.parse_args()
    if args.top<1:ap.error('--top doit être positif.')
    events=load(args.ratings,args.rating_min)
    def make(rows):return KNN(rows,args.method,args.metric,args.k,args.min_support,args.shrinkage,args.rating_min)
    if args.evaluate:
        train,val,test=temporal_split(events);model=make(train)
        out={'protocol':'Modèle figé entraîné sur le premier bloc ; validation puis test, sans réentraînement.',
             'sizes':{'train':len(train),'validation':len(val),'test':len(test)},
             'validation':evaluate(model,val,args.top)}
        if not args.validation_only: out['test']=evaluate(model,test,args.top)
    else:
        model=make(events)
        out={'warning':'Démonstration sur les données fournies, sans mesure de généralisation.',
             'method':args.method,'metric':args.metric if args.method=='user' else 'adjusted_cosine',
             'target_already_observed':args.item in model.rows.get(args.user,{}),
             'prediction':model.predict(args.user,args.item),'recommendations':model.recommend(args.user,args.top)}
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
