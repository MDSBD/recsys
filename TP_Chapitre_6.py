"""BPR-MF pédagogique, Python >= 3.9 et NumPy.
CSV: userId,itemId,count,timestamp ; un couple unique par fichier ; count>0.
Les absences servent de contrastes échantillonnés, pas de rejets établis.
"""
import argparse,csv,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np


def load(path):
    rows=[];seen=set()
    with open(path,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            u,i=r['userId'].strip(),r['itemId'].strip();c=float(r['count']);t=int(r['timestamp'])
            if not u or not i or not math.isfinite(c) or c<=0:raise ValueError('Identifiants et compte positif requis.')
            if (u,i) in seen:raise ValueError('Couple répété : fixer une politique temporelle avant agrégation.')
            seen.add((u,i));rows.append((u,i,c,t))
    if not rows:raise ValueError('Fichier vide.')
    return sorted(rows,key=lambda e:(e[3],e[0],e[1]))


def split(rows):
    if len(rows)<5:raise ValueError('Au moins cinq observations requises.')
    a,b=rows[max(0,int(.6*len(rows))-1)][3],rows[max(0,int(.8*len(rows))-1)][3]
    parts=([r for r in rows if r[3]<=a],[r for r in rows if a<r[3]<=b],[r for r in rows if r[3]>b])
    if not all(parts):raise ValueError('Bloc temporel vide.')
    return parts


class BPR:
    def __init__(self,train,factors=8,reg=.01,seed=42):
        if not train or factors<1 or not math.isfinite(reg) or reg<0:raise ValueError('Configuration invalide.')
        self.users=sorted({r[0] for r in train});self.items=sorted({r[1] for r in train})
        self.uid={u:k for k,u in enumerate(self.users)};self.iid={i:k for k,i in enumerate(self.items)}
        self.hist=defaultdict(set)
        for u,i,_,_ in train:self.hist[u].add(i)
        self.pop=Counter(i for _,i,_,_ in train)  # couples uniques = utilisateurs distincts
        self.rng=np.random.default_rng(seed);self.reg=reg
        self.P=self.rng.normal(0,.1,(len(self.users),factors));self.Q=self.rng.normal(0,.1,(len(self.items),factors));self.b=np.zeros(len(self.items))
        self.pools={u:np.array([self.iid[i] for i in self.items if i not in self.hist[u]],dtype=int) for u in self.users}
        self.pairs=[(u,self.iid[i]) for u,i,_,_ in train if len(self.pools[u])]
        if not self.pairs:raise ValueError('Aucun utilisateur ne possède un candidat non observé.')
        self.history=[];self.best_epoch=0

    def score(self,u,i):
        if i not in self.iid:raise ValueError('Item hors catalogue d’entraînement.')
        j=self.iid[i]
        # Repli popularité pour un utilisateur inconnu ; pas de vecteur aléatoire.
        if u not in self.uid:return float(self.pop[i])
        return float(self.b[j]+self.P[self.uid[u]]@self.Q[j])

    def negative(self,u,sampler,probes=5):
        pool=self.pools[u]
        if sampler=='uniform':return int(self.rng.choice(pool))
        if sampler=='popularity':
            weights=np.array([self.pop[self.items[j]]**.75 for j in pool]);weights/=weights.sum()
            return int(self.rng.choice(pool,p=weights))
        if sampler=='hard':
            candidates=self.rng.choice(pool,min(probes,len(pool)),replace=False)
            return int(max(candidates,key=lambda j:(self.score(u,self.items[j]),-int(j))))
        raise ValueError('Échantillonneur inconnu.')

    def update(self,u,i,j,lr):
        a=self.uid[u];p=self.P[a].copy();qi=self.Q[i].copy();qj=self.Q[j].copy()
        delta=float(self.b[i]-self.b[j]+p@(qi-qj))
        g=float(np.exp(-np.logaddexp(0.,delta))) # sigmoid(-delta), stable
        self.P[a]+=lr*(g*(qi-qj)-self.reg*p)
        self.Q[i]+=lr*(g*p-self.reg*qi);self.Q[j]+=lr*(-g*p-self.reg*qj)
        self.b[i]+=lr*(g-self.reg*self.b[i]);self.b[j]+=lr*(-g-self.reg*self.b[j])
        return float(np.logaddexp(0.,-delta))

    def recommend(self,u,k=5,popular=False):
        candidates=[i for i in self.items if i not in self.hist.get(u,set())]
        candidates.sort(key=lambda i:(-(self.pop[i] if popular else self.score(u,i)),i))
        return candidates[:k]

    def fit(self,validation,epochs=30,lr=.05,k=5,sampler='uniform',patience=6):
        if epochs<1 or patience<1 or k<1 or not math.isfinite(lr) or lr<=0:raise ValueError('Paramètres invalides.')
        metric=evaluate(self,validation,k)['bpr']['ndcg']
        if metric is None:raise ValueError('Aucun utilisateur de validation admissible ; adapter le protocole avant le test.')
        best=metric;checkpoint=(self.P.copy(),self.Q.copy(),self.b.copy());stall=0
        self.history=[{'epoch':0,'sampled_logistic_loss':None,'validation_ndcg':metric}]
        for epoch in range(1,epochs+1):
            losses=[]
            # Bootstrap de couples positifs : distribution documentée, pas uniforme sur tous les triplets.
            for _ in range(len(self.pairs)):
                u,i=self.pairs[int(self.rng.integers(len(self.pairs)))];j=self.negative(u,sampler)
                losses.append(self.update(u,i,j,lr))
            if not all(np.isfinite(x).all() for x in (self.P,self.Q,self.b)):raise FloatingPointError('Divergence : diminuer le pas.')
            metric=evaluate(self,validation,k)['bpr']['ndcg']
            self.history.append({'epoch':epoch,'sampled_logistic_loss':float(np.mean(losses)),'validation_ndcg':metric})
            if metric>best+1e-12:
                best=metric;self.best_epoch=epoch;checkpoint=(self.P.copy(),self.Q.copy(),self.b.copy());stall=0
            else:stall+=1
            if stall>=patience:break
        self.P,self.Q,self.b=checkpoint
        return self


def evaluate(model,heldout,k):
    users=sorted({u for u,_,_,_ in heldout});rel=defaultdict(set)
    for u,i,_,_ in heldout:
        if i in model.iid and i not in model.hist.get(u,set()):rel[u].add(i)
    eligible=[u for u in users if u in model.uid and rel[u] and len(model.items)-len(model.hist[u])>=k]
    out={'users_total':len(users),'users_eligible':len(eligible),'users_excluded':len(users)-len(eligible),
         'cold_user_events':sum(u not in model.uid for u,_,_,_ in heldout),
         'cold_item_events':sum(i not in model.iid for _,i,_,_ in heldout)}
    for popular in [False,True]:
        ps,rs,ns=[],[],[];coverage=set()
        for u in eligible:
            rec=model.recommend(u,k,popular);hits=[int(i in rel[u]) for i in rec]
            ps.append(sum(hits)/k);rs.append(sum(hits)/len(rel[u]));coverage.update(rec)
            idcg=sum(1/math.log2(j+2) for j in range(min(k,len(rel[u]))))
            ns.append(sum(h/math.log2(j+2) for j,h in enumerate(hits))/idcg)
        avg=lambda x:sum(x)/len(x) if x else None
        out['popularity' if popular else 'bpr']={'precision':avg(ps),'recall':avg(rs),'ndcg':avg(ns),
            'coverage':len(coverage)/len(model.items) if eligible else None}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events',type=Path,default=Path(__file__).with_name('interactions_chapitre6.csv'))
    ap.add_argument('--factors',type=int,default=8);ap.add_argument('--reg',type=float,default=.01)
    ap.add_argument('--lr',type=float,default=.05);ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--patience',type=int,default=6);ap.add_argument('--k',type=int,default=5)
    ap.add_argument('--sampler',choices=['uniform','popularity','hard'],default='uniform')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--user',default='U00')
    ap.add_argument('--test',action='store_true',help='À activer seulement après choix des hyperparamètres.')
    args=ap.parse_args();train,val,test=split(load(args.events))
    model=BPR(train,args.factors,args.reg,args.seed).fit(val,args.epochs,args.lr,args.k,args.sampler,args.patience)
    out={'protocol':'Catalogue et historiques d’entraînement figés ; tous les candidats admissibles au ranking ; meilleure époque choisie par NDCG de validation.',
         'sizes':{'train':len(train),'validation':len(val),'test':len(test)},'seed':args.seed,'sampler':args.sampler,
         'best_epoch':model.best_epoch,'training_history':model.history,'validation':evaluate(model,val,args.k),
         'recommendations':model.recommend(args.user,args.k)}
    if args.test:out['test']=evaluate(model,test,args.k)
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
