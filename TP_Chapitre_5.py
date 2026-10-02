"""Chapitre 5 — factorisation matricielle explicite SGD/ALS et ALS implicite.
Python >= 3.9 ; dépendance : numpy. Exemples et données synthétiques pédagogiques.
Objectif explicite : 1/2 somme_observations [e² + reg*(||p||²+||q||²)
                                           + reg_bias*(bu²+bi²)].
Les pénalités sont donc pondérées par le nombre d'observations de chaque entité.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def load_csv(path, rating_min=1.):
    events, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            u,i = row['userId'].strip(), row['movieId'].strip()
            r,t = float(row['rating']), int(row['timestamp'])
            if not u or not i or not math.isfinite(r) or not rating_min<=r<=5:
                raise ValueError('Identifiants ou note invalide.')
            if (u,i) in seen: raise ValueError('Couple répété : définir une politique temporelle.')
            seen.add((u,i)); events.append((u,i,r,t))
    if not events: raise ValueError('Données vides.')
    return sorted(events,key=lambda e:(e[3],e[0],e[1]))


def synthetic(seed=42):
    """80 utilisateurs, 60 items, 20 notes chacun ; aucune prétention de réalisme."""
    rng=np.random.default_rng(seed)
    p,q=rng.normal(0,.65,(80,3)),rng.normal(0,.65,(60,3))
    bu,bi=rng.normal(0,.25,80),rng.normal(0,.25,60)
    raw=[]
    for u in range(80):
        for i in rng.choice(60,20,replace=False):
            r=float(np.clip(np.round((3.4+bu[u]+bi[i]+p[u]@q[i]+rng.normal(0,.15))*2)/2,1,5))
            raw.append((f'U{u:03d}',f'I{i:03d}',r))
    rng.shuffle(raw)
    return [(u,i,r,1700000000+j*60) for j,(u,i,r) in enumerate(raw)]


def split_time(events):
    n=len(events)
    if n<5: raise ValueError('Au moins cinq événements requis.')
    a,b=events[max(0,int(.6*n)-1)][3],events[max(0,int(.8*n)-1)][3]
    parts=([e for e in events if e[3]<=a],[e for e in events if a<e[3]<=b],[e for e in events if e[3]>b])
    if not all(parts): raise ValueError('Un bloc temporel est vide ; vérifier les dates.')
    return parts


class MatrixFactorization:
    def __init__(self, events, factors=3, reg=.05, reg_bias=.05, seed=42, rating_min=1.):
        if factors<0 or reg<0 or reg_bias<0: raise ValueError('Paramètres négatifs interdits.')
        if not all(map(math.isfinite,[reg,reg_bias,rating_min])):raise ValueError('Paramètres non finis.')
        if not events:raise ValueError('Entraînement vide.')
        self.users=sorted({u for u,_,_,_ in events});self.items=sorted({i for _,i,_,_ in events})
        self.uid={u:j for j,u in enumerate(self.users)};self.iid={i:j for j,i in enumerate(self.items)}
        self.obs=[(self.uid[u],self.iid[i],float(r)) for u,i,r,_ in events]
        self.hist={u:set() for u in self.users}
        for u,i,_,_ in events:self.hist[u].add(i)
        self.mu=float(np.mean([r for _,_,r in self.obs]));self.reg=reg;self.reg_bias=reg_bias
        self.rating_min=rating_min;self.rng=np.random.default_rng(seed)
        self.P=self.rng.normal(0,.1,(len(self.users),factors));self.Q=self.rng.normal(0,.1,(len(self.items),factors))
        self.bu=np.zeros(len(self.users));self.bi=np.zeros(len(self.items))
        self.by_user=[[] for _ in self.users];self.by_item=[[] for _ in self.items]
        for u,i,r in self.obs:self.by_user[u].append((i,r));self.by_item[i].append((u,r))
        self.history=[];self.best_epoch=None

    def raw(self,u,i):
        a,b=self.uid.get(u),self.iid.get(i)
        value=self.mu+(self.bu[a] if a is not None else 0)+(self.bi[b] if b is not None else 0)
        if a is not None and b is not None:value+=self.P[a]@self.Q[b]
        return float(value)

    def predict(self,u,i):return float(np.clip(self.raw(u,i),self.rating_min,5))

    def loss(self):
        value=0.
        for u,i,r in self.obs:
            e=r-self.mu-self.bu[u]-self.bi[i]-self.P[u]@self.Q[i]
            value+=.5*(e*e+self.reg*(self.P[u]@self.P[u]+self.Q[i]@self.Q[i])+
                        self.reg_bias*(self.bu[u]**2+self.bi[i]**2))
        return float(value)

    def sgd_epoch(self,lr):
        if not math.isfinite(lr) or lr<=0:raise ValueError('Pas d’apprentissage positif requis.')
        for j in self.rng.permutation(len(self.obs)):
            u,i,r=self.obs[int(j)]
            p,q=self.P[u].copy(),self.Q[i].copy()
            e=r-self.mu-self.bu[u]-self.bi[i]-p@q
            self.bu[u]+=lr*(e-self.reg_bias*self.bu[u]);self.bi[i]+=lr*(e-self.reg_bias*self.bi[i])
            self.P[u]+=lr*(e*q-self.reg*p);self.Q[i]+=lr*(e*p-self.reg*q)
        if not all(np.isfinite(a).all() for a in (self.P,self.Q,self.bu,self.bi)):
            raise FloatingPointError('Divergence ; diminuer le pas et inspecter les données.')

    def als_epoch(self):
        # Résoudre les moindres carrés augmentés évite l'inversion explicite.
        def block(neighbors,other_factors,other_bias):
            ids=np.array([i for i,_ in neighbors]);ratings=np.array([r for _,r in neighbors])
            X=np.column_stack([np.ones(len(ids)),other_factors[ids]])
            target=ratings-self.mu-other_bias[ids]
            penalty=len(ids)*np.array([self.reg_bias]+[self.reg]*self.P.shape[1])
            A=np.vstack([X,np.diag(np.sqrt(penalty))]);b=np.r_[target,np.zeros(X.shape[1])]
            return np.linalg.lstsq(A,b,rcond=None)[0]
        for u,neighbors in enumerate(self.by_user):
            z=block(neighbors,self.Q,self.bi);self.bu[u],self.P[u]=z[0],z[1:]
        for i,neighbors in enumerate(self.by_item):
            z=block(neighbors,self.P,self.bu);self.bi[i],self.Q[i]=z[0],z[1:]

    def rmse_raw(self,events):
        return math.sqrt(sum((self.raw(u,i)-r)**2 for u,i,r,_ in events)/len(events))

    def fit(self,events,method='sgd',epochs=40,lr=.02,validation=None,patience=8):
        if method not in ('sgd','als') or epochs<1 or patience<1:raise ValueError('Configuration d’entraînement invalide.')
        def state():return tuple(a.copy() for a in (self.P,self.Q,self.bu,self.bi))
        best=self.rmse_raw(validation) if validation else math.inf;checkpoint=state();stall=0
        self.best_epoch=0 if validation else None;self.history=[]
        self.history.append({'epoch':0,'objective':self.loss(),'train_rmse_raw':self.rmse_raw(events),
                             'validation_rmse_raw':self.rmse_raw(validation) if validation else None})
        for epoch in range(1,epochs+1):
            self.sgd_epoch(lr) if method=='sgd' else self.als_epoch()
            valid=self.rmse_raw(validation) if validation else None
            self.history.append({'epoch':epoch,'objective':self.loss(),'train_rmse_raw':self.rmse_raw(events),
                                 'validation_rmse_raw':valid})
            if validation:
                if valid<best-1e-8:best=valid;checkpoint=state();self.best_epoch=epoch;stall=0
                else:stall+=1
                if stall>=patience:break
        if validation:self.P,self.Q,self.bu,self.bi=checkpoint
        else:self.best_epoch=len(self.history)-1
        return self

    def recommend(self,u,top=5):
        unseen=[i for i in self.items if i not in self.hist.get(u,set())]
        # Ranking brut : éviter les ex aequo artificiels dus au bornage.
        unseen.sort(key=lambda i:(-self.raw(u,i),i))
        return [{'item':i,'score_raw':self.raw(u,i),'rating_clipped':self.predict(u,i)} for i in unseen[:top]]


def evaluate(model,events,top=5):
    errors=[model.predict(u,i)-r for u,i,r,_ in events];users=sorted({u for u,_,_,_ in events})
    eligible=[];rels={}
    for u in users:
        rel={i for v,i,r,_ in events if v==u and r>=4 and i in model.iid and i not in model.hist.get(u,set())}
        if u in model.uid and rel and len(set(model.items)-model.hist[u])>=top:eligible.append(u);rels[u]=rel
    ps,rs,ns=[],[],[];covered=set()
    for u in eligible:
        rec=[a['item'] for a in model.recommend(u,top)];hits=[int(i in rels[u]) for i in rec]
        ps.append(sum(hits)/top);rs.append(sum(hits)/len(rels[u]));covered.update(rec)
        ideal=sum(1/math.log2(j+2) for j in range(min(top,len(rels[u]))))
        ns.append(sum(h/math.log2(j+2) for j,h in enumerate(hits))/ideal)
    avg=lambda x:sum(x)/len(x) if x else None
    return {'mae_clipped':sum(map(abs,errors))/len(errors),'rmse_clipped':math.sqrt(sum(e*e for e in errors)/len(errors)),
       'rmse_raw':model.rmse_raw(events),'events':len(events),
       'cold_user_events':sum(u not in model.uid for u,_,_,_ in events),
       'cold_item_events':sum(i not in model.iid for _,i,_,_ in events),
       'ranking_users':len(eligible),'ranking_users_excluded':len(users)-len(eligible),
       'precision':avg(ps),'recall':avg(rs),'ndcg':avg(ns),
       'coverage':len(covered)/len(model.items) if eligible else None}


def implicit_als(counts,factors=2,alpha=10.,reg=.1,epochs=20,seed=42):
    """Prototype dense uniquement ; objectif sur TOUS les couples, biais absents.
    z=1[count>0], c=1+alpha*count. Ici reg est globale, NON pondérée par degré.
    """
    counts=np.asarray(counts,dtype=float)
    if counts.ndim!=2 or not counts.size or not np.isfinite(counts).all() or (counts<0).any():
        raise ValueError('Matrice de comptes non négatifs requise.')
    if factors<1 or alpha<0 or reg<=0 or epochs<1 or not all(map(math.isfinite,[alpha,reg])):
        raise ValueError('Configuration ALS implicite invalide.')
    rng=np.random.default_rng(seed);m,n=counts.shape
    P=rng.normal(0,.1,(m,factors));Q=rng.normal(0,.1,(n,factors))
    z=(counts>0).astype(float);C=1+alpha*counts;eye=np.eye(factors)
    def loss():return float(np.sum(C*(z-P@Q.T)**2)+reg*(np.sum(P*P)+np.sum(Q*Q)))
    history=[loss()]
    for _ in range(epochs):
        for u in range(m):P[u]=np.linalg.solve(Q.T@(C[u,:,None]*Q)+reg*eye,Q.T@(C[u]*z[u]))
        for i in range(n):Q[i]=np.linalg.solve(P.T@(C[:,i,None]*P)+reg*eye,P.T@(C[:,i]*z[:,i]))
        history.append(loss())
    return P,Q,history


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ratings',type=Path,default=Path(__file__).with_name('notes_chapitre5.csv'))
    ap.add_argument('--synthetic',action='store_true',help='Générer 1600 notes artificielles en mémoire.')
    ap.add_argument('--method',choices=['sgd','als'],default='sgd')
    ap.add_argument('--factors',type=int,default=3);ap.add_argument('--reg',type=float,default=.05)
    ap.add_argument('--reg-bias',type=float,default=.05);ap.add_argument('--lr',type=float,default=.02)
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--patience',type=int,default=8)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--top',type=int,default=3)
    ap.add_argument('--user',default='Yasmine');ap.add_argument('--rating-min',type=float,choices=[.5,1.],default=1.)
    ap.add_argument('--evaluate',action='store_true');ap.add_argument('--validation-only',action='store_true')
    ap.add_argument('--implicit-demo',action='store_true');ap.add_argument('--alpha',type=float,default=10.)
    args=ap.parse_args()
    if args.top<1:ap.error('--top doit être positif.')
    if args.implicit_demo:
        counts=np.array([[3,1,0,0,0],[2,0,1,0,0],[0,0,0,4,1],[0,1,0,2,3]],dtype=float)
        P,Q,h=implicit_als(counts,args.factors,args.alpha,args.reg,args.epochs,args.seed)
        out={'note':'Comptes fictifs ; scores latents, pas des probabilités. Objectif implicite global.',
             'counts':counts.tolist(),'scores':(P@Q.T).tolist(),'objective_history':h}
    else:
        events=synthetic(args.seed) if args.synthetic else load_csv(args.ratings,args.rating_min)
        train,val,test=split_time(events) if args.evaluate else (events,None,None)
        model=MatrixFactorization(train,args.factors,args.reg,args.reg_bias,args.seed,args.rating_min)
        model.fit(train,args.method,args.epochs,args.lr,val,args.patience)
        out={'method':args.method,'factors':args.factors,'best_epoch':model.best_epoch,
             'training_history':model.history,'recommendations':model.recommend(args.user,args.top)}
        if val:
            # facteurs=0 donne un modèle de biais seul pour une baseline comparable.
            base=MatrixFactorization(train,0,args.reg,args.reg_bias,args.seed,args.rating_min)
            base.fit(train,'als',args.epochs,args.lr,val,args.patience)
            out.update({'protocol':'Découpage global 60/20/20 ; choix de l’époque sur validation, restauration du meilleur état ; aucune note test à l’entraînement.',
                        'sizes':{'train':len(train),'validation':len(val),'test':len(test)},
                        'validation_mf':evaluate(model,val,args.top),'validation_bias':evaluate(base,val,args.top)})
            if not args.validation_only:out.update(test_mf=evaluate(model,test,args.top),test_bias=evaluate(base,test,args.top))
        else:out['note']='Ajustement aux données complètes : aucune mesure de généralisation.'
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
