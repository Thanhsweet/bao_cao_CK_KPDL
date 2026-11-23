"""Bai tap PCA 3.2.1 tu tep PCA-Excercise.pdf, trang 6.
Python 3, numpy, scikit-learn, matplotlib. Cung seed 42 de tai lap.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'figures' / 'pca'
OUT.mkdir(parents=True,exist_ok=True)
S = np.array([[.3,1.,1.],[1.,9.,1.],[1.,1.,9.]])
CFG = {'X1':(np.array([-6.,6.,6.]),np.array([6.,6.,6.])),
       'X2':(np.array([-2.,0.,0.]),np.array([2.,0.,0.]))}
rng = np.random.default_rng(42)
summary={}
for name, (mu0,mu1) in CFG.items():
    A=rng.multivariate_normal(mu0, S, size=200)
    B=rng.multivariate_normal(mu1, S, size=200)
    X=np.vstack([A,B]); y=np.r_[np.zeros(200,dtype=int),np.ones(200,dtype=int)]
    pca=PCA(n_components=3,svd_solver='full')
    Y=pca.fit_transform(X)
    ratio=(pca.explained_variance_ratio_*100)
    summary[name]={'total':len(X),'mean0':mu0.tolist(),'mean1':mu1.tolist(),
                   'cov':S.tolist(),'eigenvalues':pca.explained_variance_.round(5).tolist(),
                   'explained_ratio':ratio.round(5).tolist(),
                   'variance_2pcs':round(float(ratio[:2].sum()),5)}
    for tag,M,axes in [('X',X,[(0,1),(0,2),(1,2)]),('Y',Y,[(0,1),(0,2),(1,2)])]:
        for i,j in axes:
            fig,ax=plt.subplots(figsize=(5.5,4.2))
            ax.scatter(M[y==0,i],M[y==0,j],s=9,alpha=.5,label='Class 1')
            ax.scatter(M[y==1,i],M[y==1,j],s=9,alpha=.5,label='Class 2')
            ax.set_xlabel(f'{tag}{i+1}');ax.set_ylabel(f'{tag}{j+1}')
            ax.set_title(f'{name}: {tag}{i+1}-{tag}{j+1}')
            ax.grid(alpha=.2); ax.legend(fontsize=8);fig.tight_layout()
            fig.savefig(OUT/f'{name}_{tag}{i+1}{tag}{j+1}.png',dpi=150)
            plt.close(fig)
    # user request PCA: demonstrate projection onto first 2 PC
    np.savetxt(OUT/f'{name}_PCA_2D.csv',Y[:,:2],delimiter=',',header='PC1,PC2',comments='')
    print(name, 'eigenvalues:',summary[name]['eigenvalues'],'explained ratios:',summary[name]['explained_ratio'])
(OUT/'pca_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
