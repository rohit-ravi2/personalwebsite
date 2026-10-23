from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[1]
out=r/'public/images/projects';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#0F3B2E','axes.labelcolor':'#0F3B2E','xtick.color':'#0F3B2E','ytick.color':'#0F3B2E','font.size':13})
f,ax=plt.subplots(figsize=(10,5.625),facecolor='#F3EDE2');ax.set_facecolor('#F3EDE2');x=np.arange(4)
a=ax.barh(x+.17,[30.07,26.99,26.76,20.33],.32,color='#567B6B',label='Greedy decoder');b=ax.barh(x-.17,[25.43,21.87,21.89,15.02],.32,color='#0F3B2E',label='+ language model')
ax.set_yticks(x,['Generic model','Input correction','Early adapter','Personal fine-tune']);ax.invert_yaxis();ax.set_xlim(0,36);ax.set_xlabel('Character error rate (%) · lower is better');ax.set_title('Wrist-EMG handwriting decoding',loc='left',fontweight='bold',pad=25);ax.bar_label(a,fmt='%.2f',padding=5,fontsize=10);ax.bar_label(b,fmt='%.2f',padding=5,fontsize=10);ax.legend(loc='lower right',frameon=False,fontsize=10);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);f.text(.98,.02,'Ten held-out users · supervised personal calibration',ha='right',fontsize=10,color='#567B6B');f.tight_layout(rect=[0,.05,1,1]);f.savefig(out/'handwriting-emg-results.png',dpi=140);plt.close(f)
# A conceptual workflow, deliberately not a biological/mechanistic illustration.
def diagram(filename,title,labels,subtitle):
 boxes=[]
 for i,(a,b) in enumerate(labels):
  x=36+i*310
  boxes.append(f'<rect x="{x}" y="180" width="280" height="150" rx="15" fill="#fff" stroke="#8FA99B"/><text x="{x+140}" y="237" text-anchor="middle" font-size="23" font-weight="700">{a}</text><text x="{x+140}" y="279" text-anchor="middle" font-size="17">{b}</text>')
  if i<2:boxes.append(f'<path d="M{x+284} 255h20m-7-6 7 6-7 6" fill="none" stroke="#0F3B2E" stroke-width="2"/>')
 s=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 562.5"><rect width="1000" height="562.5" fill="#F3EDE2"/><g font-family="system-ui,sans-serif" fill="#0F3B2E"><text x="36" y="80" font-size="32" font-weight="700">{title}</text>'+''.join(boxes)+f'<text x="36" y="440" font-size="19">{subtitle}</text><text x="36" y="478" font-size="15" fill="#567B6B">Research workflow · no causal mechanism established by this diagram</text></g></svg>'
 (out/filename).write_text(s)
diagram('asd-myelin-workflow.svg','Investigating myelination and developmental timing',[('Genetic evidence','GWAS &amp; gene sets'),('Cellular context','Single-cell evidence'),('Functional models','Conduction &amp; timing')],'Testing a hypothesis against evidence at different scales')
diagram('code-ocr-workflow.svg','Source-code OCR',[('Image','Source-code capture'),('Recognition','Characters &amp; layout'),('Editable code','Syntax &amp; indentation')],'An early computer-vision engineering project')

# Published graph data already shipped with the interactive explorer.
import json
from matplotlib.collections import LineCollection
graph = json.loads((r/'public/data/connectome.json').read_text())
strong = [edge for edge in graph['edges'] if edge['weight'] >= 10]
angles = np.linspace(0, 2*np.pi, len(graph['nodes']), endpoint=False)
pos = {node['id']:(np.cos(angle),np.sin(angle)) for node,angle in zip(graph['nodes'],angles)}
colors = {'sensory':'#B86A38', 'inter':'#0F3B2E', 'motor':'#668EBA', 'other':'#888888'}
f, ax = plt.subplots(figsize=(10,5.625), facecolor='#F3EDE2')
ax.set_facecolor('#F3EDE2')
ax.add_collection(LineCollection([[pos[e['source']],pos[e['target']]] for e in strong],colors='#0F3B2E',alpha=.12,linewidths=.6))
ax.scatter([pos[n['id']][0] for n in graph['nodes']],[pos[n['id']][1] for n in graph['nodes']],s=18,c=[colors[n['role']] for n in graph['nodes']])
ax.set_aspect('equal'); ax.axis('off')
f.text(.05,.91,'C. elegans cell-class connectivity',fontsize=20,fontweight='bold')
f.text(.05,.84,f"{len(graph['nodes'])} classes · {len(strong)} connections shown (weight ≥ 10)",fontsize=12)
f.text(.05,.05,'Cook et al. 2019 · circular layout; positions are not anatomical',fontsize=10)
f.savefig(out/'celegans-connectivity.png',dpi=140);plt.close(f)
