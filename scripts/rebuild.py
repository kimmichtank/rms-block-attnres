#!/usr/bin/env python3
"""Recompute every headline number and static figure from public evidence only."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import textwrap

ROOT=Path(__file__).resolve().parents[1]
NAMES={'full':'Full','full_rms':'Full+RMS','block4':'Block4','block':'Block4','norm_rms':'RMS Block4','rms_block':'RMS Block4'}
COLORS={'Full':'#31688e','Full+RMS':'#7a5195','Block4':'#c27024','RMS Block4':'#178c74'}

def read(path):return json.loads((ROOT/path).read_text())
def rows(path):return list(csv.DictReader((ROOT/path).open()))
def lower(a,b):return 100*(1-a/b)
def save(path,obj):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def derive():
 runs=read('evidence/small/runs.json');small={r['run_id']:r for r in runs}
 result={'small':small,'57m_optimization':{},'diagnostics':{},
         '450m':read('evidence/450m/matched-summary.json'),
         '4b':read('evidence/4b/matched-comparison.json')}
 full={int(r['step']):r for r in rows('evidence/small/57m_full/validation.csv')}
 rms={int(r['step']):r for r in rows('evidence/small/57m_full_rms/validation.csv')}
 gap=[dict(step=s,labels=int(full[s]['labels']),nll_full=float(full[s]['nll']),nll_full_rms=float(rms[s]['nll']),
           advantage=float(full[s]['nll'])-float(rms[s]['nll'])) for s in sorted(full.keys()&rms.keys())]
 result['57m_optimization']=dict(validation_points=len(gap),full_rms_lower=sum(r['advantage']>0 for r in gap),
       gap=gap,last5_mean_advantage=statistics.mean(g['advantage'] for g in gap[-5:]),
       final_test_ppl_relative_reduction_pct=lower(small['57m_full_rms']['test_ppl'],small['57m_full']['test_ppl']))
 result['157m_comparison']={
       'rms_block_vs_full_ppl_reduction_pct':lower(small['157m_rms_block']['test_ppl'],small['157m_full']['test_ppl']),
       'rms_block_vs_block_ppl_reduction_pct':lower(small['157m_rms_block']['test_ppl'],small['157m_block']['test_ppl'])}
 result['450m_comparison']={
       'rms_block_vs_full_ppl_reduction_pct':lower(small['450m_rms_block']['test_ppl'],small['450m_full']['test_ppl']),
       'rms_block_vs_block_ppl_reduction_pct':lower(small['450m_rms_block']['test_ppl'],small['450m_block']['test_ppl'])}
 for size in ('57m','157m'):
  data=read(f'evidence/diagnostics/{size}/summary.json')[size];f=data['full'];out=f['output_rms']
  ratios=[out[i+1]['median']/out[i]['median'] for i in range(0,len(out),2)]
  cf={}
  for variant,v in data.items():
   cf[variant]={z['metric']:z for z in v['counterfactual'] if z['read_kind']=='next_sublayer' and z['member_in_block']=='all' and z['c']==4}
  result['diagnostics'][size]=dict(
      residual_median_rms_max_min_ratio=max(v['median'] for v in out)/min(v['median'] for v in out),
      ffn_over_attention_median=statistics.median(ratios),ffn_over_attention_min=min(ratios),ffn_over_attention_max=max(ratios),
      ffn_larger_layers=sum(v>1 for v in ratios),layers=len(ratios),valid_positions=out[0]['count'],
      slope_log_alpha_log_r=f['coefficient_relation']['slope_log_alpha_on_log_r'],
      raw_ratio=f['within_block_raw_r_ratio'],effective_ratio=f['within_block_directional_coefficient_ratio'],counterfactual_c4=cf)
 byvariant={r['variant']:r for r in result['4b']['rows']}
 full4={int(r['step']):float(r['nll']) for r in rows('evidence/4b/full/validation.csv')}
 rms4={int(r['step']):float(r['nll']) for r in rows('evidence/4b/rms_block/validation.csv')}
 shared=sorted(full4.keys()&rms4.keys())
 result['4b']['rms_vs_full_validation_points_lower_nll']=sum(rms4[s]<full4[s] for s in shared)
 result['4b']['rms_vs_full_validation_points_total']=len(shared)
 result['4b']['rms_vs_full_ppl_reduction_pct']=lower(byvariant['rms_block']['ppl'],byvariant['full']['ppl'])
 result['4b']['rms_vs_block_ppl_reduction_pct']=lower(byvariant['rms_block']['ppl'],byvariant['block']['ppl'])
 result['4b']['rms_minus_full_nll']=byvariant['rms_block']['nll']-byvariant['full']['nll']
 timing=read('evidence/4b/step-time-summary.json')
 block_curve=[(int(r['step']),float(r['nll'])) for r in rows('evidence/4b/block/validation.csv')]
 rms_curve=[(int(r['step']),float(r['nll'])) for r in rows('evidence/4b/rms_block/validation.csv')]
 def crossing(curve,target):
  for step,value in curve:
   if value==target:return float(step)
  for (s0,y0),(s1,y1) in zip(curve,curve[1:]):
   if y0>target and y1<=target:
    return s0+(s1-s0)*(y0-target)/(y0-y1)
  raise ValueError(f'no downward crossing for NLL {target}')
 targets=[]
 block_ms=timing['block']['median_step_ms'];rms_ms=timing['rms_block']['median_step_ms']
 for target in (3.20,3.10,3.05,3.00,2.98,byvariant['block']['nll']):
  block_step=crossing(block_curve,target);rms_step=crossing(rms_curve,target)
  data_speedup=block_step/rms_step
  time_speedup=(block_step*block_ms)/(rms_step*rms_ms)
  targets.append(dict(target_nll=target,block_step=block_step,rms_block_step=rms_step,
      data_efficiency_speedup=data_speedup,token_reduction_pct=100*(1-rms_step/block_step),
      time_to_quality_speedup=time_speedup,training_time_reduction_pct=100*(1-1/time_speedup)))
 result['4b']['efficiency']={
     'timing':timing,
     'block_tokens_per_second':timing['tokens_per_step']/(block_ms/1000),
     'rms_block_tokens_per_second':timing['tokens_per_step']/(rms_ms/1000),
     'rms_block_throughput_over_block':block_ms/rms_ms,
     'matched_loss':targets}
 save('results/derived.json',result)
 lines=['# Generated results','', 'Rebuild with `python scripts/rebuild.py --figures`.', '',
        '## Models below 4B — final held-out test, seed 42','',
        '| Model | Variant | Labels | NLL | PPL |','|---|---|---:|---:|---:|']
 for r in runs:lines.append(f"| {r['size']} | {NAMES[r['variant']]} | {r['labels']:,} | {r['test_nll']:.9f} | {r['test_ppl']:.5f} |")
 lines+=['','57M Full+RMS is essentially tied with Full at the endpoint; DDP versus accumulation mismatch.','',
         '## 4B — completed 9,537-step runs; last scheduled validation at step 9,500 (no test split)','',
         '| Variant | Step | Token positions | NLL | PPL |','|---|---:|---:|---:|---:|']
 for r in result['4b']['rows']:lines.append(f"| {NAMES[r['variant']]} | {r['step']} | {r['token_positions']:,} | {r['nll']:.6f} | {r['ppl']:.5f} |")
 final_eff=targets[-1]
 lines+=['','## 4B — RMS Block versus Block at the same validation NLL','',
         f"At Block's last scheduled NLL ({final_eff['target_nll']:.6f}), RMS Block reaches the same loss at an interpolated step {final_eff['rms_block_step']:.1f} instead of {final_eff['block_step']:.0f}.",'',
         '| Data-efficiency speedup | Token reduction | RMS Block throughput / Block | Time-to-quality speedup | Training-time reduction |',
         '|---:|---:|---:|---:|---:|',
         f"| {final_eff['data_efficiency_speedup']:.3f}x | {final_eff['token_reduction_pct']:.2f}% | {result['4b']['efficiency']['rms_block_throughput_over_block']:.4f} | {final_eff['time_to_quality_speedup']:.3f}x | {final_eff['training_time_reduction_pct']:.2f}% |"]
 (ROOT/'results/tables.md').write_text('\n'.join(lines)+'\n')
 return result

def figures(result):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import MaxNLocator
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,
                      'savefig.dpi':190,'axes.titleweight':'bold','svg.fonttype':'none'})
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 def finish(fig,name,note):
  note=textwrap.fill(note,width=132)
  fig.text(.02,.015,note,fontsize=9,color='#505862')
  fig.tight_layout(rect=(0,.10 if '\n' in note else .06,1,1))
  fig.savefig(out/f'{name}.png',facecolor='white')
  plt.close(fig)
 def tail_inset(ax,curves):
  xmax=max(max(x) for _,x,_ in curves);xmin=.95*xmax
  inset=ax.inset_axes([.50,.17,.46,.38])
  tail=[]
  for name,x,y in curves:
   inset.plot(x,y,color=COLORS[name],lw=1.5)
   tail.extend(v for u,v in zip(x,y) if u>=.90*xmax)
  pad=max((max(tail)-min(tail))*.12,0.002)
  inset.set_xlim(xmin,xmax);inset.set_ylim(min(tail)-pad,max(tail)+pad)
  inset.set_title('Last 5%',fontsize=8);inset.grid(alpha=.2);inset.tick_params(labelsize=7)

 # Main three-variant training and validation curves at every tested scale.
 panels={
  '57M':dict(params=56836352,runs=[('Full','evidence/small/57m_full'),('Block4','evidence/small/57m_block'),('RMS Block4','evidence/small/57m_rms_block')],small=True),
  '157M':dict(params=157433856,runs=[('Full','evidence/small/157m_full'),('Block4','evidence/small/157m_block'),('RMS Block4','evidence/small/157m_rms_block')],small=True),
  '512M (450M config)':dict(params=512416576,runs=[('Full','evidence/small/450m_full'),('Block4','evidence/small/450m_block'),('RMS Block4','evidence/small/450m_rms_block')],small=True),
  '4B':dict(params=4023411712,runs=[('Full','evidence/4b/full'),('Block4','evidence/4b/block'),('RMS Block4','evidence/4b/rms_block')],small=False),
 }
 fig,axs=plt.subplots(4,2,figsize=(11.8,15.8))
 for row,(size,panel) in enumerate(panels.items()):
  train_curves=[];valid_curves=[]
  for name,path in panel['runs']:
   train=rows(path+'/training-selected.csv');valid=rows(path+'/validation.csv')
   if panel['small']:
    tx=[float(x['tokens'])/1e9 for x in train];ty=[float(x['nll']) for x in train]
    vx=[float(x['labels'])/1e9 for x in valid]
   else:
    tx=[float(x['samples'])*8192/1e9 for x in train];ty=[float(x['loss']) for x in train]
    vx=[float(x['token_positions'])/1e9 for x in valid]
   vy=[float(x['nll']) for x in valid]
   train_curves.append((name,tx,ty));valid_curves.append((name,vx,vy))
   label={'Block4':'Block ($S=4$)','RMS Block4':'RMS Block ($S=4$)'}.get(name,name)
   axs[row,0].plot(tx,ty,label=label,color=COLORS[name],lw=1.8)
   axs[row,1].plot(vx,vy,label=label,color=COLORS[name],lw=1.8)
  axs[row,0].set_title(size+' training NLL');axs[row,1].set_title(size+' validation NLL')
  for ax in axs[row]:
   ax.set_xlabel('Training tokens / positions (billions)');ax.set_ylabel('NLL');ax.grid(alpha=.2);ax.legend()
  tail_inset(axs[row,1],valid_curves)
 finish(fig,'training_validation_nll','Validation panels enlarge the final 5% of training; training-loss panels show only the full curves.')
 gap=result['57m_optimization']['gap']
 fig,axs=plt.subplots(1,2,figsize=(11.5,4.6))
 for ax,subset in zip(axs,[gap,[r for r in gap if r['step']>=2500]]):
  ax.plot([r['labels']/1e9 for r in subset],[r['advantage'] for r in subset],color=COLORS['Full+RMS'],marker='o',ms=4)
  ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('Training labels (billions)')
  ax.set_ylabel('Validation NLL: Full − Full+RMS')
 axs[0].set_title('57M: early advantage, later convergence');axs[1].set_title('Later training (expanded vertical scale)')
 finish(fig,'57m_validation_gap','Single seed. Full: 4-GPU DDP; Full+RMS: 1 MIG GPU, accumulation 4. Final test PPL essentially tied.')

 fig,axs=plt.subplots(1,2,figsize=(10.8,4.5))
 for size,ax in zip(('57m','157m'),axs):
  r=read(f'evidence/diagnostics/{size}/summary.json')[size]['full']['output_rms']
  for kind,color in [('attention',COLORS['Full']),('ffn',COLORS['Block4'])]:
   z=[x for x in r if x['sublayer']==kind]
   ax.plot([x['transformer_layer']+1 for x in z],[x['median'] for x in z],marker='o',ms=3,label=kind.upper(),color=color)
  ax.set_yscale('log');ax.set_xlabel('Transformer layer');ax.set_ylabel('Median residual-output RMS');ax.set_title(size.upper()+' Full, final checkpoint');ax.grid(alpha=.2);ax.legend()
  ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
 finish(fig,'magnitude_by_layer','Median across valid positions in sampled chunks, not a trajectory over training. No embedding included.')

 fig,axs=plt.subplots(1,2,figsize=(10.8,4.5))
 for size,ax in zip(('57m','157m'),axs):
  data=read(f'evidence/diagnostics/{size}/summary.json')[size]
  for variant,r in data.items():
   z=sorted((x for x in r['counterfactual'] if x['metric']=='relative_rms' and x['read_kind']=='next_sublayer' and x['member_in_block']=='all'),key=lambda x:x['c'])
   # Scaling by one is exactly the identity, not an additional measured sample.
   points=sorted([(x['c'],x['median']) for x in z]+[(1.,0.)])
   ax.plot([x[0] for x in points],[x[1] for x in points],marker='o',label=NAMES[variant],color=COLORS[NAMES[variant]])
  ax.set_xscale('log',base=2);ax.set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4'])
  ax.set_xlabel('Positive multiplier on one completed source');ax.set_ylabel('Median relative RMS change at next read');ax.set_title(size.upper());ax.grid(alpha=.2);ax.legend()
 finish(fig,'scaling_counterfactual','Frozen local intervention; multiplier 1 is the identity. RMS Block median 0 has small nonzero tails. Not training causality.')

 fig,axs=plt.subplots(1,2,figsize=(11.5,4.6))
 for r in result['4b']['rows']:
  z=rows('evidence/4b/'+r['run_id']+'/validation.csv');z=[x for x in z if int(x['step'])<=result['4b']['selected_step']]
  for ax in axs:ax.plot([int(x['token_positions'])/1e9 for x in z],[float(x['nll']) for x in z],label=NAMES[r['variant']],color=COLORS[NAMES[r['variant']]])
 axs[0].set_title('4B: completed target-budget runs');axs[1].set_title('Late common validation range');axs[1].set_xlim(16,20);axs[1].set_ylim(2.91,3.07)
 for ax in axs:ax.set_xlabel('Training token positions (billions)');ax.set_ylabel('Validation NLL');ax.grid(alpha=.2);ax.legend()
 finish(fig,'4b_completed_validation','All arms stopped at step 9537; last scheduled validation step 9500. Seed 42, 64 PPUs; no independent test split.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--figures',action='store_true');a=p.parse_args()
 result=derive()
 if a.figures:figures(result)
 print(json.dumps({'57m_final_ppl_delta_pct':result['57m_optimization']['final_test_ppl_relative_reduction_pct'],
                   '57m_lower_validation_points':result['57m_optimization']['full_rms_lower'],
                   '157m':result['157m_comparison'],'450m':result['450m_comparison'],
                   '4b_last_validation_delta_pct':result['4b']['rms_vs_full_ppl_reduction_pct'],
                   '4b_same_loss_efficiency':result['4b']['efficiency']['matched_loss'][-1]},indent=2))
