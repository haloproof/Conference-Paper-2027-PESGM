"""Public Figure 2 data export and CSV-only plotting; no model evaluation.

Export saved results and plot:
  python scripts/plot_training.py --max-seconds 55
Replot the public CSV without access to private runs or model checkpoints:
  python scripts/plot_training.py --csv results/training_results.csv --force
New protocol: HNN heat pretraining 200 + daily calibration 100 epochs;
MLPs 300 epochs. Both HNN stages report daily damage on all 160 validation days.
The plot requires 300 validation epochs; old heat-only logs are rejected.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pypdf import PdfReader

EXPERIMENT = 'estimator-compare-300epoch'
PROTOCOL = {'stage1': 200, 'stage2': 100, 'mlp': 300}
PLOT_COLUMNS = {'plot_epoch', 'plot_epoch_limit', 'stage_epoch_limit'}
METHODS = [('HNN', 'HNN-compare', 'HNN-model'),
           ('SOC-only', 'SOC-only', 'SOC-only'),
           ('SOC-NN', 'SOC NN', 'SOC-NN'),
           ('End-to-End NN', 'End-to-End NN', 'End-to-End-NN')]
FIELDS = ['model', 'source_method', 'stage', 'epoch', 'plot_epoch',
          'plot_epoch_limit', 'stage_epoch_limit', 'split', 'target', 'unit',
          'sample_count', 'raw_sse', 'raw_mse', 'raw_mae', 'objective',
          'calendar_raw_mse', 'cyclic_raw_mse', 'heat_raw_mse_w2', 'heat_sample_count', 'heat_objective',
          'temperature_objective', 'calendar_objective', 'record_type',
          'is_regular_checkpoint', 'is_best_checkpoint', 'display_in_fig2',
          'source_run', 'source_file']


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


STYLE_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'result-figure-style.json'
# User-specified colors, without saturation or brightness adjustment.
# Shared by Figures 2-5; all colored curves use full opacity.
CASE_STUDY_COLORS = {'M1': '#808080', 'M2': '#1E90FF', 'M3': '#FF8C00',
                     'M4': '#FF4500', 'M5': '#32CD32'}


def figure_style():
    cfg = read_json(STYLE_PATH)
    # Keep the requested Case Study sizing in the results plotting scripts.
    cfg['figure_aspect_ratio'].update(fig2=[4, 1.6], fig3=[4, 1.6], fig4=[4, 1.6])
    return cfg


def figure_font(role):
    cfg = figure_style()
    # Matplotlib uses 72 pt/in; TeX uses 72.27 pt/in. Compensate for
    # inclusion at exactly 252 TeX pt so printed sizes match the JSON.
    return cfg['font'][role + '_pt'] * cfg['canvas_width_inches'] * 72 / cfg['template']['column_width_tex_pt']


def figure_size(number):
    cfg = figure_style()
    w, h = cfg['figure_aspect_ratio']['fig' + str(number)]
    width = cfg['canvas_width_inches']
    return width, width * h / w


def setup_publication_style():
    cfg = figure_style()
    font = cfg['font']['family']
    try:
        path = font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        font = cfg['font']['fallback']
        path = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family':font, 'font.size':figure_font('tick_label'),
        'mathtext.fontset':'stix', 'axes.titlesize':figure_font('panel_title'),
        'axes.labelsize':figure_font('axis_label'), 'xtick.labelsize':figure_font('tick_label'),
        'ytick.labelsize':figure_font('tick_label'), 'legend.fontsize':figure_font('legend'),
        'axes.linewidth':0.8, 'lines.linewidth':1.7, 'pdf.fonttype':42, 'ps.fonttype':42,
        'savefig.transparent':False, 'figure.facecolor':'white',
        'axes.spines.top':False, 'axes.spines.right':False})
    return {'family':font, 'path':path}


def write_caption_style():
    cfg = figure_style()['font']
    bs = chr(92)
    text = ('% Generated from result-figure-style.json by the result plotting scripts.\n'
            '% Scoped to result figures; IEEEtran body, Fig.1 and tables remain unchanged.\n'
            + bs + 'newcommand{' + bs + 'ResultFigureCaption}[1]{' + bs + 'let' + bs + 'ResultSavedFootnotesize' + bs + 'footnotesize\n'
            + bs + 'renewcommand{' + bs + 'footnotesize}{' + bs + 'fontsize{'
            + str(cfg['caption_pt']) + '}{' + str(cfg['caption_baseline_pt']) + '}'
            + bs + 'selectfont}\n' + bs + 'caption{#1}' + bs + 'let' + bs + 'footnotesize' + bs + 'ResultSavedFootnotesize}\n')
    path = STYLE_PATH.with_suffix('.tex')
    if not path.exists() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')


def export_training(root, out, metadata):
    """Export completed new-protocol logs, retaining stage-local epoch values."""
    root, out = Path(root), Path(out)
    rows = []
    common_data = None
    sources = {}
    plot_limit = None
    for label, source_method, folder in METHODS:
        path = root / 'runs' / EXPERIMENT / folder
        manifest, complete, history = [read_json(path / name) for name in
                                     ('manifest.json', 'complete.json', 'history.json')]
        if history['method'] != source_method or complete['method'] != source_method:
            raise ValueError(f'{label}: method identity mismatch')
        sig = manifest['signature']
        cfg = sig['config']
        data = sig['dataset']
        key = (data['complete_sha256'], data['metadata_sha256'])
        if common_data is None:
            common_data = key
        if common_data != key:
            raise ValueError('Models must share exactly the same dataset/split manifest')
        every = cfg['evaluate_every']
        if every != 5:
            raise ValueError('Expected five-epoch frozen evaluation schedule')
        # Read the limits from each run manifest, then verify that the logs really
        # belong to the requested 200+100/300 protocol rather than the old runs.
        configured_limits = {s: cfg['training'][s]['epochs'] for s in PROTOCOL}
        if configured_limits != PROTOCOL or any(type(v) is not int for v in configured_limits.values()):
            raise ValueError(f'{label}: expected HNN 200+100 and MLP 300 epoch limits')
        total_limit = configured_limits['stage1'] + configured_limits['stage2']
        if total_limit != configured_limits['mlp']:
            raise ValueError('HNN cumulative and MLP epoch limits must agree')
        if plot_limit is None:
            plot_limit = total_limit
        if total_limit != plot_limit:
            raise ValueError('Inconsistent cumulative plotting limits')
        stages = ['stage1', 'stage2'] if label == 'HNN' else ([] if label == 'SOC-only' else ['mlp'])
        stage_limits = {s: configured_limits[s] for s in stages}
        for stage, limit in stage_limits.items():
            epochs = [e['epoch'] for e in history['epochs'] if e['stage'] == stage]
            if epochs != list(range(1, limit + 1)):
                raise ValueError(f'{label}/{stage}: requires all {limit} completed epochs')
            available = {r['epoch'] for r in history['frozen'] if r['stage'] == stage}
            required = set(range(0, limit + 1, every)) | {limit}
            if not required.issubset(available):
                raise ValueError(f'{label}/{stage}: missing frozen training points')
        if complete['stop_reason'] != ('zero_trainable_parameters' if label == 'SOC-only' else 'epoch_limit'):
            raise ValueError(f'{label}: incomplete or unexpected stop reason')
        selected = complete['selected']
        if file_hash(path / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise ValueError(f'{label}: selected checkpoint hash mismatch')
        selected_epochs = {s: r['epoch'] for s, r in history['selected'].items()}
        records = {}

        def add(stage, epoch, split, metrics, record_type):
            if stage not in (stage_limits if stages else {'fixed': 0}):
                raise ValueError(f'{label}: unexpected stage {stage}')
            limit = stage_limits.get(stage, 0)
            if type(epoch) is not int or not 0 <= epoch <= limit:
                raise ValueError(f'{label}/{stage}: epoch exceeds declared stage limit')
            daily = True  # Both HNN stages now evaluate the complete daily mapping.
            if metrics['target'] != ('daily_total_degradation' if daily else 'equivalent_heat'):
                raise ValueError('Metric target mismatch')
            if metrics['unit'] != ('pu^2' if daily else 'W^2'):
                raise ValueError('Metric unit mismatch')
            count = metrics['count']
            if count <= 0 or (daily and count != (640 if split == 'train' else 160)):
                raise ValueError('Unexpected evaluation sample count')
            for field in ('raw_sse', 'raw_mse', 'raw_mae', 'objective'):
                if not math.isfinite(metrics[field]) or metrics[field] < 0:
                    raise ValueError(f'Invalid {field}')
            if not math.isclose(metrics['raw_sse'] / count, metrics['raw_mse'], rel_tol=1e-12, abs_tol=0):
                raise ValueError('Raw SSE/count disagrees with raw MSE')
            regular = stage == 'fixed' or epoch % every == 0 or epoch == limit
            offset = configured_limits['stage1'] if label == 'HNN' and stage == 'stage2' else 0
            row = dict(model=label, source_method=source_method, stage=stage, epoch=epoch,
                plot_epoch=offset + epoch, plot_epoch_limit=total_limit, stage_epoch_limit=limit,
                split=split, target=metrics['target'], unit=metrics['unit'], sample_count=count,
                raw_sse=metrics['raw_sse'], raw_mse=metrics['raw_mse'], raw_mae=metrics['raw_mae'],
                objective=metrics['objective'],
                calendar_raw_mse=metrics.get('calendar_raw_mse', ''),
                cyclic_raw_mse=metrics.get('cyclic_raw_mse', ''),
                heat_raw_mse_w2=metrics.get('heat_metrics', {}).get('raw_mse', ''),
                heat_sample_count=metrics.get('heat_metrics', {}).get('count', ''),
                heat_objective=metrics.get('q', ''), temperature_objective=metrics.get('t', ''),
                calendar_objective=metrics.get('calendar', ''), record_type=record_type,
                is_regular_checkpoint=int(regular),
                is_best_checkpoint=int(selected_epochs.get(stage) == epoch),
                display_in_fig2=int(daily and split == 'validation' and (stage == 'fixed' or epoch > 0)),
                source_run=f'{EXPERIMENT}/{folder}', source_file='history.json')
            k = (stage, epoch, split)
            if k in records:
                previous = records[k]
                for field in ('raw_mse', 'raw_sse', 'objective'):
                    if not math.isclose(previous[field], row[field], rel_tol=1e-12, abs_tol=0):
                        raise ValueError('Duplicate epoch records disagree')
            records[k] = row

        # Every validation epoch already uses frozen weights. No online training
        # objective is converted into an invented full-training raw MSE.
        for e in history['epochs']:
            add(e['stage'], e['epoch'], 'validation', e['validation'], 'epoch_end_validation')
        for point in history['frozen']:
            for split, metrics in point['metrics'].items():
                add(point['stage'], point['epoch'], split, metrics, 'frozen_checkpoint')
        rows.extend(records[k] for k in sorted(records))
        sources[label] = dict(run_id=f'{EXPERIMENT}/{folder}', seed=cfg['seed'],
            stages=stage_limits, selected_epochs=selected_epochs,
            hashes={name: file_hash(path / name) for name in ('manifest.json', 'complete.json', 'history.json')},
            selected_checkpoint_sha256=selected['checkpoint_sha256'])
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'training_results.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata.update(sources=sources, rows=len(rows), dataset_complete_sha256=common_data[0],
        dataset_metadata_sha256=common_data[1], csv='training_results.csv', csv_sha256=file_hash(csv_path),
        plotted_quantity='Full-validation daily total-degradation raw MSE (pu^2); 160 days',
        epoch_axis='cumulative_training_epoch', plot_epoch_limit=plot_limit,
        plotted_points='HNN stage1 1-200 plus stage2 201-300; MLPs 1-300; SOC-only constant',
        stage1_policy='Daily damage MSE evaluated every epoch; heat objective and heat MSE retained separately',
        validation_policy='Figure 2 uses every validation epoch; training checkpoints remain available every five epochs',
        pretraining='Stage 2 starts from selected heat-pretraining weights; its local epoch 0 is stored but not plotted; plotted Stage 2 epochs are 201-300',
        epoch_interpretation='Equal total epoch counts are not equal compute, sample or supervision budgets',
        legacy_csv=False, test_used=False)
    return csv_path


def training_plot_data(rows):
    """Require all 300 real validation epochs; never substitute old heat MSE."""
    series = {}
    for label, _, _ in METHODS:
        data = [r for r in rows if r['model'] == label and r['display_in_fig2'] == '1']
        if not data or any(r['split'] != 'validation' or r['target'] != 'daily_total_degradation'
                           or r['unit'] != 'pu^2' or int(r['sample_count']) != 160 for r in data):
            raise ValueError(f'{label}: full 160-day validation daily MSE required; rerun HNN with daily Stage 1 evaluation')
        stages = [('stage1',200,0),('stage2',100,200)] if label=='HNN' else [('fixed',0,0)] if label=='SOC-only' else [('mlp',300,0)]
        segments=[]
        for stage, limit, offset in stages:
            items=sorted((r for r in data if r['stage']==stage),key=lambda r:int(r['epoch']))
            local=[int(r['epoch']) for r in items]
            expected=[0] if stage=='fixed' else list(range(1,limit+1))
            if local!=expected:
                raise ValueError(f'{label}/{stage}: missing or duplicate daily validation epochs')
            values=[float(r['raw_mse']) for r in items]
            if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid daily MSE')
            x=[offset+e for e in local]
            if any(int(r['plot_epoch'])!=e for r,e in zip(items,x)):raise ValueError('Incorrect cumulative epoch')
            segments.append(dict(stage=stage,epochs=x,values=values))
        if len(data)!=sum(len(s['epochs']) for s in segments):raise ValueError('Unexpected stage records')
        series[label]=segments
    return dict(series=series,plot_epoch_limit=300,split='validation')


def plot_training_csv(csv_path, target, metadata=None):
    """Single 4:1.6 figure with a broken logarithmic y axis, no data removal."""
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    layout=training_plot_data(rows)
    setup_publication_style()
    all_values=[v for segs in layout['series'].values() for seg in segs for v in seg['values']]
    gap_low,gap_high=1e-10,1e-8
    if any(v<=0 for v in all_values):
        raise ValueError('Logarithmic broken axis requires positive MSE; no epsilon substitution')
    if any(gap_low<v<gap_high for v in all_values):
        raise ValueError('The proposed axis break contains data; change limits rather than hide observations')
    low_min=min(3e-12,min(all_values)/1.15)
    high_max=max(1e-6,max(all_values)*1.15)
    # One common x coordinate system. Two axes implement only the y-axis cut;
    # equal physical height per logarithmic decade is retained on both sides.
    ratios=[math.log10(high_max/gap_high),math.log10(gap_low/low_min)]
    fig,(upper,lower)=plt.subplots(2,1,figsize=figure_size(2),sharex=True,
                                 gridspec_kw={'height_ratios':ratios})
    fig.subplots_adjust(left=.19,right=.95,bottom=.25,top=.73,hspace=.35)
    upper.set_ylim(gap_high,high_max)
    lower.set_ylim(low_min,gap_low)
    styles={'HNN':(CASE_STUDY_COLORS['M5'],'o'),
            'SOC-NN':(CASE_STUDY_COLORS['M2'],'s'),
            'End-to-End NN':(CASE_STUDY_COLORS['M4'],'^')}
    fixed=layout['series']['SOC-only'][0]['values'][0]
    handles={}
    for ax in (upper,lower):
        for label in ('HNN','SOC-NN','End-to-End NN'):
            color,marker=styles[label]
            for j,seg in enumerate(layout['series'][label]):
                marks=[i for i,e in enumerate(seg['epochs']) if e%5==0]
                line,=ax.plot(seg['epochs'],seg['values'],color=color,alpha=1.0,lw=1.35,
                    marker=marker,markevery=marks,ms=2.8,mfc='white',mew=.65,
                    label=label if j==0 else None)
                if ax is upper and j==0:handles[label]=line
        ref=ax.axhline(fixed,color=CASE_STUDY_COLORS['M1'],alpha=1.0,ls='--',lw=1.4,label='SOC-only')
        if ax is upper:handles['SOC-only']=ref
        ax.set(xlim=(0,300),yscale='log')
        ax.grid(False,axis='both',which='both')
        ax.set_axisbelow(True)
        ax.axvline(200,color='#AAAAAA',ls=':',lw=.8)
    upper.set_yticks([1e-8,1e-7,1e-6])
    upper.spines['bottom'].set_visible(False)
    upper.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
    lower.spines['top'].set_visible(False)
    lower.set_xticks([0,50,100,150,200,250,300])
    lower.set_xlabel('Epoch')
    fig.text(.065,.49,'Validation MSE\n'+r'(pu$^2$)',ha='center',va='center',rotation='vertical',fontsize=figure_font('axis_label'))
    # Short slanted strokes at the omitted interval, not additional subpanels.
    for ax,y in ((upper,0),(lower,1)):
        ax.plot([-.012,.012],[y-.018,y+.018],transform=ax.transAxes,
                color='black',lw=1,clip_on=False)
    order=['HNN','SOC-only','SOC-NN','End-to-End NN']
    fig.legend([handles[m] for m in order],['HPNN' if m == 'HNN' else m for m in order],loc='upper center',
               bbox_to_anchor=(.57,1.01),ncol=2,frameon=False,
               handlelength=1.8,columnspacing=1.2,labelspacing=.25)
    # HNN stage segments are kept separate: Stage 2 restores best Stage 1 weights.
    fig.savefig(target,format='pdf',metadata={
        'Title':'Validation MSE of different estimation on 300 epochs',
        'Subject':'Single broken logarithmic y axis; empty open interval (1e-10,1e-8) omitted. All 300 validation epochs retained; markers every five epochs.'})
    plt.close(fig)
    pdf=PdfReader(target);box=pdf.pages[0].mediabox
    width,height=figure_size(2)
    if len(pdf.pages)!=1 or not math.isclose(float(box.width)/float(box.height),width/height,abs_tol=1e-9):raise ValueError('Expected configured Figure 2 aspect ratio')
    if metadata is not None:
        metadata.update(split='validation',sample_count=160,units='pu^2',figure_inches=list(figure_size(2)), typography=figure_style(), style_config_sha256=file_hash(STYLE_PATH),
            layout='single plot with broken y axis',title=None,
            caption='Validation MSE of different estimation on 300 epochs. HNN stage 1: 200 epochs; stage 2: 100 epochs.',
            ylabel='Validation MSE (pu^2)',yscale='log',y_axis_omitted_open_interval=[gap_low,gap_high],
            data_points_in_omitted_interval=0,markers_every_epochs=5,
            epoch_axis='cumulative_training_epoch',csv_sha256=file_hash(csv_path),
            transition='Stage 2 restores best Stage 1 weights; no artificial connecting segment',
            plotted={m:sum(len(seg['epochs']) for seg in segs) for m,segs in layout['series'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--csv', type=Path, help='Plot an existing public CSV without reading runs')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=55)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('positive finite --max-seconds required')
    write_caption_style()
    started = time.monotonic()
    out = (args.out_dir or args.root / 'results').resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ['fig2_training.pdf', 'training_metadata.json'] + ([] if args.csv else ['training_results.csv'])
    if not args.force and any((out / n).exists() for n in names):
        raise FileExistsError('Output exists; use --force for an intentional replacement')
    metadata = {'script_sha256': file_hash(__file__)}
    with tempfile.TemporaryDirectory(prefix='.training-figure-', dir=out) as temporary:
        dest = Path(temporary)
        csv_path = args.csv.resolve() if args.csv else export_training(args.root, dest, metadata)
        plot_training_csv(csv_path, dest / 'fig2_training.pdf', metadata)
        metadata['elapsed_seconds'] = time.monotonic() - started
        if metadata['elapsed_seconds'] > args.max_seconds:
            raise TimeoutError('Figure build exceeded --max-seconds')
        (dest / 'training_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        for name in names:
            os.replace(dest / name, out / name)
    print(f'Created Figure 2 from frozen saved metrics in {out}; no inference or fitting')


if __name__ == '__main__':
    main()
