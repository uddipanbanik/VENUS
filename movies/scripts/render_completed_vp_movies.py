#!/usr/bin/env python3
# Render completed VP profiles from NPZ arrays and source-clock ledgers.
# Inputs are named by cache_manifest.json. Outputs are an MP4, previews and an exact frame ledger.
# The static power-law renderer keeps the source sampling mask and measured time ordering.

"""Animate complete saved VP density profiles and centered logarithmic slopes.

Thin curves show every positive native density and every finite centered
21-cell slope. Heavy segments require effective marker count at least 32.
Static gamma=0.5 curves use plasma time colors from yellow to dark blue.
Gray dotted curves retain the initial profile. Each distinct saved clock is
shown once, with a final hold. The radius axis covers the complete positive
timeline. Gaussian movies display log10 density from -15 to 1 and signed
logarithmic slopes from -10 to 5. Dashed guides mark slopes -3, -1.5 and -1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'venus-complete-vp-movies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from PIL import Image

CASES = {
    'gaussian_static_vp': ('Vlasov–Poisson Gaussian shell: static','Collisionless'),
    'gaussian_expanding_vp': ('Vlasov–Poisson Gaussian shell: expanding','Collisionless'),
    'static_g05_vp': (r'Vlasov–Poisson, $\gamma_i=0.5$ (static background)',''),
}


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(base,case):
    """Authenticate the exact source ledger and the already reduced arrays."""
    # Authenticate both the compressed numerical cache and the source-clock ledger before using either.
    catalog=json.loads((base/'cache_manifest.json').read_text())
    entry=next(item for item in catalog['cases'] if item['case']==case)
    path=base/entry['cache']
    ledger_path=base/entry['frame_ledger']
    require(sha(path)==entry['cache_sha256'],'Movie cache SHA differs')
    require(sha(ledger_path)==entry['frame_ledger_sha256'],'Source ledger SHA differs')
    with np.load(path,allow_pickle=False) as source:
        arrays={key:source[key] for key in source.files}
    ledger=json.loads(ledger_path.read_text())
    require(len(arrays['time'])==len(ledger)==entry.get('native_frames',1001) and arrays['time'][0]==0 and arrays['time'][-1]==10000 and np.isfinite(arrays['time']).all() and (np.diff(arrays['time'])>0).all(),'Native profile timeline differs')
    require(np.isfinite(arrays['radius']).all() and (arrays['radius']>0).all(),'Invalid radii')
    require((np.diff(arrays['radius'],axis=1)>0).all(),'Unordered radius grid')
    require(np.isfinite(arrays['density']).all() and (arrays['density']>=0).all(),'Invalid density')
    require(np.isfinite(arrays['neff']).all() and (arrays['neff']>=0).all(),'Invalid effective count')
    # Keep positivity, finite derivative support and effective-count masks as distinct scientific conditions.
    require(np.array_equal(arrays['density_positive'],arrays['density']>0),'Positive-density mask differs')
    require(np.array_equal(arrays['slope_positive_window'],np.isfinite(arrays['slope'])),'Finite-slope mask differs')
    require(np.array_equal(arrays['mask'],arrays['neff']>=32),'Effective-count mask differs')
    for i,row in enumerate(ledger):
        require(row['native_index']==i and row['actual_time']==arrays['time'][i],'Source clock binding differs')
    return arrays,ledger,entry


def limits(arrays_list):
    """Include every positive density and finite signed slope across the movie."""
    # Determine radial support from all positive source cells across the compared cases.
    x=np.concatenate([np.log10(a['radius'][a['density_positive']]) for a in arrays_list])
    density=np.concatenate([np.log10(a['density'][a['density_positive']]) for a in arrays_list])
    slope=np.concatenate([a['slope'][a['slope_positive_window']] for a in arrays_list])
    def padded(values,fraction):
        lo,hi=float(values.min()),float(values.max())
        pad=max((hi-lo)*fraction,.05)
        return [lo-pad,hi+pad]
    return {'log10_radius':padded(x,.015),'log10_density':padded(density,.035),'signed_slope':padded(slope,.04)}


def frame_repeats(clock,case,fps=24):
    """Hold actual early profiles long enough to show collapse; retain every clock."""
    require(len(clock)>=2 and clock[0]==0 and np.all(np.diff(clock)>0),'Ordered native clocks required')
    repeats=np.ones(len(clock),dtype=int)
    if case.startswith('gaussian_'):
        repeats[(clock>0)&(clock<=50)]=6
        peak=30 if case=='gaussian_static_vp' else 26
        index=np.flatnonzero(clock==peak)
        require(len(index)==1,'Actual maximum-compression profile required')
        repeats[index[0]]=24
    repeats[0]=1
    # The final viewing hold repeats the same last state and leaves its physical-time label unchanged.
    repeats[-1]=round(2.4*fps)
    return repeats


def main():
    """Keep drawing independent of the saved scientific data and its reduction."""
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=sorted(CASES),required=True)
    parser.add_argument('--input-directory',type=Path,default=here.parent/'data/completed_vp')
    parser.add_argument('--output-directory',type=Path,default=here.parent/'generated/halos')
    parser.add_argument('--ffmpeg',help='Encoder executable, with PATH as fallback.')
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args()
    base=args.input_directory.resolve()
    out=args.output_directory.resolve()
    out.mkdir(parents=True,exist_ok=True)
    arrays,source,entry=load_case(base,args.case)
    comparable=[arrays]
    if args.case.startswith('gaussian_'):
        other='gaussian_expanding_vp' if args.case=='gaussian_static_vp' else 'gaussian_static_vp'
        comparable.append(load_case(base,other)[0])
    axes_limits=limits(comparable)
    if args.case=='static_g05_vp':
        axes_limits={'log10_radius':[-1.0,2.05], 'log10_density':[-9.2,0.2], 'signed_slope':[-4.65,0.3]}
    if args.case.startswith('gaussian_'):
        axes_limits['log10_density']=[-15.0,1.0]
        axes_limits['signed_slope']=[-10.0,5.0]
    encoder=args.ffmpeg or shutil.which('ffmpeg')
    if encoder is None and not args.preview_only:
        try:
            import imageio_ffmpeg
            encoder=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            parser.error('Supply --ffmpeg or install imageio-ffmpeg.')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':19,'axes.labelsize':27,'axes.titlesize':24,'axes.linewidth':1.8,'xtick.labelsize':20,'ytick.labelsize':20,'legend.fontsize':18,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(16,7.2),dpi=120)
    fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.81,wspace=.26)
    title,subtitle=CASES[args.case]
    fig.text(.5,.955,title,ha='center',va='center',fontsize=27)
    fig.text(.085,.905,subtitle,ha='left',va='center',fontsize=20)
    label=fig.text(.97,.905,'',ha='right',va='center',fontsize=27,animated=True)
    if args.case!='static_g05_vp':
        fig.text(.5,.035,r'Thin: native positive samples. Heavy: $N_{\rm eff}\geq32$. Code units: $G=1$.',ha='center',fontsize=17)
    current=[]
    guide_labels=[]
    # Draw the initial profile once as a reference. Current curves are updated with cached source values.
    x0=np.log10(arrays['radius'][0])
    for column,ax in enumerate(axes):
        positive=arrays['density_positive'][0] if column==0 else arrays['slope_positive_window'][0]
        values=np.log10(np.where(arrays['density'][0]>0,arrays['density'][0],1)) if column==0 else arrays['slope'][0]
        ax.plot(x0,np.where(positive,values,np.nan),color='.72',ls=':',lw=1.2,marker='.',markersize=1.5,zorder=1)
        ax.plot(x0,np.where(positive&arrays['mask'][0],values,np.nan),color='.55',ls=':',lw=2.5,label='Initial profile',zorder=2)
        # Thin curves show available positive/finite values. Heavy curves additionally require N_eff>=32.
        thin,=ax.plot([],[],color=plt.get_cmap('plasma_r')(0.0) if args.case=='static_g05_vp' else '#0065FF',alpha=.65,lw=1.1,marker='.',markersize=1.5,animated=True,zorder=3)
        heavy,=ax.plot([],[],color=plt.get_cmap('plasma_r')(0.0) if args.case=='static_g05_vp' else '#0065FF',lw=3.4,animated=True,zorder=5,label='_nolegend_' if args.case=='static_g05_vp' else 'Current profile')
        current.append((thin,heavy))
        ax.set_xlim(axes_limits['log10_radius'])
        ax.set_xlabel(r'$\log_{10}r$')
        ax.grid(False)
        ax.tick_params(which='both',direction='in',top=True,right=True)
        ax.tick_params(which='major',length=9,width=1.7,pad=6)
        ax.tick_params(which='minor',length=5,width=1.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    axes[0].set_ylim(axes_limits['log10_density'])
    axes[0].set_ylabel(r'$\log_{10}\rho$')
    axes[0].set_title('Density',pad=12)
    axes[0].legend(frameon=False,loc='lower left',handlelength=2.8)
    axes[1].set_ylim(axes_limits['signed_slope'])
    axes[1].set_ylabel(r'$\mathrm{d}\log\rho/\mathrm{d}\log r$')
    axes[1].set_title('Logarithmic slope',pad=12)
    if args.case.startswith('gaussian_') or args.case=='static_g05_vp':
        for value,x_fraction in [(-3.0,.22),(-1.5,.50),(-1.0,.78)]:
            axes[1].axhline(value,color='.58',ls='--',lw=1.4,zorder=.5)
            guide_labels.append(axes[1].annotate(
                rf'${value:g}$',xy=(x_fraction,value),
                xycoords=axes[1].get_yaxis_transform(),xytext=(0,1),
                textcoords='offset points',ha='center',va='bottom',
                fontsize=18,color='.30',animated=True,zorder=7,
                bbox={'facecolor':'white','edgecolor':'none','pad':.15,'alpha':.9}))
    fig.canvas.draw()
    # Cache the fixed axes so each frame redraws only current curves, guides and the time label.
    background=fig.canvas.copy_from_bbox(fig.bbox)
    width,height=fig.canvas.get_width_height()
    fps=24
    # Translate each saved-state viewing duration into an integer number of encoder frames.
    repeats=frame_repeats(arrays['time'],args.case,fps)
    peak=30 if args.case=='gaussian_static_vp' else 26 if args.case=='gaussian_expanding_vp' else 5000
    preview={0:'initial',int(np.searchsorted(arrays['time'],peak)):'compression',len(arrays['time'])-1:'final'}
    movie=out/(args.case+'.mp4')
    writer=None
    if not args.preview_only:
        command=[encoder,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(movie)]
        # Stream raw RGBA pixels to the encoder instead of storing a complete movie in memory.
        writer=subprocess.Popen(command,stdin=subprocess.PIPE)
    ledger=[]
    encoded=0
    started=time.monotonic()
    try:
        # Visit each actual source epoch in order; preview mode selects a subset of these same states.
        for index,t in enumerate(arrays['time']):
            if args.preview_only and index not in preview:
                continue
            fig.canvas.restore_region(background)
            radius=np.log10(arrays['radius'][index])
            for column,(thin,heavy) in enumerate(current):
                positive=arrays['density_positive'][index] if column==0 else arrays['slope_positive_window'][index]
                values=np.log10(np.where(arrays['density'][index]>0,arrays['density'][index],1)) if column==0 else arrays['slope'][index]
                if args.case=='static_g05_vp':
                    color=plt.get_cmap('plasma_r')(float(t)/float(arrays['time'][-1]))
                    thin.set_color(color);heavy.set_color(color)
                # A NaN breaks the plotted line where the corresponding scientific mask is false.
                thin.set_data(radius,np.where(positive,values,np.nan))
                heavy.set_data(radius,np.where(positive&arrays['mask'][index],values,np.nan))
                axes[column].draw_artist(thin)
                axes[column].draw_artist(heavy)
            for guide_label in guide_labels:
                axes[1].draw_artist(guide_label)
            label.set_text(f'$t={t:.8g}$')
            fig.draw_artist(label)
            if index in preview:
                Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(out/(args.case+'_'+preview[index]+'.png'))
            if writer:
                raw=bytes(fig.canvas.buffer_rgba())
                for unused in range(int(repeats[index])):
                    writer.stdin.write(raw)
            ledger.append({'native_index':index,'actual_time':float(t),'source_profile':source[index]['source'],'source_sha256':source[index]['sha256'],'encoded_first_frame':encoded,'encoded_repeat_count':int(repeats[index]),'positive_density_bins':int(arrays['density_positive'][index].sum()),'finite_slope_bins':int(arrays['slope_positive_window'][index].sum())})
            encoded+=int(repeats[index])
    finally:
        if writer:
            # Close the pixel stream and check encoder completion before writing final movie metadata.
            writer.stdin.close()
            require(writer.wait(timeout=60)==0,'Encoder failed')
        plt.close(fig)
    if args.preview_only:
        print(json.dumps({'case':args.case,'status':'PREVIEWS_RENDERED','axes':axes_limits}))
        return
    path=out/(args.case+'_frames.json')
    path.write_text(json.dumps(ledger,indent=2)+'\n')
    record={'case':args.case,'model':'VP','physical_time':[0,10000],'native_distinct_frames':len(arrays['time']),'encoded_frames':encoded,'fps':fps,'duration_seconds':encoded/fps,'pixel_size':[width,height],'axes':axes_limits,'positive_density_samples':int(arrays['density_positive'].sum()),'finite_centered_slopes':int(arrays['slope_positive_window'].sum()),'sampling':'Every positive density bin and finite centered21-cell logarithmic slope. Heavy segments have effective marker count at least 32. All density bins in each slope window are positive.','time_scope':'Every distinct saved profile is present in chronological order. Initial frame once, next actual saved state immediately after it. Early Gaussian states are held for visibility, and the actual compression state for one second. The final state is held2.4seconds.','cache_sha256':entry['cache_sha256'],'source_ledger_sha256':entry['frame_ledger_sha256'],'frame_ledger_sha256':sha(path),'renderer_sha256':sha(__file__),'artifacts':{movie.name:{'sha256':sha(movie),'bytes':movie.stat().st_size}},'render_seconds':time.monotonic()-started}
    if args.case=='static_g05_vp':
        record['time_color']={'colormap':'plasma_r','normalization':[0.0,10000.0],'early':'yellow','late':'dark blue'}
        record['range_source']='Example expanding gamma=0.5 BP/VP density/slope comparison: analysis/scripts/reproduce_figures.py, density[-9.2,.2], slope[-4.65,.3], log10r[-1,2.05].'
        d=np.log10(arrays['density'][arrays['density_positive']]);s=arrays['slope'][arrays['slope_positive_window']]
        record['outside_visible_ordinates']={'positive_density_samples':int(np.count_nonzero((d < -9.2) | (d > .2))),'finite_slope_samples':int(np.count_nonzero((s < -4.65) | (s > .3)))}
    if args.case.startswith('gaussian_'):
        density=np.log10(arrays['density'][arrays['density_positive']])
        slopes=arrays['slope'][arrays['slope_positive_window']]
        record['ordinate_display']='Density outside [10^-15,10^1] and slopes outside [-10,5] lie outside the visible ordinates. All source values and masks are retained.'
        record['slope_reference_guides']={'values':[-3.0,-1.5,-1.0],'line_style':'dashed','numeric_labels':'Above each guide at distinct horizontal positions.'}
        record['outside_visible_ordinates']={'positive_density_samples':int(np.count_nonzero((density < -15) | (density > 1))),'finite_slope_samples':int(np.count_nonzero((slopes < -10) | (slopes > 5)))}
    (out/(args.case+'_movie.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
