#!/usr/bin/env python3
# Render four Gaussian VP/BP profile movies from authenticated arrays.
# The playback schedule chooses a preceding actual state at each uniform physical-time tick.
# Outputs are an MP4, still previews and a ledger joining encoded slots to source states.

"""Animate accepted Gaussian density profiles at a linear physical-time rate.

Plasma curves show every positive native density and every finite centered
21-cell slope. Heavy segments require effective marker count at least 32.
Gray dotted curves retain the initial profile. The playback clock is linear in physical time. Every displayed curve is a saved state. The radius axis covers the complete positive
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
    'gaussian_static_vp': ('Vlasov–Poisson (static background)','Collisionless'),
    'gaussian_expanding_vp': ('Vlasov–Poisson (expanding background)','Collisionless'),
    'gaussian_static_bp': ('Boltzmann–Poisson (static background)',r'$s=0.1$'),
    'gaussian_expanding_bp': ('Boltzmann–Poisson (expanding background)',r'$s=0.1$'),
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
    require(len(arrays['time'])==len(ledger)==entry.get('native_frames',1001) and arrays['time'][0]==0 and arrays['time'][-1]==entry['physical_time'][-1] and np.isfinite(arrays['time']).all() and (np.diff(arrays['time'])>0).all(),'Native profile timeline differs')
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


def playback_schedule(clock, physical_step=2.0, end_time=None):
    """Sample a linear physical clock using the latest saved state at each tick.

    Every displayed density/slopes array is one real saved state. Off-grid
    restart endpoints stay in the source ledger. A final off-grid accepted
    state receives one final frame. State holds report their actual source time.
    """
    require(len(clock)>=2 and clock[0]==0 and np.isfinite(clock).all()
            and np.all(np.diff(clock)>0),'Ordered native clocks required')
    require(np.isfinite(physical_step) and physical_step>0,'Positive playback step required')
    end=float(clock[-1] if end_time is None else end_time)
    require(0<end<=clock[-1],'Playback endpoint outside accepted timeline')
    # Build regular physical-time ticks independently of the number of stored restart/snapshot epochs.
    ticks=np.arange(int(np.floor(end/physical_step))+1,dtype=float)*physical_step
    if end-ticks[-1]>128*np.finfo(float).eps*max(1,abs(end)):
        ticks=np.r_[ticks,end]
    # Choose the most recent saved state at or before each tick. Preserve its own actual-time label.
    indices=np.searchsorted(clock,ticks,side='right')-1
    require(indices[0]==0 and np.all(indices>=0),'Playback source index differs')
    return ticks,indices


def main():
    """Keep drawing independent of the saved scientific data and its reduction."""
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=sorted(CASES),required=True)
    parser.add_argument('--input-directory',type=Path,default=here.parent/'data/gaussian')
    parser.add_argument('--output-directory',type=Path,default=here.parent/'generated/halos')
    parser.add_argument('--ffmpeg',help='Encoder executable, with PATH as fallback.')
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--physical-step',type=float,default=2.0)
    parser.add_argument('--fps',type=int,choices=[24,48],default=48)
    parser.add_argument('--end-time',type=float)
    parser.add_argument('--stem',help='Output stem, with case as default.')
    args=parser.parse_args()
    base=args.input_directory.resolve()
    out=args.output_directory.resolve()
    out.mkdir(parents=True,exist_ok=True)
    arrays,source,entry=load_case(base,args.case)
    comparable=[arrays]
    if args.case.startswith('gaussian_'):
        for other in json.loads((base/'cache_manifest.json').read_text())['cases']:
            if other['case']!=args.case:
                comparable.append(load_case(base,other['case'])[0])
    axes_limits=limits(comparable)
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
    label=fig.text(.97,.905,'',ha='right',va='center',fontsize=27,animated=True)
    fig.text(.5,.035,r'Heavy curves: $N_{\rm eff}\geq32$',ha='center',fontsize=17)
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
        thin,=ax.plot([],[],color='#0065FF',alpha=.65,lw=1.1,marker='.',markersize=1.5,animated=True,zorder=3)
        heavy,=ax.plot([],[],color='#0065FF',lw=3.4,animated=True,zorder=5,label=None)
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
    if args.case.startswith('gaussian_'):
        for value,x_fraction in [(-3.0,.97),(-1.5,.90),(-1.0,.97)]:
            axes[1].axhline(value,color='.58',ls='--',lw=1.4,zorder=.5)
            guide_labels.append(axes[1].annotate(
                rf'${value:g}$',xy=(x_fraction,value),
                xycoords=axes[1].get_yaxis_transform(),xytext=(0,1),
                textcoords='offset points',ha='right',va='bottom',
                fontsize=18,color='.30',animated=True,zorder=7,
                bbox={'facecolor':'white','edgecolor':'none','pad':.15,'alpha':.9}))
    fig.canvas.draw()
    # Cache the fixed axes so each frame redraws only current curves, guides and the time label.
    background=fig.canvas.copy_from_bbox(fig.bbox)
    width,height=fig.canvas.get_width_height()
    fps=args.fps
    # Map the scientific saved clock to output slots before drawing any frames.
    ticks,indices=playback_schedule(arrays['time'],args.physical_step,args.end_time)
    repeats=np.ones(len(ticks),dtype=int)
    stem=args.stem or args.case
    peak=30 if args.case=='gaussian_static_vp' else 26 if args.case=='gaussian_expanding_vp' else 5000
    preview={0:'initial',int(np.argmin(np.abs(ticks-peak))):'compression',len(ticks)-1:'final'}
    movie=out/(stem+'.mp4')
    writer=None
    if not args.preview_only:
        command=[encoder,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(movie)]
        # Stream raw RGBA pixels to the encoder instead of storing a complete movie in memory.
        writer=subprocess.Popen(command,stdin=subprocess.PIPE)
    ledger=[]
    encoded=0
    started=time.monotonic()
    try:
        # Each slot uses one unchanged source profile. Its playback/source-time difference is recorded below.
        for slot,(playback_time,index) in enumerate(zip(ticks,indices)):
            t=float(arrays['time'][index])
            if args.preview_only and slot not in preview:
                continue
            fig.canvas.restore_region(background)
            radius=np.log10(arrays['radius'][index])
            for column,(thin,heavy) in enumerate(current):
                positive=arrays['density_positive'][index] if column==0 else arrays['slope_positive_window'][index]
                values=np.log10(np.where(arrays['density'][index]>0,arrays['density'][index],1)) if column==0 else arrays['slope'][index]
                # Map actual elapsed time to yellow-to-dark-blue plasma colors consistently across the two panels.
                color=plt.get_cmap('plasma_r')(.05+.90*float(t)/float(ticks[-1]))
                thin.set_color(color)
                heavy.set_color(color)
                # A NaN breaks the plotted line where the corresponding scientific mask is false.
                thin.set_data(radius,np.where(positive,values,np.nan))
                heavy.set_data(radius,np.where(positive&arrays['mask'][index],values,np.nan))
                axes[column].draw_artist(thin)
                axes[column].draw_artist(heavy)
            for guide_label in guide_labels:
                axes[1].draw_artist(guide_label)
            label.set_text(f'$t={t:.8g}$')
            fig.draw_artist(label)
            if slot in preview:
                Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(out/(stem+'_'+preview[slot]+'.png'))
            if writer:
                raw=bytes(fig.canvas.buffer_rgba())
                for unused in range(int(repeats[slot])):
                    writer.stdin.write(raw)
            ledger.append({'playback_slot':slot,'playback_time':float(playback_time),'native_index':int(index),'actual_time':float(t),'saved_state_lag':float(playback_time-t),'source_profile':source[index]['source'],'source_sha256':source[index]['sha256'],'encoded_first_frame':encoded,'encoded_repeat_count':int(repeats[slot]),'positive_density_bins':int(arrays['density_positive'][index].sum()),'finite_slope_bins':int(arrays['slope_positive_window'][index].sum())})
            encoded+=int(repeats[slot])
    finally:
        if writer:
            # Close the pixel stream and check encoder completion before writing final movie metadata.
            writer.stdin.close()
            require(writer.wait(timeout=60)==0,'Encoder failed')
        plt.close(fig)
    if args.preview_only:
        print(json.dumps({'case':args.case,'status':'PREVIEWS_RENDERED','axes':axes_limits,'linear_slots':len(ticks)}))
        return
    path=out/(stem+'_frames.json')
    path.write_text(json.dumps(ledger,indent=2)+'\n')
    record={'case':args.case,'initial_condition_scope':'Existing truncated control: density on 7<r<13, velocity samples below3 Gaussian scales. No tail extension.','model':entry['model'],'physical_time':[0,float(ticks[-1])],'native_distinct_frames':len(set(int(i)for i in indices)),'encoded_frames':encoded,'fps':fps,'duration_seconds':encoded/fps,'pixel_size':[width,height],'axes':axes_limits,'positive_density_samples':int(arrays['density_positive'].sum()),'finite_centered_slopes':int(arrays['slope_positive_window'].sum()),'sampling':'Every positive density bin and finite centered21-cell logarithmic slope. Heavy segments have effective marker count at least 32. All density bins in each slope window are positive.','time_scope':'Constant-rate physical clock. Every plotted profile is an actual saved state. Each slot uses the latest preceding profile; the final actual accepted profile has one final slot. No density interpolation or extended final hold.', 'physical_step':args.physical_step,'physical_units_per_playback_second':args.physical_step*fps,'last_physical_interval':float(ticks[-1]-ticks[-2]),'final_slot_is_shorter_than_nominal_physical_step':bool(abs(float(ticks[-1]-ticks[-2])-args.physical_step)>128*np.finfo(float).eps*max(1,abs(float(ticks[-1])))),'final_slot_scope':'A final off-grid accepted profile occupies one frame. Its actual final physical interval can be shorter than the nominal playback step; this one bounded endpoint slot retains the final data.','maximum_native_gap':float(np.diff(arrays['time']).max()),'maximum_saved_state_lag':float(max(r['saved_state_lag']for r in ledger)),'source_native_states':len(arrays['time']),'displayed_distinct_states':len(set(int(i)for i in indices)),'off_grid_source_states_omitted_from_display':int(len(arrays['time'])-len(set(int(i)for i in indices))),'cache_sha256':entry['cache_sha256'],'source_ledger_sha256':entry['frame_ledger_sha256'],'frame_ledger_sha256':sha(path),'renderer_sha256':sha(__file__),'artifacts':{movie.name:{'sha256':sha(movie),'bytes':movie.stat().st_size}},'render_seconds':time.monotonic()-started}
    if args.case.startswith('gaussian_'):
        density=np.log10(arrays['density'][arrays['density_positive']])
        slopes=arrays['slope'][arrays['slope_positive_window']]
        record['ordinate_display']='Density outside [10^-15,10^1] and slopes outside [-10,5] lie outside the visible ordinates. All source values and masks are retained.'
        record['slope_reference_guides']={'values':[-3.0,-1.5,-1.0],'line_style':'dashed','numeric_labels':'One point above each line. Right-aligned at x=.97 for -3/-1 and x=.90 for -1.5 to separate labels.'}
        record['outside_visible_ordinates']={'positive_density_samples':int(np.count_nonzero((density < -15) | (density > 1))),'finite_slope_samples':int(np.count_nonzero((slopes < -10) | (slopes > 5)))}
    (out/(stem+'_movie.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
