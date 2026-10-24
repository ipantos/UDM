"""Figures of the rewritten paper.

    python3 paper_figures.py <repository directory> <paper directory>

The repository directory holds the outputs of figures.py, extensions.py and assumptions.py
(figure_data/, extensions.json and assumptions.json).  The script copies the tables it needs to <paper>/figure_data, adds
a few derived tables (prefix paper_), writes PGFPlots sources to <paper>/figure_tex with
the plot style of figures.py, and compiles them to <paper>/fig1_rungs.pdf ...
fig5_functions.pdf.  It fits nothing: every curve comes from a table written by
figures.py, extensions.py or assumptions.py, or from a closed formula given below.
"""
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

REPO, PAPER = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
PAPER.mkdir(parents=True, exist_ok=True)
SRC = REPO / 'figure_data'
OUT = PAPER / 'figure_data'
TEX = PAPER / 'figure_tex'
J = json.loads((REPO / 'extensions.json').read_text())
LEVEL = '95% one-sided (Delta chi2 = 2.71)'
C_OVER_H0 = 2997.92458                        # h^-1 Mpc
C_KMS = 299792.458
Z_NODES = [.295, .510, .706, .934, 1.321, 1.484, 2.330]
Z_REF = .51

PROF_LIM = J['profile']['with_V_nonnegative']['upper_limits_c_s2'][LEVEL]
POST_LIM = J['posterior']['upper95']
MIN_LIM = J['grids']['minimal']['growing_mode_initial_rate']['all_data']['profile_upper_limits'][LEVEL]
OMEGA_M = J['lcdm']['growing_mode_initial_rate']['omega_m']

# The plot style of figures.py.  Three colours differ from that file: its blue (21618C) and
# purple (765589) are almost indistinguishable for protanopes and both are low in chroma, and its
# grey (545B63) is close to the blue.  The values below keep the hues and pass a palette check
# for all pairs on white (OKLab colour differences of at least 15, and of at least 8 under
# simulated protanopia and deuteranopia; contrast with white of at least 3).  The light and dark
# steps of blue and orange are used where a colour encodes an ordered quantity.
STYLE = r'''\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
\usetikzlibrary{arrows.meta}
\pgfplotsset{compat=1.18}
\definecolor{udmblue}{HTML}{0E5B8A}
\definecolor{udmorange}{HTML}{B75519}
\definecolor{udmpurple}{HTML}{9069A7}
\definecolor{udmgray}{HTML}{858B92}
\definecolor{udmbluelight}{HTML}{93B5CA}
\definecolor{udmbluedark}{HTML}{083753}
\definecolor{udmorangelight}{HTML}{D7A280}
\definecolor{udmorangedark}{HTML}{6E330F}
\colorlet{udmink}{black!78}
\pgfplotsset{every axis/.append style={
  font=\fontsize{9.5}{11}\selectfont,
  label style={font=\fontsize{10}{12}\selectfont},
  tick label style={font=\fontsize{9}{10}\selectfont},
  title style={font=\fontsize{10}{12}\selectfont,align=left},
  axis line style={black!65},axis x line*=bottom,axis y line*=left,
  tick align=outside,tick style={black!60},
  major tick length=2.5pt,minor tick length=1.2pt,
  grid=none,scaled ticks=false,
  legend style={draw=none,fill=white,fill opacity=.92,text opacity=1,
    font=\fontsize{9}{10.5}\selectfont,cells={anchor=west},inner sep=2pt},
  legend cell align=left,
  every axis plot/.append style={line join=round},
}}
\begin{document}
'''
SMALL = r'\fontsize{9}{10.5}\selectfont'


def read(name, folder=None):
    with open((folder or SRC) / (name + '.csv')) as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def write(name, columns):
    np.savetxt(OUT / (name + '.csv'), np.column_stack(list(columns.values())), delimiter=',',
               header=','.join(columns), comments='', fmt='%.10g')


def copy(*names):
    for name in names:
        shutil.copy(SRC / (name + '.csv'), OUT / (name + '.csv'))


def tag(value):
    """Column suffix used by extensions.py for a sound speed, e.g. 6.00e-07 -> 6p00em07."""
    return f'{value:.2e}'.replace('.', 'p').replace('-', 'm')


def sci(x, digits=2):
    e = math.floor(math.log10(abs(x)))
    m = round(x/10**e, digits - 1)
    if m >= 10:
        m, e = m/10, e + 1
    return (f'{m:.{digits - 1}f}' + rf'\times10^{{{e}}}') if abs(m - 1) > 1e-9 else rf'10^{{{e}}}'


counter = 0
TINT95, TINT68 = 20, 42                       # per cent of the colour in the 95% and 68% bands


def bands(name, color='udmblue', x='z', median=True):
    """68% and 95% bands and the median, as in figures.py."""
    global counter
    counter += 1
    key, result = f'b{counter}', ''
    for level, opacity in (('95', TINT95), ('68', TINT68)):
        for side in ('lo', 'hi'):
            result += (rf'\addplot[draw=none,forget plot,name path={key}{side}{level}] '
                       rf'table[col sep=comma,x={x},y={side}{level}] {{../figure_data/{name}.csv}};' + '\n')
        result += rf'\addplot[draw=none,fill={color}!{opacity},forget plot] fill between[of={key}lo{level} and {key}hi{level}];' + '\n'
    if median:
        result += rf'\addplot[{color},line width=1.2pt,forget plot] table[col sep=comma,x={x},y=median] {{../figure_data/{name}.csv}};' + '\n'
    return result


def figure(name, body):
    (TEX / (name + '.tex')).write_text('\\input{plot_style.tex}\n' + body + '\n\\end{document}\n')


# ----------------------------------------------------------------------------
# Fig. 1: the three cases of Sec. III
# ----------------------------------------------------------------------------
def fig_rungs():
    a = np.geomspace(.03, 1., 241)
    chaplygin = np.sqrt((.7/.3)*a**6/(1 + (.7/.3)*a**6))           # c_s^2/(1-c_s^2) = (0.7/0.3) a^6: w = -0.7 today
    minimal = np.sqrt(MIN_LIM*a**6/(1 - MIN_LIM + MIN_LIM*a**6))    # the same law, c_s^2 = MIN_LIM today
    constant = np.full_like(a, math.sqrt(PROF_LIM))
    # the limits for galaxies that respond to gravity alone (assumptions.py)
    asm = json.loads((REPO / 'assumptions.json').read_text())
    min_tracer = asm['tracer']['minimal']['pressureless fraction']['profile_upper_limits'][LEVEL]
    gen_tracer = asm['tracer_general']['pressureless component']['upper_limits_c_s2'][LEVEL]
    minimal_tracer = np.sqrt(min_tracer*a**6/(1 - min_tracer + min_tracer*a**6))
    # Jeans threshold c_J = sqrt(1+q) aH/k for LambdaCDM with Omega_m = 0.3 and k = 0.2 h/Mpc:
    # (1+q)(aH/H0)^2 = (3/2) Omega_m/a
    jeans = np.sqrt(1.5*.3/a)/(C_OVER_H0*.2)
    write('paper_rungs', {'a': a, 'chaplygin': chaplygin, 'minimal': minimal, 'constant': constant,
                          'minimal_tracer': minimal_tracer, 'jeans': jeans, 'floor': np.full_like(a, 1e-8)})
    zticks = [0, 1, 2.33, 5, 10, 30]
    body = r'''\begin{tikzpicture}
\begin{axis}[width=8.5cm,height=7.6cm,xmode=log,ymode=log,
  xmin=.03,xmax=1,ymin=1e-8,ymax=3,
  xtick={.03,.1,.3,1},xticklabels={0.03,0.1,0.3,1},
  ytick={1e-8,1e-6,1e-4,1e-2,1},
  xlabel={Scale factor $a$},ylabel={Sound speed $c_s$ (units of $c$)},
  legend image code/.code={\draw[#1] (0cm,0cm) -- (.5cm,0cm);},
  legend style={at={(.01,.985)},anchor=north west,font=\fontsize{8.6}{10}\selectfont,fill=none}]
\addplot[draw=none,forget plot,name path=jtop] table[col sep=comma,x=a,y=jeans] {../figure_data/paper_rungs.csv};
\addplot[draw=none,forget plot,name path=jbot] table[col sep=comma,x=a,y=floor] {../figure_data/paper_rungs.csv};
\addplot[draw=none,fill=black!7,forget plot] fill between[of=jbot and jtop];
\addplot[black!40,line width=.5pt,forget plot] table[col sep=comma,x=a,y=jeans] {../figure_data/paper_rungs.csv};
\addplot[udmgray,dashed,line width=1.3pt] table[col sep=comma,x=a,y=chaplygin] {../figure_data/paper_rungs.csv};
\addlegendentry{Chaplygin gas}
\addplot[udmorange,line width=1.5pt] table[col sep=comma,x=a,y=minimal] {../figure_data/paper_rungs.csv};
\addlegendentry{Minimal model}
\addplot[udmblue,line width=1.5pt] coordinates {(''' + f'{1/3.33:.5f},{math.sqrt(PROF_LIM):.6g}) (1,{math.sqrt(PROF_LIM):.6g}' + r''')};
\addlegendentry{General model}
\addplot[udmorange,densely dotted,line width=1.5pt,forget plot] table[col sep=comma,x=a,y=minimal_tracer] {../figure_data/paper_rungs.csv};
\addplot[udmblue,densely dotted,line width=1.5pt,forget plot] coordinates {(''' + f'{1/3.33:.5f},{math.sqrt(gen_tracer):.6g}) (1,{math.sqrt(gen_tracer):.6g}' + r''')};
\addlegendimage{black!70,densely dotted,line width=1.5pt}
\addlegendentry{pressureless galaxies}
\node[anchor=south east,text=udmink,font=''' + SMALL + r''',align=right] at (axis cs:.283,1.3e-8)
  {structure grows\\at $k=0.2\,h\,\mathrm{Mpc}^{-1}$};
\node[anchor=north west,text=black,font=\fontsize{10}{12}\selectfont,rotate=-9.5] at (axis cs:.0315,5.4e-3) {$c_s=c_J$};
\draw[black!45,densely dotted] (axis cs:.3003,1e-8) -- (axis cs:.3003,3);
\draw[black!60,-{Stealth[length=4pt]}] (axis cs:.3003,3.5e-7) -- (axis cs:.56,3.5e-7);
\node[anchor=west,text=black!70,font=''' + SMALL + r'''] at (axis cs:.56,3.5e-7) {fitted};
\end{axis}
\begin{axis}[width=8.5cm,height=7.6cm,xmode=log,ymode=log,
  xmin=.03,xmax=1,ymin=''' + f'{1e-8*C_KMS:.6g},ymax={3*C_KMS:.6g}' + r''',
  axis x line*=top,axis y line*=right,
  xtick={''' + ','.join(f'{1/(1 + z):.5f}' for z in zticks) + r'''},
  xticklabels={''' + ','.join(f'{z:g}' for z in zticks) + r'''},
  xminorticks=false,
  ytick={1e-2,1,1e2,1e4},
  xlabel={Redshift $z$},ylabel={$c_s$ ($\mathrm{km\,s^{-1}}$)}]
\end{axis}
\end{tikzpicture}'''
    figure('fig1_rungs', body)


# ----------------------------------------------------------------------------
# Fig. 2: what a sound speed does to the growth observable
# ----------------------------------------------------------------------------
GROUPS = ((.232, 'mark=*,mark size=2.3pt,mark options={draw=white,fill=black,line width=.5pt}', 'filled circles'),
          (.155, 'mark=square*,mark size=2.1pt,mark options={draw=black,fill=white,line width=.6pt}', 'open squares'),
          (.116, 'mark=triangle*,mark size=3pt,mark options={draw=white,fill=black,line width=.5pt}', 'triangle'))


def fig_response():
    copy('ext_response_constant', 'ext_response_minimal', 'asm_response_constant', 'asm_response_minimal')
    data = read('ext_response_data')
    minimal = read('ext_response_minimal')
    order = np.argsort(minimal['z'])
    reference = np.interp(data['z'], minimal['z'][order], minimal['reference'][order])
    for k, _, _ in GROUPS:
        rows = np.abs(data['keff'] - k) < .002
        write(f'paper_response_data_k{k:.3f}'.replace('.', 'p'),
              {'z': data['z'][rows], 'ratio_a': data['ratio'][rows], 'error_a': data['ratio_error'][rows],
               'ratio_b': data['observed'][rows]/reference[rows], 'error_b': data['sigma'][rows]/reference[rows]})
    assert sum(int(np.sum(np.abs(data['keff'] - k) < .002)) for k, _, _ in GROUPS) == 17
    values_a = J['figure']['constant_sound_speed_values']          # 1e-7, posterior limit, profile limit, 1e-6
    values_b = J['figure']['minimal_model_values_today']           # 1e-7, 1e-6, profile limit, 1e-5
    assert values_a[2] == PROF_LIM and values_b[2] == MIN_LIM
    tracer = json.loads((REPO / 'assumptions.json').read_text())['figure']
    assert tracer['constant_sound_speed'] == PROF_LIM and tracer['minimal_model_value_today'] == MIN_LIM
    # the scalar in the two-fluid tables follows the one-fluid tables (the pressureless fraction changes it little)
    for name, prefix, value in (('constant', 'cs2_', PROF_LIM), ('minimal', 'cs2today_', MIN_LIM)):
        one, two = read(f'ext_response_{name}'), read(f'asm_response_{name}')
        column = f'ratio_{prefix}{tag(value)}_k0p232'
        assert np.allclose(one['z'], two['z']) and np.max(np.abs(one[column] - two['scalar'])) < .03, name

    def panel(table, prefix, values, suffix, title, first, hue):
        shades = (f'udm{hue}light', f'udm{hue}', f'udm{hue}dark')
        text = r'\nextgroupplot[title={' + title + '},'
        text += r'ylabel={Prediction / prediction for $c_s=0$},' if first else ''
        text += ']\n' + r'\addplot[black!45,forget plot] coordinates {(0,1) (2.06,1)};' + '\n'
        for n, (value, shade) in enumerate(zip(values, shades)):
            column = f'ratio_{prefix}{tag(value)}_k0p232'
            label = (r'$c_s^2=' if n == 0 else '$') + rf'{sci(value)}$' + (r' (limit)' if n == 1 else '')
            text += (rf'\addplot[{shade},line width=1.4pt] table[col sep=comma,x=z,y={column}] '
                     rf'{{../figure_data/{table}.csv}};' + '\n' + rf'\addlegendentry{{{label}}}' + '\n')
        for k, style in ((.155, 'dashed'), (.116, 'densely dotted')):
            column = f'ratio_{prefix}{tag(values[1])}_k{k:.3f}'.replace('.', 'p')
            text += (rf'\addplot[udm{hue},{style},line width=1pt,forget plot] table[col sep=comma,x=z,y={column}] '
                     rf'{{../figure_data/{table}.csv}};' + '\n')
        text += (r'\addplot[black!70,dashdotted,line width=1pt,forget plot] table[col sep=comma,x=z,y=pressureless] '
                 rf'{{../figure_data/{table.replace("ext_", "asm_")}.csv}};' + '\n')
        for k, mark, _ in GROUPS:
            name = f'paper_response_data_k{k:.3f}'.replace('.', 'p')
            text += (rf'\addplot[only marks,black,{mark},forget plot,error bars/.cd,y dir=both,y explicit,'
                     r'error bar style={line width=.5pt},error mark options={mark size=1.4pt}] '
                     rf'table[col sep=comma,x=z,y=ratio_{suffix},y error=error_{suffix}] {{../figure_data/{name}.csv}};' + '\n')
        return text

    body = r'''\begin{tikzpicture}
\begin{groupplot}[group style={group name=g,group size=2 by 1,horizontal sep=34pt},width=8.9cm,height=6.9cm,
  xmin=0,xmax=2.06,ymin=.35,ymax=1.82,xtick={0,.5,1,1.5,2},ytick={.4,.6,.8,1,1.2,1.4,1.6},
  xlabel={Redshift $z$},clip mode=individual,
  legend columns=3,legend image code/.code={\draw[#1] (0cm,0cm) -- (.42cm,0cm);},
  legend style={at={(.99,.985)},anchor=north east,font=\fontsize{8.8}{10.3}\selectfont,fill=none,
    /tikz/every even column/.append style={column sep=5pt}}]
'''
    body += panel('ext_response_constant', 'cs2_', [values_a[0], values_a[2], values_a[3]], 'a',
                  r'(a) Constant sound speed', True, 'blue')
    body += panel('ext_response_minimal', 'cs2today_', values_b[1:], 'b',
                  r'(b) Minimal model, $c_s\propto a^3$: values today', False, 'orange')
    body += r'''\end{groupplot}
\newcommand{\mk}[1]{\tikz[baseline=-.55ex]{\pgfsetplotmarksize{2pt}\pgfuseplotmark{#1}}}
\newcommand{\sample}[1]{\tikz[baseline=-.55ex]{\draw[black!70,line width=1pt,#1] (0,0) -- (16pt,0);}}
\path (g c1r1.south west) -- (g c2r1.south east) coordinate[midway] (mid);
\node[anchor=north,font=''' + SMALL + r''',align=center,inner sep=1pt] at ([yshift=-34pt]mid)
  {Wavenumber $k$ in $h\,\mathrm{Mpc}^{-1}$:\quad \sample{}\;\mk{*}\;$0.232$\qquad \sample{dashed}\;\mk{square}\;$0.155$\qquad
   \sample{densely dotted}\;\mk{triangle*}\;$0.116$\qquad\qquad
   \sample{dashdotted}\; pressureless component, middle $c_s^2$, $0.232$};
\end{tikzpicture}'''
    figure('fig2_growth_response', body)


# ----------------------------------------------------------------------------
# Fig. 3: the general model, the data and LambdaCDM
# ----------------------------------------------------------------------------
def fig_observations():
    names = ('bao_dm_curve', 'bao_dh_curve', 'supernova_curve', 'bao_dm_points', 'bao_dh_points',
             'bao_dv_points', 'supernova_points', 'growth_points')
    copy(*names)
    lcdm = read('ext_lcdm_curves_growing_mode_initial_rate')
    at = lambda z, column: np.interp(z, lcdm['z'], lcdm[column])
    for name, column in (('bao_dm', 'dm'), ('bao_dh', 'dh'), ('bao_dv', 'dv'), ('supernova', 'mu')):
        pts = read(name + '_points')
        write(f'paper_lcdm_{name}', {'z': pts['z'], 'offset': (at(pts['z'], column) - pts['median'])/pts['sigma']})
    pts, lg = read('growth_points'), read('ext_lcdm_growth_points_growing_mode_initial_rate')
    assert np.allclose(pts['z'], lg['z'])
    write('paper_lcdm_growth', {'z': pts['z'], 'offset': (lg['prediction'] - pts['median'])/pts['sigma']})
    write('paper_lcdm_growth_curve', {'z': lcdm['z'], 'growth': lcdm['growth']})

    def rng(name, columns):
        t = read(name)
        return min(t[c].min() for c in columns), max(t[c].max() for c in columns)

    def data_points(name, color='black', mark='*', size='1.8pt'):
        return rf'''\addplot[only marks,{color},mark={mark},mark size={size},
  error bars/.cd,y dir=both,y explicit,error bar style={{line width=.55pt}},error mark options={{mark size=1.5pt}}]
  table[col sep=comma,x=z,y=observed,y error=sigma] {{../figure_data/{name}.csv}};
'''

    def prediction_points(name, color='udmblue'):
        return rf'''\addplot[only marks,{color}!45,mark=none,line width=2pt,forget plot,
  error bars/.cd,y dir=both,y explicit,error bar style={{line width=2pt}},error mark=none]
  table[col sep=comma,x=z,y=median,y error plus expr=\thisrow{{hi95}}-\thisrow{{median}},
  y error minus expr=\thisrow{{median}}-\thisrow{{lo95}}] {{../figure_data/{name}.csv}};
\addplot[only marks,{color},mark=square*,mark size=1.6pt,forget plot,
  error bars/.cd,y dir=both,y explicit,error bar style={{line width=1.6pt}},error mark=none]
  table[col sep=comma,x=z,y=median,y error plus expr=\thisrow{{hi68}}-\thisrow{{median}},
  y error minus expr=\thisrow{{median}}-\thisrow{{lo68}}] {{../figure_data/{name}.csv}};
'''

    def residual(name, color='black', mark='*'):
        return rf'''\addplot[only marks,{color},mark={mark},mark size=1.8pt,
  error bars/.cd,y dir=both,y explicit,error bar style={{line width=.5pt}},error mark options={{mark size=1.4pt}}]
  table[col sep=comma,x=z,y=residual,y error expr=1] {{../figure_data/{name}.csv}};
'''

    def lcdm_dash(name, color):
        style = 'only marks,mark=-,mark size=4pt,line width=1.1pt' if name.endswith('bao_dv') else 'dashed,line width=.9pt'
        return (rf'\addplot[{color},{style}] '
                rf'table[col sep=comma,x=z,y=offset] {{../figure_data/{name}.csv}};' + '\n')

    dm_top = 1.15*max(rng('bao_dm_curve', ['hi95'])[1], rng('bao_dm_points', ['observed'])[1])
    mu_lo, mu_hi = rng('supernova_points', ['observed'])
    g_lo = min(rng('growth_points', ['lo95'])[0], rng('growth_points', ['observed'])[0] - .11)
    g_hi = max(rng('growth_points', ['hi95'])[1], rng('growth_points', ['observed'])[1] + .18)
    body = r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 2,horizontal sep=38pt,vertical sep=13pt},
  width=5.95cm,xmin=0,xmax=2.43,xtick={0,1,2},
  tick label style={font=\fontsize{9}{10}\selectfont},
  legend style={font=\fontsize{8.5}{9.5}\selectfont}]
\nextgroupplot[height=5.8cm,ymin=0,ymax=''' + f'{dm_top:.1f}' + r''',ytick={0,10,20,30,40},xticklabels=\empty,
  title={(a) DESI DR2 BAO},ylabel={Distance / $r_d$},legend pos=north west]
'''
    body += bands('bao_dm_curve') + bands('bao_dh_curve', 'udmorange')
    body += data_points('bao_dm_points', 'udmblue') + '\\addlegendentry{$D_M/r_d$}\n'
    body += data_points('bao_dh_points', 'udmorange', 'triangle*', '2.2pt') + '\\addlegendentry{$D_H/r_d$}\n'
    body += data_points('bao_dv_points', 'udmpurple', 'diamond*', '2.4pt') + '\\addlegendentry{$D_V/r_d$}\n'
    body += r'''\nextgroupplot[height=5.8cm,ymin=''' + f'{mu_lo - 1.2:.1f},ymax={mu_hi + 1.0:.1f}' + r''',ytick={36,39,42,45},xticklabels=\empty,
  title={(b) Union3 supernovae},ylabel={Binned distance modulus}]
''' + bands('supernova_curve') + data_points('supernova_points', 'black', '*', '1.5pt')
    body += r'''\nextgroupplot[height=5.8cm,ymin=''' + f'{g_lo - .05:.2f},ymax={g_hi + .22:.2f}' + r''',ytick={.2,.4,.6,.8},xticklabels=\empty,
  title={(c) Growth measurements},ylabel={$f_{\mathrm g}\sigma_8$},legend pos=north east]
''' + prediction_points('growth_points')
    bar = lambda style: (r'\addlegendimage{legend image code/.code={\draw[' + style
                         + r'] (.3cm,-.13cm) -- (.3cm,.13cm);}}' + '\n')
    body += bar('udmblue!45,line width=2pt') + r'\addlegendentry{95\% of the posterior}' + '\n'
    body += bar('udmblue,line width=1.6pt') + r'\addlegendentry{68\% of the posterior}' + '\n'
    body += data_points('growth_points', 'black', 'o', '2.2pt') + '\\addlegendentry{Measurements}\n'
    body += (r'\addplot[black!60,dashed,line width=.9pt] table[col sep=comma,x=z,y=growth] '
             r'{../figure_data/paper_lcdm_growth_curve.csv};' + '\n' + r'\addlegendentry{$\Lambda$CDM}' + '\n')
    panels = (([('bao_dm_points', 'udmblue', '*'), ('bao_dh_points', 'udmorange', 'triangle*'),
                ('bao_dv_points', 'udmpurple', 'diamond*')],
               [('paper_lcdm_bao_dm', 'udmblue!75'), ('paper_lcdm_bao_dh', 'udmorange!75'), ('paper_lcdm_bao_dv', 'udmpurple!75')], True),
              ([('supernova_points', 'black', '*')], [('paper_lcdm_supernova', 'black!60')], False),
              ([('growth_points', 'black', 'o')], [('paper_lcdm_growth', 'black!60')], False))
    for points, dashes, ylabel in panels:
        body += r'\nextgroupplot[height=3.45cm,ymin=-3,ymax=3,ytick={-2,0,2},xlabel={Redshift $z$}'
        body += r',ylabel={$\Delta/\sigma$}' if ylabel else ''
        body += r''']
\path[fill=black!5] (axis cs:0,-1) rectangle (axis cs:2.43,1);
\addplot[black!50,densely dashed,forget plot] coordinates {(0,0) (2.43,0)};
'''
        body += ''.join(lcdm_dash(*item) for item in dashes)
        body += ''.join(residual(*item) for item in points)
    body += '\\end{groupplot}\n\\end{tikzpicture}'
    figure('fig3_observations', body)


# ----------------------------------------------------------------------------
# Fig. 4: the limit on a constant sound speed
# ----------------------------------------------------------------------------
def fig_sound_speed():
    copy('ext_profile_with_V_nonnegative', 'ext_profile_without_V_restriction', 'ext_profile_without_lowest_z',
         'sound_speed_density', 'asm_profile_general_pressureless')
    density = read('sound_speed_density')
    plateau = float(np.mean(density['density'][density['u'] < -9.0]))
    prof = read('ext_profile_with_V_nonnegative')
    write('paper_profile_likelihood', {'u': prof['u'], 'curve': plateau*np.exp(-.5*(prof['delta_chi2'] - prof['delta_chi2'][0]))})
    peak = float(density['density'].max())
    top = 1.75*peak
    u_prof, u_post = math.log10(PROF_LIM), math.log10(POST_LIM)
    general = json.loads((REPO / 'assumptions.json').read_text())['tracer_general']
    u_tracer = math.log10(general['pressureless component']['upper_limits_c_s2'][LEVEL])
    body = r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=50pt},width=8.5cm,height=7cm,
  xlabel={$\log_{10}c_s^2$}]
\nextgroupplot[xmin=-9,xmax=-4.5,ymin=0,ymax=7,xtick={-9,-8,-7,-6,-5},ytick={0,1,2,3,4,5,6,7},
  ylabel={Rise of the profile $\chi^2$ above its minimum},
  title={(a) Profile likelihood},legend style={at={(.03,.98)},anchor=north west}]
\addplot[black!45,line width=.5pt,forget plot] coordinates {(-9,2.706) (-4.95,2.706)};
\addplot[black!45,line width=.5pt,densely dashed,forget plot] coordinates {(-9,3.841) (-4.95,3.841)};
\node[anchor=south west,text=black!70,font=''' + SMALL + r'''] at (axis cs:-8.98,2.706) {$2.71$: 95\% limit};
\node[anchor=south west,text=black!70,font=''' + SMALL + r'''] at (axis cs:-8.98,3.841) {$3.84$};
\addplot[udmblue,line width=1.5pt] table[col sep=comma,x=u,y=delta_chi2] {../figure_data/ext_profile_with_V_nonnegative.csv};
\addlegendentry{General model}
\addplot[udmorange,dashed,line width=1.2pt] table[col sep=comma,x=u,y=delta_chi2] {../figure_data/ext_profile_without_V_restriction.csv};
\addlegendentry{without $V\geq0$}
\addplot[udmpurple,densely dashdotted,line width=1.2pt] table[col sep=comma,x=u,y=delta_chi2] {../figure_data/ext_profile_without_lowest_z.csv};
\addlegendentry{without $z=0.07$}
\addplot[udmgray,densely dotted,line width=1.7pt] table[col sep=comma,x=u,y=delta_chi2] {../figure_data/asm_profile_general_pressureless.csv};
\addlegendentry{pressureless galaxies}
''' + rf'''\draw[udmblue,line width=.8pt] (axis cs:{u_prof:.4f},0) -- (axis cs:{u_prof:.4f},2.706);
\addplot[udmblue,only marks,mark=*,mark size=1.8pt,forget plot] coordinates {{({u_prof:.4f},2.706)}};
\draw[udmgray,line width=.8pt] (axis cs:{u_tracer:.4f},0) -- (axis cs:{u_tracer:.4f},2.706);
\addplot[udmgray,only marks,mark=*,mark size=1.8pt,forget plot] coordinates {{({u_tracer:.4f},2.706)}};
\node[anchor=west,text=udmink,font={SMALL},rotate=90,inner sep=1pt] at (axis cs:{u_tracer + .13:.4f},.12) {{${sci(10**u_tracer)}$}};
\node[anchor=east,text=udmink,font={SMALL},inner sep=1pt] (lim) at (axis cs:-7.05,2.05) {{$c_s^2<{sci(PROF_LIM)}$}};
\draw[black!60,line width=.5pt,-{{Stealth[length=3.5pt]}}] (lim.east) -- (axis cs:{u_prof - .05:.4f},2.62);
''' + r'''\nextgroupplot[xmin=-12,xmax=-5.6,ymin=0,ymax=''' + f'{top:.3f}' + r''',
  xtick={-12,-10,-8,-6},yticklabel style={/pgf/number format/fixed},
  ylabel={Posterior density of $\log_{10}c_s^2$},
  title={(b) Posterior density},legend style={at={(.03,.97)},anchor=north west}]
\addplot[ybar interval,fill=black!13,draw=black!30,line width=.3pt,area legend]
  table[col sep=comma,x=u,y=density] {../figure_data/sound_speed_density.csv};
\addlegendentry{Posterior samples}
''' + rf'''\addplot[udmgray,dashed,line width=1.3pt] coordinates {{({u_post:.4f},0) ({u_post:.4f},{1.32*peak:.3f})}};
\addlegendentry{{Posterior limit, ${sci(POST_LIM)}$}}
''' + r'''\addplot[udmblue,line width=1.3pt] table[col sep=comma,x=u,y=curve] {../figure_data/paper_profile_likelihood.csv};
\addlegendentry{Profile likelihood, rescaled}
''' + rf'''\addplot[udmblue,line width=1.3pt,forget plot] coordinates {{({u_prof:.4f},0) ({u_prof:.4f},{1.32*peak:.3f})}};
\node[anchor=west,text=udmink,font={SMALL},rotate=90] at (axis cs:{u_prof + .17:.4f},{.42*peak:.3f}) {{profile limit, ${sci(PROF_LIM)}$}};
''' + r'''\end{groupplot}
\end{tikzpicture}'''
    figure('fig4_sound_speed', body)


# ----------------------------------------------------------------------------
# Fig. 5: the reconstructed expansion history and field functions
# ----------------------------------------------------------------------------
def fig_functions():
    copy('ext_deceleration', 'ext_prefactor_full_range')
    pot = read('ext_potential_full_range')
    floor = .02                                                    # lower edge of the logarithmic axis
    write('paper_potential_full_range', {key: (np.maximum(value, floor) if key != 'z' else value) for key, value in pot.items()})
    z = np.linspace(0, 2.33, 234)
    e2 = OMEGA_M*(1 + z)**3 + 1 - OMEGA_M
    write('paper_lcdm_q', {'z': z, 'q': 1.5*OMEGA_M*(1 + z)**3/e2 - 1})
    pre = read('ext_prefactor_full_range')
    node_f = np.interp(Z_NODES, pre['z'], pre['median'])
    node_v = np.interp(Z_NODES, pot['z'], pot['median'])
    dec = read('ext_deceleration')
    write('paper_nodes', {'z': np.array(Z_NODES), 'q': np.interp(Z_NODES, dec['z'], dec['median']),
                          'prefactor': node_f, 'potential': node_v})
    # The best fit, from its seven nodes: eps_H = dlnH/dlna is a not-a-knot cubic spline in x = ln(1+z);
    # q = -1 - eps_H, (H/H0)^2 = exp(-2 int_0^x eps_H dx'); for a constant sound speed
    # f a^3 is proportional to (1+q)(H/H0)^2/(1+z)^3 and V to [3 - 2(1+q)/(1-c_s^2)](H/H0)^2.
    best = J['bestfit']['with_V_nonnegative']
    spline = CubicSpline(np.log1p(Z_NODES), best['eps_nodes'], bc_type='not-a-knot')
    antiderivative = spline.antiderivative()
    zb = np.expm1(np.linspace(0, math.log(3.33), 1401))

    def at(zz):
        x = np.log1p(zz)
        q = -1 - spline(x)
        energy = np.exp(-2*(antiderivative(x) - antiderivative(0.)))
        return q, (1 + q)*energy/(1 + zz)**3, (3 - 2*(1 + q)/(1 - best['c_s2']))*energy
    q_best, f_best, v_best = at(zb)
    _, f_ref, v_ref = at(np.array([Z_REF]))
    assert abs(q_best[0] - best['q0']) < 1e-6 and v_best.min() > -1e-6*v_ref[0]
    write('paper_best_fit', {'z': zb, 'q': q_best, 'prefactor': f_best/f_ref[0],
                             'potential': np.maximum(v_best/v_ref[0], .5*floor)})

    def frame(ymin, ymax):
        """Shading of the extrapolated range, z < 0.295, behind the bands."""
        return (rf'\pgfonlayer{{axis background}}\path[fill=black!6] (axis cs:0,{ymin}) rectangle (axis cs:.295,{ymax});'
                rf'\draw[black!35,line width=.4pt] (axis cs:.295,{ymin}) -- (axis cs:.295,{ymax});\endpgfonlayer' + '\n')

    marks = lambda column: (r'\addplot[udmblue,only marks,mark=*,mark size=1.6pt,forget plot] '
                            rf'table[col sep=comma,x=z,y={column}] {{../figure_data/paper_nodes.csv}};' + '\n')
    fit = lambda column, keep='': (rf'\addplot[black,line width=.6pt{keep}] '
                                   rf'table[col sep=comma,x=z,y={column}] {{../figure_data/paper_best_fit.csv}};' + '\n')
    anchor = r'\addplot[black,only marks,mark=*,mark size=2.2pt,forget plot] coordinates {(.51,1)};' + '\n'
    body = r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1,horizontal sep=42pt},width=6.05cm,height=6.3cm,
  set layers=standard,
  xmin=0,xmax=2.33,xtick={0,.5,1,1.5,2},xlabel={Redshift $z$},
  legend style={font=\fontsize{8.6}{10}\selectfont}]
\nextgroupplot[ymin=-.95,ymax=.75,ytick={-.8,-.4,0,.4},ylabel={Deceleration parameter $q$},
  title={(a) Expansion history},legend style={at={(.97,.03)},anchor=south east}]
''' + bands('ext_deceleration') + frame(-.95, .75) + marks('q')
    body += r'''\addplot[black!45,densely dotted,line width=.8pt,forget plot] coordinates {(0,.5) (2.33,.5)};
\addplot[udmorange,dashed,line width=1pt] table[col sep=comma,x=z,y=q] {../figure_data/paper_lcdm_q.csv};
\addlegendentry{$\Lambda$CDM}
''' + fit('q') + r'''\addlegendentry{best fit}
\addlegendimage{area legend,fill=udmblue!''' + str(TINT68) + r''',draw=none}\addlegendentry{68\%}
\addlegendimage{area legend,fill=udmblue!''' + str(TINT95) + r''',draw=none}\addlegendentry{95\%}
\node[anchor=south east,text=black!70,font=''' + SMALL + r'''] at (axis cs:2.31,.5) {$w_\phi=0$};
\nextgroupplot[ymode=log,ymin=.4,ymax=3.4,ytick={.5,1,2,3},yticklabels={0.5,1,2,3},
  ylabel={$\mathcal{F}$: normalized $f a^3$},
  title={(b) Prefactor},legend style={at={(.97,.96)},anchor=north east}]
''' + bands('ext_prefactor_full_range') + frame(.4, 3.4) + marks('prefactor')
    body += r'''\addplot[udmorange,dashed,line width=1pt] coordinates {(0,1) (2.33,1)};
\addlegendentry{$f\propto a^{-3}$}
''' + fit('prefactor', ',forget plot') + anchor
    body += r'''\nextgroupplot[ymode=log,ymin=.02,ymax=14,ytick={.03,.1,.3,1,3,10},yticklabels={0.03,0.1,0.3,1,3,10},
  ylabel={$\mathcal{V}$: normalized $V$},
  title={(c) Potential},legend style={at={(.42,.975)},anchor=north}]
''' + bands('paper_potential_full_range') + frame(.02, 14) + marks('potential')
    body += r'''\addplot[udmorange,dashed,line width=1pt] coordinates {(0,1) (2.33,1)};
\addlegendentry{constant $V$}
''' + fit('potential', ',forget plot') + anchor + '\\end{groupplot}\n\\end{tikzpicture}'
    figure('fig5_functions', body)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / 'plot_style.tex').write_text(STYLE)
    fig_rungs()
    fig_response()
    fig_observations()
    fig_sound_speed()
    fig_functions()
    for source in sorted(TEX.glob('fig[1-5]_*.tex')):
        print('compiling', source.name, flush=True)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(PAPER), source.name],
                       cwd=TEX, check=True, stdout=subprocess.DEVNULL)
    for extra in PAPER.glob('fig[1-5]_*'):
        if extra.suffix in ('.aux', '.log'):
            extra.unlink()
    print('figures written to', PAPER)
