'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''
import io
import os
import subprocess
from base64 import b64decode
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, Spinner, FileInput, Button, TextInput
)

from matgen.base import CellComplex

if not os.path.exists('/app/results/'):
    os.mkdir('/app/results/')

os.chdir('/app/results/')

def omega(cell_complex: CellComplex, n0: int):
    """
    """ 
    D0 = 0
    D1 = 0
    D2 = 0
    w = 0
    number_of_internal_GBs = 0

    for gb in cell_complex._GBs.values():
        if not gb.is_external:
            number_of_internal_GBs += 1
            cell_ids = np.array(gb.incident_ids)
            d = (cell_ids > n0).sum()
            if d == 2:
                D2 += 1
            elif d == 1:
                D1 += 1
            elif d == 0:
                D0 += 1
            else:
                raise ValueError("Number of new grains must be 0, 1 or 2")

    D2 = D2 / number_of_internal_GBs
    D1 = D1 / number_of_internal_GBs    
    D0 = D0 / number_of_internal_GBs

    n = cell_complex.grainnb # number of grains
    m = n - n0 # number of new grains
    p = m / n # fraction of new grains
    
    D0r = (1 - p) * (1 - p)
    D1r = 2 * (1 - p) * p
    D2r = p * p
    
    if p == 0 or p == 1:
        w = -9999
    elif D1 <= D1r:
        w = 1 - D1 / D1r
    elif D1 > D1r:
        w = D0 * D2 / D0r / D2r - 1
    
    return w

    # GB_frac = round(len(e_spec) / len(e_int), 3)



def extract_seeds(complex: CellComplex, seeds_pathname: str):
    """
    """
    N = complex.grainnb
    seeds = [complex._grains[g_id].seed for g_id in range(1, N + 1)]
    np.savetxt(seeds_pathname, seeds, fmt='%.12f')


def get_xy_for_edges(complex: CellComplex, e_ids: list):
    """
    """
    xs = []
    ys = []

    for e in complex.get_many('e', e_ids):
        xs.append([v.x for v in complex.get_many('v', e.v_ids)])
        ys.append([v.y for v in complex.get_many('v', e.v_ids)])

    return xs, ys


# Choose working directory (TextInput)
def check_wdir(attrname, old, new):
    """
    """
    if not os.path.exists(new): # Check directory exists
        os.makedirs(new)
        div_message.text = f'Created directory: {new}'
    elif os.listdir(new): # Check directory is empty
        div_message.text = f'Directory is not empty: {new}'


input_wdir = TextInput(title='Save .tess files to', value=os.getcwd())
input_wdir.on_change('value', check_wdir)


# Choose results filename (TextInput)
def check_resfilename(attrname, old, new):
    """
    """
    if os.path.exists(new): # Check directory exists
        div_message.text = f'File exists: {new}'

# input_resfilename = TextInput(title='Save results to', value='/app/results/results.txt')
# input_resfilename.on_change('value', check_resfilename)


# Load initial complex (FileInput)
def load_initial_complex(attrname, old, new):
    """
    """
    # get working directory
    wdir = input_wdir.value
    # decode file content
    file = io.StringIO(b64decode(new).decode(encoding='utf-8'))
    # create CellComplex example
    global initial_complex
    global pairs_with_special_GB
    global NUMBER_OF_GRAINS
    try:
        initial_complex = CellComplex.from_tess_file(file)
        # save initial seeds to a file
        seeds_pathname = os.path.join(wdir, 'seeds.txt')
        extract_seeds(initial_complex, seeds_pathname)
        # set measures
        initial_complex.set_measures_from_coo()
        # set special from ori
        initial_complex.set_theta_from_ori()
        initial_complex.reset_special(lower_thrd=15, warn_external=False)
        pairs_with_special_GB = []
        for special_id in initial_complex.get_special_ids():
            pair = set(initial_complex._GBs[special_id].incident_ids)
            pairs_with_special_GB.append(pair)
        NUMBER_OF_GRAINS = initial_complex.grainnb
        div_message.text = f'Complex loaded! Seeds saved: {seeds_pathname}'
    except:
        div_message.text = f'Complex not loaded! Check .tess file!'

    if initial_complex.dim == 2:
        ext_ids = initial_complex.get_external_ids('e')
        int_ids = initial_complex.get_internal_ids('e')
        spec_ids = initial_complex.get_special_ids()

        ax = initial_complex.plot_edges(ext_ids, color='k')
        initial_complex.plot_edges(int_ids, color='b', ax=ax)
        initial_complex.plot_edges(spec_ids, color='r', ax=ax)
        plt.savefig(os.path.join(wdir, 'initial-complex.png'), dpi=300)

        xs, ys = get_xy_for_edges(initial_complex, ext_ids)
        complex0_ext.data = dict(x=xs, y=ys)
        complex_ext.data = dict(x=xs, y=ys)
        xs, ys = get_xy_for_edges(initial_complex, int_ids)
        complex0_int.data = dict(x=xs, y=ys)
        complex_int.data = dict(x=xs, y=ys)
        # reset special GB on figure
        complex0_spec.data = dict(x=[], y=[])
        complex_spec.data = dict(x=[], y=[])
        complex_new_seeds.data = dict(x=[], y=[])

input_complex = FileInput(multiple=False)
input_complex.on_change('value', load_initial_complex)


# Load grain size (FileInput)
def load_grain_size(attrname, old, new):
    """
    """
    # decode file content
    file = io.StringIO(b64decode(new).decode(encoding='utf-8'))
    try:
        # parse file
        data = np.loadtxt(file)
        # Check data dim is 1 and its len is n
        n = len(initial_complex._grains)
        if n != len(data) or len(data.shape) != 1:
            raise ValueError
        else: # set sizes
            for i in range(n):
                initial_complex._grains[i + 1].set_measure(data[i])
            div_message.text = f'Grain sizes loaded!'
    except NameError:
        div_message.text = f'Load complex first!'
    except:
        div_message.text = f'Check file with grain size!'

input_size = FileInput(multiple=False)
input_size.on_change('value', load_grain_size)


# Load special GBs (FileInput)
def load_special(attrname, old, new):
    """
    """
    # decode file content
    file = io.StringIO(b64decode(new).decode(encoding='utf-8'))
    try:
        # parse file
        special_ids = np.loadtxt(file, dtype=int).tolist()
        # set special
        initial_complex.reset_special(special_ids=special_ids, warn_external=False)
        pairs_with_special_GB = []
        # find pairs of grains with special GB
        for special_id in initial_complex.get_special_ids():
            pair = set(initial_complex._GBs[special_id].incident_ids)
            pairs_with_special_GB.append(pair)
        # Success message
        div_message.text = f'Special GBs set!'
    except NameError:
        div_message.text = f'Load complex first!'
    except:
        div_message.text = f'Check file with special GB ids!'
    
    if initial_complex.dim == 2:
        spec_ids = initial_complex.get_special_ids()
        xs, ys = get_xy_for_edges(initial_complex, spec_ids)
        complex0_spec.data = dict(x=xs, y=ys)
        complex_spec.data = dict(x=xs, y=ys)

input_special = FileInput(multiple=False)
input_special.on_change('value', load_special)


def create_new_complex(
    n: int,
    neper_id: int,
    dim: int
):
    """
    """
    wdir = input_wdir.value
    seeds_filename = os.path.join(wdir, 'seeds.txt')
    # output_file = os.path.join(wdir, f'n{n}-id{neper_id}-{dim}D.tess')
    output_file = os.path.join(wdir, 'final-complex.tess')

    height = spinner_height.value
    width = spinner_width.value

    com_line_list = [
        'neper', '-T', '-n', str(n),
        '-id', str(neper_id), '-dim', str(dim),
        '-morphooptiini', f'coo:file({seeds_filename})',
        '-domain', f'square({height},{width})',
        #'-statcell', 'size',
        '-o', output_file.rstrip('.tess')
    ]
      
    run = subprocess.run(com_line_list, capture_output=True)
    # print(run.stdout)
    cell_complex = CellComplex.from_tess_file(
        output_file, with_cell_size=False)
    cell_complex.set_measures_from_coo()
    return cell_complex


def run_simulation(event):
    """
    """
    wdir = input_wdir.value
    seeds_filename = os.path.join(wdir, 'seeds.txt')
    resultspath = os.path.join(wdir, 'results.txt')
    cell_complex = initial_complex
    n0 = initial_complex.grainnb 
    n = n0
    k = spinner_new_seeds.value # TODO: k may be different for each step
    ps = []
    ws = []
    results = []
    with open(resultspath, 'w') as file:
        file.write('m,frac,vol_frac,omega,j0,j1,j2,j3,S,HAGBs_frac\n')
    for step_idx in tqdm(range(spinner_steps.value)):
        # Generate k new random seeds
        new_seeds = np.array(
            cell_complex.get_new_random_seeds(
                k=k, spec_prob=spinner_spec_prob.value
            )
        )
        # Append new seeds to seeds.txt
        with open(seeds_filename, 'a') as file:
            np.savetxt(file, new_seeds, fmt='%.12f')
        
        # # Plot new seeds
        # xs, ys = new_seeds[:, 0].tolist(), new_seeds[:, 1].tolist()
        # complex_new_seeds.data = dict(x=xs, y=ys)
        
        # Generate new complex from seeds.txt
        n += k
        # div_progress.text = f'Progress {n}/{n0 + spinner_steps.value}'
        # logging.info(f'Step {step_idx + 1}/{spinner_steps.value}')
        # print(f'Step {step_idx + 1}/{spinner_steps.value}')
        cell_complex = create_new_complex(
            n, neper_id=1, dim=initial_complex.dim
        )
        
        special_ids = []
        # Set special GBs from initial complex
        for cell in cell_complex._GBs.values():
            if set(cell.incident_ids) in pairs_with_special_GB:
                    special_ids.append(cell.id)
                    # if not cell.is_external:
                    #     cell.set_special(True)
        # Set special GBs for new grains
        # And calculate total size of new grains
        new_grains_total_size = 0
        for grain_id in range(n0 + 1, n + 1):
            new_grains_total_size += cell_complex._grains[grain_id].size
            gb_ids = cell_complex._grains[grain_id].gb_ids
            special_ids += gb_ids
            # for gb_id in gb_ids:
            #     cell = cell_complex._GBs[gb_id]
            #     if not cell.is_external:
            #         cell.set_special(True)
        cell_complex.reset_special(special_ids=set(special_ids), warn_external=False)
        initial_grains_total_size = 0
        for grain_id in range(1, n0 + 1):
            initial_grains_total_size += cell_complex._grains[grain_id].size
                
        vol_fraction = new_grains_total_size / (
            new_grains_total_size + initial_grains_total_size)
        ps.append((n-n0)/n)
        w = omega(cell_complex, n0)
        ws.append(w)
        state = cell_complex.to_TJset()
        result = (
            str(n - n0),
            f'{(n - n0)/n:.5f}',
            f'{vol_fraction:.5f}',
            f'{w:.5f}',
            f'{state.j0:.5f}',
            f'{state.j1:.5f}',
            f'{state.j2:.5f}',
            f'{state.j3:.5f}',
            f'{state.S:.5f}',
            f'{state.p:.5f}'
        )
        with open(resultspath, 'a') as file:
            file.write(','.join(result) + '\n')

        results.append(
            (n - n0, (n - n0)/n, vol_fraction, w,
             state.j0, state.j1, state.j2, state.j3, state.S, state.p)
        )
        logging.warning(f'Step {step_idx}')
    
    # # Plot final complex
    # p_vs_w.data = dict(x=ws, y=ps)
    # # FIXME: bind folder with results to access from host
    # resultspath = os.path.join(wdir, 'results.txt')
    # pd.DataFrame(
    #     results,
    #     columns=[
    #         'new_seeds_number',
    #         'new_seeds_frac',
    #         'vol_fraction',
    #         'omega',
    #         'j0', 
    #         'j1',
    #         'j2',
    #         'j3',
    #         'HAGBs_frac'
    #     ]
    # ).to_csv(input_resfilename.value, index=False, float_format='%.5f')

    # np.savetxt('/app/results/p_vs_w.txt', np.array([*zip(ws,ps)]), fmt='%.8f')
    if initial_complex.dim == 2:
        ext_ids = cell_complex.get_external_ids('e')
        int_ids = cell_complex.get_internal_ids('e')
        spec_ids = cell_complex.get_special_ids()

        ax = cell_complex.plot_edges(ext_ids, color='k')
        cell_complex.plot_edges(int_ids, color='b', ax=ax)
        cell_complex.plot_edges(spec_ids, color='r', ax=ax)
        plt.savefig(os.path.join(wdir, 'final-complex.png'), dpi=300)

        xs, ys = get_xy_for_edges(cell_complex, ext_ids)
        complex_ext.data = dict(x=xs, y=ys)
        xs, ys = get_xy_for_edges(cell_complex, int_ids)
        complex_int.data = dict(x=xs, y=ys)
        xs, ys = get_xy_for_edges(cell_complex, spec_ids)
        complex_spec.data = dict(x=xs, y=ys)
    


button_start = Button(label="Start simulation", width=200)
button_start.on_click(run_simulation)

def run_simulation_step(event):
    """
    """
    global cell_complex
    global NUMBER_OF_GRAINS
    wdir = input_wdir.value
    seeds_filename = os.path.join(wdir, 'seeds.txt')
    k = spinner_new_seeds.value # TODO: k may be different for each step
    if NUMBER_OF_GRAINS == initial_complex.grainnb:
        cell_complex = initial_complex
    # Generate k new random seeds
    new_seeds = np.array(
        cell_complex.get_new_random_seeds(
            k=k, spec_prob=spinner_spec_prob.value
        )
    )
    # Append new seeds to seeds.txt
    with open(seeds_filename, 'a') as file:
        np.savetxt(file, new_seeds, fmt='%.12f')
    
    # Plot new seeds
    try:
        xs, ys = new_seeds[:, 0].tolist(), new_seeds[:, 1].tolist()
        complex_new_seeds.data = dict(x=xs, y=ys)
    except IndexError:
        div_message.text = f'No new seeds!'

    
    # Generate new complex from seeds.txt
    NUMBER_OF_GRAINS += k
    cell_complex = create_new_complex(
        NUMBER_OF_GRAINS, neper_id=1, dim=initial_complex.dim
    )
    
    special_ids = []
    # Set special GBs from initial complex
    for cell in cell_complex._GBs.values():
        if set(cell.incident_ids) in pairs_with_special_GB:
                special_ids.append(cell.id)
                # if not cell.is_external:
                #     cell.set_special(True)
    # Set special GBs for new grains
    for grain_id in range(initial_complex.grainnb + 1, NUMBER_OF_GRAINS + 1):
        gb_ids = cell_complex._grains[grain_id].gb_ids
        special_ids += gb_ids
        # for gb_id in gb_ids:
        #     cell = cell_complex._GBs[gb_id]
        #     if not cell.is_external:
        #         cell.set_special(True)
    cell_complex.reset_special(special_ids=set(special_ids), warn_external=False)

    # Plot new complex
    if initial_complex.dim == 2:
        ext_ids = cell_complex.get_external_ids('e')
        int_ids = cell_complex.get_internal_ids('e')
        spec_ids = cell_complex.get_special_ids()
        xs, ys = get_xy_for_edges(cell_complex, ext_ids)
        complex_ext.data = dict(x=xs, y=ys)
        xs, ys = get_xy_for_edges(cell_complex, int_ids)
        complex_int.data = dict(x=xs, y=ys)
        xs, ys = get_xy_for_edges(cell_complex, spec_ids)
        complex_spec.data = dict(x=xs, y=ys)

button_step = Button(label="Simulation step by step", width=200)
button_step.on_click(run_simulation_step)


div_load_complex = Div(
    text="Load initial complex",
    #width=200#, height=30
)

div_load_size = Div(
    text="Load grain size",
    #width=200#, height=30
)

div_load_special = Div(
    text="Load special GB ids",
    #width=200#, height=30
)

div_message = Div(
    text="",
    #width=200#, height=30
)





def update_params(attrname, old, new):
    """
    """
    div_params.text = f"""You choose:
        <br>NUMBER_OF_STEPS: {spinner_steps.value}
        <br>NUMBER_OF_NEW_SEEDS: {spinner_new_seeds.value}
        <br>SPEC_PROB: {spinner_spec_prob.value}
        <br>MAX_VOLUME_FRACTION: {spinner_max_vol_frac.value}
        <br>CRITICAL_SIZE: {spinner_cr_size.value}
    """



spinner_steps = Spinner(
    title="Number of iterations",  # a string to display above the widget
    low=1,  # the lowest possible number to pick
    high=None,  # the highest possible number to pick
    step=1,  # the increments by which the number can be adjusted
    value=10,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_steps.on_change('value', update_params)

spinner_new_seeds = Spinner(
    title="Number of new seeds on each step",  # a string to display above the widget
    low=1,  # the lowest possible number to pick
    high=None,  # the highest possible number to pick
    step=1,  # the increments by which the number can be adjusted
    value=1,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_new_seeds.on_change('value', update_params)

spinner_spec_prob = Spinner(
    title="Probability of a special GB to produce a new seed",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=1,  # the highest possible number to pick
    step=0.05,  # the increments by which the number can be adjusted
    value=1,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_spec_prob.on_change('value', update_params)

spinner_max_vol_frac = Spinner(
    title="Maximum volume fraction",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=1,  # the highest possible number to pick
    step=0.05,  # the increments by which the number can be adjusted
    value=0.9,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_max_vol_frac.on_change('value', update_params)

spinner_cr_size = Spinner(
    title="Critical size",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=None,  # the highest possible number to pick
    step=0.001,  # the increments by which the number can be adjusted
    value=0.005,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_cr_size.on_change('value', update_params)

spinner_height = Spinner(
    title="Domain_height",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=None,  # the highest possible number to pick
    step=1,  # the increments by which the number can be adjusted
    value=1,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_height.on_change('value', update_params)

spinner_width = Spinner(
    title="Domain_width",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=None,  # the highest possible number to pick
    step=1,  # the increments by which the number can be adjusted
    value=1,  # the initial value to display in the widget
    #width=100,  #  the width of the widget in pixels
    )
spinner_width.on_change('value', update_params)

div_params = Div(
    text=f"""You choose:
        <br>NUMBER_OF_STEPS: {spinner_steps.value}
        <br>NUMBER_OF_NEW_SEEDS: {spinner_new_seeds.value}
        <br>SPEC_PROB: {spinner_spec_prob.value}
        <br>MAX_VOLUME_FRACTION: {spinner_max_vol_frac.value}
        <br>CRITICAL_SIZE: {spinner_cr_size.value}
    """
    #width=200#, height=30
)

div_progress = Div(
    text=''
    #width=200#, height=30
)



complex0_ext = ColumnDataSource(data=dict(x=[], y=[]))
complex0_int = ColumnDataSource(data=dict(x=[], y=[]))
complex0_spec = ColumnDataSource(data=dict(x=[], y=[]))

plot_init = figure(
    title="Initial Complex", x_axis_label='x', y_axis_label='y',
    # x_range=(-0.1, 1.1), y_range=(-0.1, 1.1),
    width=500, height=500
)
plot_init.multi_line('x', 'y', source=complex0_ext, color='black')
plot_init.multi_line('x', 'y', source=complex0_int, color='blue')
plot_init.multi_line('x', 'y', source=complex0_spec, color='red')


complex_ext = ColumnDataSource(data=dict(x=[], y=[]))
complex_int = ColumnDataSource(data=dict(x=[], y=[]))
complex_spec = ColumnDataSource(data=dict(x=[], y=[]))
complex_new_seeds = ColumnDataSource(data=dict(x=[], y=[]))

plot_simul = figure(
    title="Final Complex", x_axis_label='x', y_axis_label='y',
    # x_range=(-0.1, 1.1), y_range=(-0.1, 1.1),
    width=500, height=500
)
plot_simul.multi_line('x', 'y', source=complex_ext, color='black')
plot_simul.multi_line('x', 'y', source=complex_int, color='blue')
plot_simul.multi_line('x', 'y', source=complex_spec, color='red')
plot_simul.circle('x', 'y', source=complex_new_seeds, size=20, color="navy", alpha=0.5)

p_vs_w = ColumnDataSource(data=dict(x=[], y=[]))

plot_pw = figure(
    title="p vs w", x_axis_label='x', y_axis_label='y',
    x_range=(-1, 1), 
    y_range=(0, 1),
    width=500, height=500
)
plot_pw.scatter('x', 'y', source=p_vs_w)
# layout = layout(
#     [
#         [input_wdir],
#         [div_load],
#         [input_complex],
#         [div_message]
#     ]
# )

col1 = column(
    [
        input_wdir,
        div_load_complex,
        input_complex,
        div_load_size,
        input_size,
        div_load_special,
        input_special,
        div_message
    ]
)
col2 = column(
    [
        [spinner_height, spinner_width],
        spinner_steps,
        spinner_new_seeds,
        spinner_spec_prob,
        spinner_max_vol_frac,
        spinner_cr_size
    ]
)
col3 = column(
    [
        div_params,
        button_start,
        div_progress,
        button_step
    ]
)
layout = layout(
    [
        [col1, col2, col3],
        [plot_init, plot_simul]
    ]
)

curdoc().add_root(layout)

# show(layout)