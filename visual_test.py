'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''
import io
import os
import subprocess
from base64 import b64decode

import numpy as np

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, Spinner, FileInput, Button, TextInput
)

from matgen.base import CellComplex


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


input_wdir = TextInput(title='Save files to', value=os.getcwd())
input_wdir.on_change('value', check_wdir)


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
        pairs_with_special_GB = []
        NUMBER_OF_GRAINS = initial_complex.grainnb
        div_message.text = f'Complex loaded! Seeds saved: {seeds_pathname}'
    except:
        div_message.text = f'Complex not loaded! Check .tess file!'

    if initial_complex.dim == 2:
        ext_ids = initial_complex.get_external_ids('e')
        int_ids = initial_complex.get_internal_ids('e')
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
        initial_complex.reset_special(special_ids=special_ids)
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
    output_file = os.path.join(wdir, f'n{n}-id{neper_id}-{dim}D.tess')

    com_line_list = [
        'neper', '-T', '-n', str(n),
        '-id', str(neper_id), '-dim', str(dim),
        '-morphooptiini', f'coo:file({seeds_filename})',
        '-statcell', 'size',
        '-o', output_file.rstrip('.tess')
    ]
      
    run = subprocess.run(com_line_list, capture_output=True)
    # print(run.stdout)
    cell_complex = CellComplex.from_tess_file(
        output_file, with_cell_size=True)
    return cell_complex


def run_simulation(event):
    """
    """
    wdir = input_wdir.value
    seeds_filename = os.path.join(wdir, 'seeds.txt')
    cell_complex = initial_complex
    n0 = initial_complex.grainnb 
    n = n0
    k = spinner_new_seeds.value # TODO: k may be different for each step
    for step_idx in range(spinner_steps.value):
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
        xs, ys = new_seeds[:, 0].tolist(), new_seeds[:, 1].tolist()
        complex_new_seeds.data = dict(x=xs, y=ys)
        
        # Generate new complex from seeds.txt
        n += k
        div_progress.text = f'Progress {n}/{n0 + spinner_steps.value}'
        cell_complex = create_new_complex(
            n, neper_id=1, dim=initial_complex.dim
        )
        
        # Set special GBs from initial complex
        for cell in cell_complex._GBs.values():
            if set(cell.incident_ids) in pairs_with_special_GB:
                    if not cell.is_external:
                        cell.set_special(True)
        # Set special GBs for new grains
        for grain_id in range(n0 + 1, n + 1):
            gb_ids = cell_complex._grains[grain_id].gb_ids
            for gb_id in gb_ids:
                cell = cell_complex._GBs[gb_id]
                if not cell.is_external:
                    cell.set_special(True)
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
    
    # Set special GBs from initial complex
    for cell in cell_complex._GBs.values():
        if set(cell.incident_ids) in pairs_with_special_GB:
                if not cell.is_external:
                    cell.set_special(True)
    # Set special GBs for new grains
    for grain_id in range(initial_complex.grainnb + 1, NUMBER_OF_GRAINS + 1):
        gb_ids = cell_complex._grains[grain_id].gb_ids
        for gb_id in gb_ids:
            cell = cell_complex._GBs[gb_id]
            if not cell.is_external:
                cell.set_special(True)
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
    value=0,  # the initial value to display in the widget
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

div_params = Div(
    text=f"""You choose:
        <br>NUMBER_OF_STEPS: {spinner_steps.value}
        <br>NUMBER_OF_NEW_SEEDS: {spinner_new_seeds.value}
        <br>SPEC_PROB: {spinner_spec_prob.value}
        <br>MAX_VOLUME_FRACTION: {spinner_max_vol_frac.value}
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
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1),
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
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1),
    width=500, height=500
)
plot_simul.multi_line('x', 'y', source=complex_ext, color='black')
plot_simul.multi_line('x', 'y', source=complex_int, color='blue')
plot_simul.multi_line('x', 'y', source=complex_spec, color='red')
plot_simul.circle('x', 'y', source=complex_new_seeds, size=20, color="navy", alpha=0.5)

w_vs_N = ColumnDataSource(data=dict(x=[], y=[]))

plot_wN = figure(
    title="w vs N", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1),
    width=500, height=500
)

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
        spinner_steps,
        spinner_new_seeds,
        spinner_spec_prob,
        spinner_max_vol_frac
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
        [plot_init, plot_simul, plot_wN]
    ]
)

curdoc().add_root(layout)

# show(layout)