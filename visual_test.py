'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''
import io
import os
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
    initial_complex = CellComplex.from_tess_file(file)
    # save initial seeds to a file
    seeds_pathname = os.path.join(wdir, 'seeds.txt')
    extract_seeds(initial_complex, seeds_pathname)
    div_message.text = f'Complex loaded! Seeds saved: {seeds_pathname}'

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
        div_message.text = f'Special GBs set!'
    except NameError:
        div_message.text = f'Load complex first!'
    except:
        div_message.text = f'Check file with special GB ids!'

input_special = FileInput(multiple=False)
input_special.on_change('value', load_special)


button_start = Button(label="Start simulation", width=200)

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
        button_start
    ]
)
layout = layout([[col1, col2, col3]])

curdoc().add_root(layout)

# show(layout)