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


# Load results (FileInput)
def load_results(attrname, old, new):
    """
    """
    # decode file content
    file = io.StringIO(b64decode(new).decode(encoding='utf-8'))
    # create a DataFrame
    results.data = pd.read_csv(file).to_dict('list')

input_complex = FileInput(multiple=False)
input_complex.on_change('value', load_results)

div_load_results = Div(
    text="Load results",
    #width=200#, height=30
)



results = ColumnDataSource(data=dict(
    m=[], frac=[], vol_frac=[], omega=[],
    j0=[], j1=[], j2=[], j3=[], S=[], HAGBs_frac=[]
))

# Plots

plot_w_frac = figure(
    title="omega vs new grains fraction", 
    x_axis_label='new grains fraction', 
    y_axis_label='omega', 
    x_range=(0, 1),
    y_range=(-1, 1),
    width=500, height=500
)
plot_w_frac.line('frac', 'omega', source=results, line_width=2)


plot_w_vol_frac = figure(
    title="omega vs new grains volume fraction", 
    x_axis_label='new grains volume fraction', 
    y_axis_label='omega', 
    x_range=(0, 1),
    y_range=(-1, 1),
    width=500, height=500
)
plot_w_vol_frac.line('vol_frac', 'omega', source=results, line_width=2)


plot_S_frac = figure(
    title="S vs new grains fraction", 
    x_axis_label='new grains fraction', 
    y_axis_label='S', 
    x_range=(0, 1),
    # y_range=(-1, 1),
    width=500, height=500
)
plot_S_frac.line('frac', 'S', source=results, line_width=2)


plot_S_vol_frac = figure(
    title="S vs new grains volume fraction", 
    x_axis_label='new grains volume fraction', 
    y_axis_label='S', 
    x_range=(0, 1),
    # y_range=(-1, 1),
    width=500, height=500
)
plot_S_vol_frac.line('vol_frac', 'S', source=results, line_width=2)


plot_p_frac = figure(
    title="HAGBs fraction vs new grains fraction", 
    x_axis_label='new grains fraction', 
    y_axis_label='p', 
    x_range=(0, 1),
    y_range=(0, 1),
    width=500, height=500
)
plot_p_frac.line('frac', 'HAGBs_frac', source=results, line_width=2)


plot_p_vol_frac = figure(
    title="HAGBs fraction vs new grains volume fraction", 
    x_axis_label='new grains volume fraction', 
    y_axis_label='p', 
    x_range=(0, 1),
    y_range=(0, 1),
    width=500, height=500
)
plot_p_vol_frac.line('vol_frac', 'HAGBs_frac', source=results, line_width=2)


plot_j_frac = figure(
    title="TJ fractions vs new grains fraction", 
    x_axis_label='new grains fraction', 
    y_axis_label='j', 
    x_range=(0, 1),
    y_range=(0, 1),
    width=500, height=500
)
plot_j_frac.line('frac', 'j0', source=results, legend_label='j0', color='blue', line_width=2)
plot_j_frac.line('frac', 'j1', source=results, legend_label='j1', color='red', line_width=2)
plot_j_frac.line('frac', 'j2', source=results, legend_label='j2', color='green', line_width=2)
plot_j_frac.line('frac', 'j3', source=results, legend_label='j3', color='cyan', line_width=2)


plot_j_vol_frac = figure(
    title="TJ fractions vs new grains volume fraction", 
    x_axis_label='new grains volume fraction', 
    y_axis_label='j', 
    x_range=(0, 1),
    y_range=(0, 1),
    width=500, height=500
)
plot_j_vol_frac.line('vol_frac', 'j0', source=results, legend_label='j0', color='blue', line_width=2)
plot_j_vol_frac.line('vol_frac', 'j1', source=results, legend_label='j1', color='red', line_width=2)
plot_j_vol_frac.line('vol_frac', 'j2', source=results, legend_label='j2', color='green', line_width=2)
plot_j_vol_frac.line('vol_frac', 'j3', source=results, legend_label='j3', color='cyan', line_width=2)

layout = layout(
    [
        [div_load_results, input_complex],
        [plot_w_frac, plot_w_vol_frac],
        [plot_S_frac, plot_S_vol_frac],
        [plot_p_frac, plot_p_vol_frac],
        [plot_j_frac, plot_j_vol_frac] 
    ]
)

curdoc().add_root(layout)
