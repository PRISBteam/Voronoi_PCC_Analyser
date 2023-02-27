'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''

from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import gridplot, layout
from bokeh.models import Div, ColumnDataSource, Spinner, FileInput, Button
import numpy as np
from scipy.spatial import Voronoi
from matgen import core

def get_xy_for_edges(e_ids):
    """
    """
    xs = []
    ys = []

    for e in c.get_many('e', e_ids):
        xs.append([v.x for v in c.get_many('v', e.v_ids)])
        ys.append([v.y for v in c.get_many('v', e.v_ids)])

    return xs, ys


s_seq = ColumnDataSource(data=dict(x=[], y=[]))
s_cracks = ColumnDataSource(data=dict(x=[], y=[]))

plot = figure(
    title="Edges", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1),
    width=500, height=500
)
plot.multi_line('x', 'y', source=s_cracks, color='black')
plot.multi_line('x', 'y', source=s_seq, color='blue')

spinner = Spinner(
    title="p",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=1,  # the highest possible number to pick
    step=0.05,  # the increments by which the number can be adjusted
    value=0,  # the initial value to display in the widget
    width=200,  #  the width of the widget in pixels
    )


def update_c(attrname, new, old):
    global c
    try:
        c = core.CellComplex(filename=input_c.filename)
    except:
        div_errors.text = '<b>Wrong dir or file!!</b>'
        return
    global Ne_int
    Ne_int = len(c.get_internal_ids('e'))
    div_c.text = f'Loaded cell complex: {c}'


def update_seq(attrname, new, old):
    global seq
    seq = []
    try:
        with open(input_seq.filename, 'r') as file:
            for line in file:
                seq.append(int(line.strip()))
        div_seq.text = f'Loaded {len(seq)} e_ids'
    except:
        div_errors.text = '<b>Wrong file!!</b>'

def update_cracks(attrname, new, old):
    global cracks
    cracks = []
    try:
        with open(input_cracks.filename, 'r') as file:
            for line in file:
                cracks.append(int(line.strip()))
        div_cracks.text = f'Loaded {len(cracks)} e_ids'
    except:
        div_errors.text = '<b>Wrong file!!</b>'
        return
    try:
        xs, ys = get_xy_for_edges(cracks)
        s_cracks.data = dict(x=xs, y=ys)
    except:
        div_cracks.text = f'<b>Load Complex!!</b>'

def delete_cracks(event):
    s_cracks.data = dict(x=[], y=[])

def update_plot(attrname, new, old):
    p = spinner.value
    try:
        n = round(Ne_int * p)
        div_errors.text = f'N = {n}'
    except:
        div_errors.text = '<b>Load Complex!!</b>'
        return
    try:
        xs, ys = get_xy_for_edges(seq[:n])
        s_seq.data = dict(x=xs, y=ys)
    except:
        div_errors.text = '<b>Load edges!!</b>'

input_c = FileInput()
input_c.on_change('filename', update_c)

input_seq = FileInput()
input_seq.on_change('filename', update_seq)

input_cracks = FileInput()
input_cracks.on_change('filename', update_cracks)

spinner.on_change('value', update_plot)

div_c = Div(
    text='Load Cell Complex (.tess)',
    width=200,
    height=30
)

div_seq = Div(
    text='Load edges (e_ids)',
    width=200,
    height=30
)

div_cracks = Div(
    text='Load cracks (e_ids)',
    width=200,
    height=30
)

div_errors = Div(
    text='',
    width=200,
    height=30
)

button = Button(label="Delete cracks", width=200,)
button.on_click(delete_cracks)

grid1 = gridplot(
    [
        [input_c, input_seq, input_cracks],
        [div_c, div_seq, div_cracks]
    ]
)

grid2 = gridplot(
    [
        [spinner],
        [div_errors],
        [button]
    ]
)

layout = layout(
    [
        [grid1],
        [grid2, plot]
    ]
)

curdoc().add_root(layout)