'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, Spinner, FileInput, Button
)
import numpy as np
from scipy.spatial import Voronoi
from matgen import core

# Widgets

div_header = Div(
    text="A header with the description of everything here is needed",
    #width=1000#, height=30
)

div_panel = Div(
    text="Command panel",
    #width=200#, height=30
)

div_p = Div(
    text="p =",
    #width=100#, height=30
)

spinner = Spinner(
    title="",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=1,  # the highest possible number to pick
    step=0.05,  # the increments by which the number can be adjusted
    value=0,  # the initial value to display in the widget
    width=100,  #  the width of the widget in pixels
    )

button = Button(label="Clear", width=200)

div_load = Div(
    text="Load",
    #width=200#, height=30
)

input_complex = FileInput()

input_p_seq = FileInput()

input_c_seq = FileInput()

div_data = Div(
    text="Related data",
    #width=200#, height=30
)

div_complex = Div(
    text="Complex",
    #width=100#, height=30
)

div_p_seq = Div(
    text="p_seq",
    #width=100#, height=30
)

div_c_seq = Div(
    text="c_seq",
    width=100#, height=30
)

div_load_complex = Div(
    text="Complex",
    #width=100#, height=30
)

div_load_p_seq = Div(
    text="p_seq",
    #width=100#, height=30
)

div_load_c_seq = Div(
    text="c_seq",
    width=100#, height=30
)

div_errors = Div(
    text=""
    #width=100#, height=30
)

# range_slider = RangeSlider(
#     title="Choose theta", # a title to display above the slider
#     start=0,  # set the minimum value for the slider
#     end=62,  # set the maximum value for the slider
#     step=5,  # increments for the slider
#     value=(0, 0),  # initial values for slider
#     )

# Figures

#s0 = ColumnDataSource(data=dict(x=[], y=[]))
s_TJ1 = ColumnDataSource(data=dict(x=[], y=[]))
s_TJ2 = ColumnDataSource(data=dict(x=[], y=[]))
s_TJ3 = ColumnDataSource(data=dict(x=[], y=[]))

s_sGB = ColumnDataSource(data=dict(x=[], y=[]))
s_crGB = ColumnDataSource(data=dict(x=[], y=[]))
#s6 = ColumnDataSource(data=dict(x=[], y=[]))

p_TJ1 = figure(
    title="J1", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
# p0.scatter('x', 'y', source=s0)
p_TJ1.multi_line('x', 'y', source=s_TJ1, color='black')

p_TJ2 = figure(
    title="J2", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
# p1.scatter('x', 'y', source=s1)
p_TJ2.multi_line('x', 'y', source=s_TJ2, color='black')

p_TJ3 = figure(
    title="J2", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
# p2.scatter('x', 'y', source=s2)
p_TJ3.multi_line('x', 'y', source=s_TJ3, color='black')

# p3 = figure(
#     title="J3", x_axis_label='x', y_axis_label='y',
#     x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
#     )
# # p3.scatter('x', 'y', source=s3)
# p3.multi_line('x', 'y', source=s3, color='black')

p_GB = figure(
    title="GBs", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
    )
# p3.scatter('x', 'y', source=s3)
p_GB.multi_line('x', 'y', source=s_sGB, color='blue')
p_GB.multi_line('x', 'y', source=s_crGB, color='red')
#p4.multi_line('x', 'y', source=s6, color='green')

# filename = 'tests/test_data/pass1_model_2d.txt'
# #c = core.CellComplex(filename=filename, measures=True, theta=True)
# c = core.CellComplex(filename=filename)

# with open('tests/test_data/pass_1_misorientation.txt', 'r') as file:
#     for line, edge in zip(file, c.edges):
#         edge.theta = float(line.strip())

# def get_xy(v_ids):
#     vs = c.get_many('v', v_ids)
#     xs = [v.x for v in vs]
#     ys = [v.y for v in vs]
#     return xs, ys

def get_xy_for_type(type):
    v_ids = c.get_junction_ids_of_type(type)
    points = np.array([v.coord2D for v in c.get_many('v', v_ids)])
    xs = []
    ys = []
    if len(points) >= 4:
        vor = Voronoi(points)
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                xs.append(list(vor.vertices[simplex, 0]))
                ys.append(list((vor.vertices[simplex, 1])))
    return xs, ys

def get_xy_for_edges(e_ids):
    """
    """
    xs = []
    ys = []

    for e in c.get_many('e', e_ids):
        xs.append([v.x for v in c.get_many('v', e.v_ids)])
        ys.append([v.y for v in c.get_many('v', e.v_ids)])

    return xs, ys

# Functions

def update_complex(attrname, new, old):
    """
    """
    global c
    try:
        c = core.CellComplex(filename=input_complex.filename)
    except:
        div_errors.text = '<p style="color:red">Wrong dir or file!!</p>'
        return
    global Ne_int
    Ne_int = len(c.get_internal_ids('e'))
    div_complex.text = (
        f"""Complex:
        <br>{c.facenb} 2-cells
        <br>{c.edgenb} 1-cells
        <br>{c.vernb} 0-cells
        """
    )

def update_p_seq(attrname, new, old):
    global seq
    seq = []
    try:
        with open(input_p_seq.filename, 'r') as file:
            for line in file:
                seq.append(int(line.strip()))
        div_p_seq.text = (
            f"""p_seq:
            <br>{len(seq)} 1-cells
            """
        )
    except:
        div_errors.text = '<p style="color:red">Wrong p_seq file!!</p>'
        return
    try:
        for e_id in seq:
            c.get_one('e', e_id).set_special()
        pmax = round(len(seq) / Ne_int, 2)
        spinner.high = round(int(pmax / 0.05) * 0.05, 2)
    except ValueError:
        div_errors.text = '<p style="color:red">Outer cannot be special</p>'
    except:
        div_errors.text = '<p style="color:red">Load complex!</p>'

    c.set_junction_types()
    x, y = get_xy_for_type(1)
    s_TJ1.data = dict(x=x, y=y)
    x, y = get_xy_for_type(2)
    s_TJ2.data = dict(x=x, y=y)
    x, y = get_xy_for_type(3)
    s_TJ3.data = dict(x=x, y=y)

def update_c_seq(attrname, new, old):
    global cracks
    cracks = []
    try:
        with open(input_c_seq.filename, 'r') as file:
            for line in file:
                cracks.append(int(line.strip()))
        div_c_seq.text = (
            f"""c_seq:
            <br>{len(cracks)} 1-cells
            """
        )
    except:
        div_errors.text = '<p style="color:red">Wrong c_seq file!!</p>'
        return
    try:
        xs, ys = get_xy_for_edges(cracks)
        s_crGB.data = dict(x=xs, y=ys)
    except:
        div_errors.text = '<p style="color:red">Load complex!!</p>'

def clear_data(event):
    spinner.value = 0
    s_TJ1.data = dict(x=[], y=[])
    s_TJ2.data = dict(x=[], y=[])
    s_TJ3.data = dict(x=[], y=[])
    s_sGB.data = dict(x=[], y=[])
    s_crGB.data = dict(x=[], y=[])

def update_data(attrname, new, old):

    div.text = f"""
          <p>Select {round(range_slider.value[0])}""" +\
        f""" - {round(range_slider.value[1])}:</p>
          """
    
    lt = round(range_slider.value[0])
    ut = round(range_slider.value[1])
    c.reset_special(lt, ut)



    n = spinner.value
    x, y, x_spec, y_spec, x_ext, y_ext = get_xy_for_edges(n)
    s4.data = dict(x=x, y=y)
    s5.data = dict(x=x_spec, y=y_spec)    
    s6.data = dict(x=x_ext, y=y_ext)


def update_GBs(attrname, new, old):
    """
    """
    p = spinner.value
    try:
        n = round(Ne_int * p)
        #div_errors.text = f'N = {n}'
        p_GB.title.text = f"GBs with p = {p}, n = {n}"
    except:
        div_errors.text = '<p style="color:red">Load complex!!</p>'
        return
    try:
        xs, ys = get_xy_for_edges(seq[:n])
        s_sGB.data = dict(x=xs, y=ys)
    except:
        div_errors.text = '<p style="color:red">Load p_seq!!</p>'


# Actions

# range_slider.on_change('value', update_data)

input_complex.on_change('filename', update_complex)
input_p_seq.on_change('filename', update_p_seq)
input_c_seq.on_change('filename', update_c_seq)

spinner.on_change('value', update_GBs)

button.on_click(clear_data)
# Layout


data_divs = row(div_complex, column(div_p_seq, div_c_seq))

grid_inputs = gridplot([
    [div_load_complex, input_complex],
    [div_load_p_seq, input_p_seq],
    [div_load_c_seq, input_c_seq]])

grid_panel = gridplot(
    [
        [div_panel],
        [row(div_p, spinner)],
        [button],
        [div_load],
        # [row(div_load_complex, input_complex)],
        # [row(div_load_p_seq, input_p_seq)],
        # [row(div_load_c_seq, input_c_seq)],
        [grid_inputs],
        [div_data],
        [data_divs],
        [div_errors]
    ])

grid_figures = gridplot([[p_TJ1, p_GB], [p_TJ2, p_TJ3, None]], width=400, height=400)

# layout = layout(
#     [
#         [div_header],
#         [grid_panel, grid_figures],
#     ]
# )

layout = row(grid_panel, column(div_header, grid_figures))

curdoc().add_root(layout)

# show(layout)