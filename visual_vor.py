'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''

from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import gridplot, layout
from bokeh.models import Div, RangeSlider, ColumnDataSource, Spinner
import numpy as np
from scipy.spatial import Voronoi
from matgen import core

filename = 'tests/test_data/pass1_model_2d.txt'
#c = core.CellComplex(filename=filename, measures=True, theta=True)
c = core.CellComplex(filename=filename)

with open('tests/test_data/pass_1_misorientation.txt', 'r') as file:
    for line, edge in zip(file, c.edges):
        edge.theta = float(line.strip())

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
    if len(points):
        vor = Voronoi(points)
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                xs.append(list(vor.vertices[simplex, 0]))
                ys.append(list((vor.vertices[simplex, 1])))
    return xs, ys

def get_xy_for_edges(n):
    xs = []
    ys = []
    xs_spec = []
    ys_spec = []
    xs_ext = []
    ys_ext = []

    for e in c.edges[:n]:
        if e.is_special:
            xs_spec.append([v.x for v in c.get_many('v', e.v_ids)])
            ys_spec.append([v.y for v in c.get_many('v', e.v_ids)])
        elif e.is_external:
            xs_ext.append([v.x for v in c.get_many('v', e.v_ids)])
            ys_ext.append([v.y for v in c.get_many('v', e.v_ids)])
        else:
            xs.append([v.x for v in c.get_many('v', e.v_ids)])
            ys.append([v.y for v in c.get_many('v', e.v_ids)])


    return xs, ys, xs_spec, ys_spec, xs_ext, ys_ext

s0 = ColumnDataSource(data=dict(x=[], y=[]))
s1 = ColumnDataSource(data=dict(x=[], y=[]))
s2 = ColumnDataSource(data=dict(x=[], y=[]))
s3 = ColumnDataSource(data=dict(x=[], y=[]))

s4 = ColumnDataSource(data=dict(x=[], y=[]))
s5 = ColumnDataSource(data=dict(x=[], y=[]))
s6 = ColumnDataSource(data=dict(x=[], y=[]))

p0 = figure(
    title="J0", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
# p0.scatter('x', 'y', source=s0)
p0.multi_line('x', 'y', source=s0, color='black')

p1 = figure(
    title="J1", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
# p1.scatter('x', 'y', source=s1)
p1.multi_line('x', 'y', source=s1, color='black')

p2 = figure(
    title="J2", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
# p2.scatter('x', 'y', source=s2)
p2.multi_line('x', 'y', source=s2, color='black')

p3 = figure(
    title="J3", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
    )
# p3.scatter('x', 'y', source=s3)
p3.multi_line('x', 'y', source=s3, color='black')

p4 = figure(
    title="Edges", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
    )
# p3.scatter('x', 'y', source=s3)
p4.multi_line('x', 'y', source=s4, color='blue')
p4.multi_line('x', 'y', source=s5, color='red')
p4.multi_line('x', 'y', source=s6, color='green')

range_slider = RangeSlider(
    title="Choose theta", # a title to display above the slider
    start=0,  # set the minimum value for the slider
    end=62,  # set the maximum value for the slider
    step=5,  # increments for the slider
    value=(0, 0),  # initial values for slider
    )

div = Div(
    text=f"""
          <p>Select {round(range_slider.value[0])}""" +\
        f""" - {round(range_slider.value[1])}:</p>
          """,
    width=200,
    height=30,
)

spinner = Spinner(
    title="Number of edges",  # a string to display above the widget
    low=0,  # the lowest possible number to pick
    high=c.edgenb,  # the highest possible number to pick
    step=10,  # the increments by which the number can be adjusted
    value=0,  # the initial value to display in the widget
    width=200,  #  the width of the widget in pixels
    )

def update_data(attrname, new, old):

    div.text = f"""
          <p>Select {round(range_slider.value[0])}""" +\
        f""" - {round(range_slider.value[1])}:</p>
          """
    
    lt = round(range_slider.value[0])
    ut = round(range_slider.value[1])
    c.reset_special(lt, ut)

    # x, y = get_xy(c.get_junction_ids_of_type(0))
    x, y = get_xy_for_type(0)
    s0.data = dict(x=x, y=y)
    # x, y = get_xy(c.get_junction_ids_of_type(1))
    x, y = get_xy_for_type(1)
    s1.data = dict(x=x, y=y)
    # x, y = get_xy(c.get_junction_ids_of_type(2))
    x, y = get_xy_for_type(2)
    s2.data = dict(x=x, y=y)
    # x, y = get_xy(c.get_junction_ids_of_type(3))
    x, y = get_xy_for_type(3)
    s3.data = dict(x=x, y=y)

    n = spinner.value
    x, y, x_spec, y_spec, x_ext, y_ext = get_xy_for_edges(n)
    s4.data = dict(x=x, y=y)
    s5.data = dict(x=x_spec, y=y_spec)    
    s6.data = dict(x=x_ext, y=y_ext)


def update_data_4(attrname, new, old):
    n = spinner.value
    x, y, x_spec, y_spec, x_ext, y_ext = get_xy_for_edges(n)
    s4.data = dict(x=x, y=y)
    s5.data = dict(x=x_spec, y=y_spec)    
    s6.data = dict(x=x_ext, y=y_ext)

range_slider.on_change('value', update_data)

spinner.on_change('value', update_data_4)

grid = gridplot([[p2, p3, p4], [p0, p1, None]], width=400, height=400)

layout = layout(
    [
        [range_slider, div, spinner],
        [grid],
    ]
)

curdoc().add_root(layout)

# show(layout)