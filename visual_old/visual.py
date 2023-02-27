'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''
from base64 import b64decode
import io
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, FileInput, Button, TextInput
)
from matgen import core

input_complex = FileInput(accept='.tess,.txt')

input_theta = FileInput()

div_complex = Div(
    text="",
    #width=100#, height=30
)

div_theta = Div(
    text="",
    #width=100#, height=30
)

text = TextInput(title='Enter filename', value='spec_edges.txt')

button = Button(label="Save special GB")

div_saved = Div(
    text="",
    #width=100#, height=30
)


# filename = 'tests/test_data/pass1_model_2d.txt'
# #c = core.CellComplex(filename=filename, measures=True, theta=True)
# c = core.CellComplex(filename=filename)

# with open('tests/test_data/pass_1_misorientation.txt', 'r') as file:
#     for line, edge in zip(file, c.edges):
#         edge.theta = float(line.strip())

def get_xy(v_ids):
    vs = c.get_many('v', v_ids)
    xs = [v.x for v in vs]
    ys = [v.y for v in vs]
    return xs, ys

s0 = ColumnDataSource(data=dict(x=[], y=[]))
s1 = ColumnDataSource(data=dict(x=[], y=[]))
s2 = ColumnDataSource(data=dict(x=[], y=[]))
s3 = ColumnDataSource(data=dict(x=[], y=[]))

p0 = figure(
    title="J0", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
p0.scatter('x', 'y', source=s0, color='black', size=2)

p1 = figure(
    title="J1", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
p1.scatter('x', 'y', source=s1, color='black', size=2)

p2 = figure(
    title="J2", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
)
p2.scatter('x', 'y', source=s2, color='black', size=2)

p3 = figure(
    title="J3", x_axis_label='x', y_axis_label='y',
    x_range=(-0.1, 1.1), y_range=(-0.1, 1.1)
    )
p3.scatter('x', 'y', source=s3, color='black', size=2)

range_slider = RangeSlider(
    title="Choose theta", # a title to display above the slider
    start=0,  # set the minimum value for the slider
    end=62,  # set the maximum value for the slider
    step=1,  # increments for the slider
    value=(0, 62),  # initial values for slider
    )

div = Div(
    text=f"""
          <p>Select {round(range_slider.value[0])}""" +\
        f""" - {round(range_slider.value[1])}:</p>
          """,
    width=200,
    height=30,
)


def update_complex(attrname, old, new):
    """
    """
    # copies .tess file to working dir
    temp_fname = 'complex_copied.tess'
    with open(temp_fname, 'w', encoding='utf-8', newline='') as file:
        file.write(b64decode(new).decode(encoding='utf-8'))
    
    # read the file from working dir
    global c
    try:
        c = core.CellComplex(filename=temp_fname)
        div_complex.text = 'Loaded!'
        x = y = []
        s0.data = dict(x=x, y=y)
        s1.data = dict(x=x, y=y)
        s2.data = dict(x=x, y=y)
        s3.data = dict(x=x, y=y)
    except:
        div_complex.text = '<p style="color:red">Wrong dir or file!!</p>'


def update_theta(attrname, old, new):
    """
    """
    div_saved.text = ''
    try:
        file = io.StringIO(b64decode(new).decode())
        for line, edge in zip(file, c.edges):
            edge.theta = float(line.strip())
        #         edge.theta = float(line.strip())
        # with open(input_theta.filename, 'r') as file:
        #     for line, edge in zip(file, c.edges):
        #         edge.theta = float(line.strip())

        div_theta.text = 'Loaded!'
    except NameError:
        div_theta.text = '<p style="color:red">Load complex!!</p>'
    except:
        div_theta.text = '<p style="color:red">Wrong dir or file!!</p>'

def update_data(attrname, old, new):

    div_saved.text = ''
    div.text = f"""
          <p>Select {round(range_slider.value[0])}""" +\
        f""" - {round(range_slider.value[1])}:</p>
          """
    
    lt = round(range_slider.value[0])
    ut = round(range_slider.value[1])
    c.reset_special(lt, ut)

    x, y = get_xy(c.get_junction_ids_of_type(0))
    s0.data = dict(x=x, y=y)
    x, y = get_xy(c.get_junction_ids_of_type(1))
    s1.data = dict(x=x, y=y)
    x, y = get_xy(c.get_junction_ids_of_type(2))
    s2.data = dict(x=x, y=y)
    x, y = get_xy(c.get_junction_ids_of_type(3))
    s3.data = dict(x=x, y=y)

def update_out(attrname, old, new):
    """
    """
    pass

def save_edges(event):
    """
    """
    try:
        with open(text.value, 'w') as file:
            for e_id in c.get_special_ids():
                file.write(str(e_id) + '\n')
        div_saved.text = 'Saved!'
    except:
        div_saved.text = '<p style="color:red">Not saved!!</p>'


range_slider.on_change('value', update_data)

input_complex.on_change('value', update_complex)
input_theta.on_change('value', update_theta)

text.on_change('value', update_out)
button.on_click(save_edges)

grid = gridplot([[p1, p2, p3], [p0, None, None]], width=400, height=400)

layout = layout(
    [
        [
            column(input_complex, div_complex),
            column(input_theta, div_theta),
            column(text, button),
            div_saved
        ],
        [range_slider, div],
        [grid],
    ]
)

curdoc().add_root(layout)

# show(layout)