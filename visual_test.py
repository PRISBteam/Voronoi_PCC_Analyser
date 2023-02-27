'''
Bokeh https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_9.html
'''

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, Spinner, FileInput, Button
)

button = Button(label="Press me!", width=200)

div_load = Div(
    text="",
    #width=200#, height=30
)



def clear_data(event):
    if div_load.text == '' or div_load.text == 'Hello, world, once again!':
        div_load.text = 'Hello, world!'
    else:
        div_load.text = 'Hello, world, once again!'


button.on_click(clear_data)
# Layout


layout = layout(
    [
        [button],
        [div_load]
    ]
)


curdoc().add_root(layout)

# show(layout)