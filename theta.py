''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot
plot = figure(height=400, width=400, title="my sine wave",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


# Set up widgets
text = TextInput(title="title", value='my sine wave')
lower_threshold = Slider(title="offset", value=15.0, start=0.0, end=180.0, step=5)
upper_threshold = Slider(title="amplitude", value=60.0, start=0.0, end=180.0, step=5)
#phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
#freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = f"{lower_threshold.value} - {upper_threshold.value}"

def update_data(attrname, old, new):
    plot.title.text = f"{lower_threshold.value} - {upper_threshold.value}"
    # Get the current slider values
    a = lower_threshold.value
    b = upper_threshold.value
    #w = phase.value
    #k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4*np.pi, N)
    y = a*np.sin(x) + b

    source.data = dict(x=x, y=y)

# for w in [offset, amplitude, phase, freq]:
#     w.on_change('value', update_data)
lower_threshold.on_change('value', update_data)
upper_threshold.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(text, lower_threshold, upper_threshold)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Theta"