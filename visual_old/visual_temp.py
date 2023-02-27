"""
Try of new capabilities
"""


from base64 import b64decode
import io
import pandas as pd
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, FileInput, Button, TextInput
)
from bokeh.models.widgets import Select
import logging
from matgen.core import ReducedCellComplex



input_complex = FileInput(multiple=False)

div_complex = Div(
    text="",
    #width=100#, height=30
)
div_characteristics = Div(
    text="",
    #width=100#, height=30
)

select = Select(title="Complex:", value="",
                options=[])

text = TextInput(title='Enter filename', value='characteristics.txt')
button = Button(label="Save all to file")

div_saved = Div(
    text="",
    #width=100#, height=30
)

def update_complex(attrname, old, new):
    """
    """
    file = io.StringIO(b64decode(new).decode(encoding='utf-8'))
    is_tess = 'tess' in file.read()
    global reduced_complexes
    reduced_complexes = {}
    complex_id = 0
    if is_tess:
        pass
    else:
        try:
            file.seek(0)
            for line in file:
                complex_id += 1
                row = [*map(float, line.split())]
                if len(row) != 5:
                    raise ValueError()
                p = row[0]
                j_tuple = tuple(row[1:])
                reduced_complexes[complex_id] = ReducedCellComplex(p, j_tuple)
        except:
            div_complex.text = 'Wrong file!'
            logging.exception('Some error')
            return

        div_complex.text = f"""File content:
            <br># rows: {len(reduced_complexes.keys())}
            <br>ids: {list(reduced_complexes.keys())}
            <br>p_seq: {[rc.p for rc in reduced_complexes.values()]}
        """

        select.options = [
            f'{k} : p = {rc.p}' for k, rc in reduced_complexes.items()
        ]
        
        # p = reduced_complexes.keys()[0]
        # j_tuple = reduced_complexes[p]
        # rc = ReducedCellComplex(p, j_tuple)

        # div_characteristics.text = f"""p = {rc.p}
        #     <br> q = {rc.p}
        #     <br> S_p = {rc.p_entropy}
        # self.p_entropy_m = matutils.entropy_m(p)
        # self.p_entropy_s = matutils.entropy_s(p)
        # self.j0, self.j1, self.j2, self.j3 = j_tuple
        # self.p_expected = (self.j1 + 2*self.j2 + 3*self.j3) / 3
        # self.delta_p = abs(self.p_expected - self.p)
        # self.S = matutils.entropy(*j_tuple)
        # self.S_m = matutils.entropy_m(*j_tuple)
        # self.S_s = matutils.entropy_s(*j_tuple)
        # self.kappa = self.S_m / self.S_s if self.S_s != 0 else 0
        # self.delta_S = self.p_entropy - self.S"""

def update_characteristics(attrname, old, new):
    """
    """
    complex_id = int(new.split()[0])
    rc = reduced_complexes[complex_id]
    
    div_characteristics.text = f"""Choosed complex with id = {complex_id}
        <br> p = {rc.p}
        <br> Sp = {rc.p_entropy}, Sp_m = {rc.p_entropy_m}, Sp_s = {rc.p_entropy_s}
        <br> S = {rc.S}, S_m = {rc.S_m}, S_s = {rc.S_s}
    """

def update_out(attrname, old, new):
    """
    """
    pass

def save_to_file(event):
    """
    """
    ids = [*range(1, len(reduced_complexes.keys()) + 1)]
    df = pd.DataFrame(
        {
            'p' : [reduced_complexes[i].p for i in ids],
            'q' : [reduced_complexes[i].q for i in ids],
            'Sp' : [reduced_complexes[i].p_entropy for i in ids],
            'Sp_m' : [reduced_complexes[i].p_entropy_m for i in ids],
            'Sp_s' : [reduced_complexes[i].p_entropy_s for i in ids],
            'S' : [reduced_complexes[i].S for i in ids],
            'S_m' : [reduced_complexes[i].S_m for i in ids],
            'S_s' : [reduced_complexes[i].S_s for i in ids]
        }
    )
    
    try:
        df.to_csv(text.value, index=False, sep=' ')
        div_saved.text = 'Saved!'
    except:
        div_saved.text = '<p style="color:red">Not saved!!</p>'

input_complex.on_change('value', update_complex)

select.on_change('value', update_characteristics)
text.on_change('value', update_out)
button.on_click(save_to_file)

layout = layout(
    [
        [input_complex, select],
        [div_complex, div_characteristics],
        [text],
        [button],
        [div_saved]
    ]
)


curdoc().add_root(layout)