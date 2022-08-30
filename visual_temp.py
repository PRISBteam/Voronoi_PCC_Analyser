"""
Try of new capabilities
"""


from base64 import b64decode
import io
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import gridplot, layout, column, row
from bokeh.models import (
    Div, RangeSlider, ColumnDataSource, FileInput, Button, TextInput
)
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

def update_complex(attrname, old, new):
    """
    """
    file = io.StringIO(b64decode(new).decode(encoding='utf-8'))
    is_tess = 'tess' in file.read()
    reduced_complexes = {}

    if is_tess:
        pass
    else:
        try:
            file.seek(0)
            for line in file:
                row = [*map(float, line.split())]
                if len(row) != 5:
                    raise ValueError()
                p = row[0]
                j_tuple = tuple(row[1:4])
                reduced_complexes[p] = j_tuple
        except:
            div_complex.text = 'Wrong file!'
            return

        div_complex.text = f"""File content:
            <br># rows: {len(reduced_complexes.keys())}
            <br>p seq: {reduced_complexes.keys()}
            <br>content: {reduced_complexes.values()}
        """

        
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

input_complex.on_change('value', update_complex)



curdoc().add_root(column(input_complex, div_complex, div_characteristics))