import pytest
import sparsemat

def test_extract_seeds():
    """
    """
    filename = 'tests/test_data/n8-id1-2D.tess'
    seeds2D = sparsemat.extract_seeds(filename)

    assert seeds2D[2] == pytest.approx((0.857150589922, 0.716451625749, 0.000000000000))

    filename = 'tests/test_data/n8-id1-3D.tess'
    seeds2D = sparsemat.extract_seeds(filename)

    assert seeds2D[3] == pytest.approx((0.531862177357, 0.818357065409, 0.577596831426))