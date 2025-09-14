"""Test code for TIL Python programming jupyter notebook Lab 5 - Algorithms"""
import pytest
from testbook import testbook

@pytest.fixture(scope='module')
def tb():
    # note: with timeout as the filtering takes some time and Python is slow
    with testbook('lab5_2025_answers.ipynb', execute=True, timeout=600) as tb:
        yield tb

# loading data
def test_data(tb):
    data = tb.ref('data')
    assert len(data) == 73908, "The number of rows is not correct."
    assert len(data.columns) == 3, "The number of columns is not correct."
    assert 'DV' in data.columns, "Column DV is not in the data."
    assert 'S' in data.columns, "Column S is not in the data."
    assert 'A' in data.columns, "Column A is not in the data."

# initializing the grid
def test_grid_init(tb):
    # dv
    dv = tb.ref('dv')
    assert len(dv) == 41, "dv should have a length of 41"
    assert abs(dv[0] - -10.0) < 1e-3, "dv should range from -10.0"
    assert abs(dv[40] - 10) < 1e-3, "dv should range to 10.0"
    # s
    s = tb.ref('s')
    assert len(s) == 21, "s should have a length of 21"
    assert abs(s[0] - 0.0) < 1e-3, "s should range from 0.0"
    assert abs(s[20] - 200.0) < 1e-3, "s should range to 200.0"
    # a
    a = tb.ref('a')
    assert a.shape[0] == 21, "a should have 21 rows"
    assert a.shape[1] == 41, "a should have 41 columns"

# as numpy
def test_as_numpy(tb):
    assert tb.value('str(type(DV))') == "<class 'numpy.ndarray'>", "DV is not a numpy.ndarray"
    assert tb.value('str(type(S))') == "<class 'numpy.ndarray'>", "S is not a numpy.ndarray"
    assert tb.value('str(type(A))') == "<class 'numpy.ndarray'>", "A is not a numpy.ndarray"
    assert tb.value('len(DV)') == 73908, "DV does not have the right length"
    assert tb.value('len(S)') == 73908, "S does not have the right length"
    assert tb.value('len(A)') == 73908, "A does not have the right length"

# result
def test_result(tb):
    """
    (code for in notebook to generate correct row/col statistics)
    row_sums = []
    row_stds = []
    for i in range(0, len(s)):
        row_sums.append(float(a[i, :].mean()))
        row_stds.append(float(a[i, :].std()))
    col_sums = []
    col_stds = []
    for j in range(0, len(dv)):
        col_sums.append(float(a[:, j].mean()))
        col_stds.append(float(a[:, j].std()))
    print('row_sums = ' + str(row_sums))
    print('col_sums = ' + str(col_sums))
    print('row_stds = ' + str(row_stds))
    print('col_stds = ' + str(col_stds))
    """
    row_sums = [-0.13538234342571506, -0.13405508693900678, -0.11760009463404308, -0.08952099485693417, -0.06866924773918509, 
                -0.05676371620641918, -0.04552733459509672, -0.02891061411018983, -0.01915028660943688, -0.009664403457889634, 
                -0.004581013582619658, -0.004056246879517071, -0.009449195819905089, -0.019017894288315875, -0.03697350019288648, 
                -0.06841827759093853, -0.11313135022695019, -0.1599066955450514, -0.17800423089103035, -0.18637027627425734, 
                -0.18565508174793047]
    col_sums = [0.29308884080147335, 0.3201202487910533, 0.34731555541941245, 0.36863980444859695, 0.3890328529566817, 
                0.4041973239035973, 0.40303775321818314, 0.3916678699231924, 0.3835233357060087, 0.3723080980585993, 
                0.3572872581890928, 0.33811032764100385, 0.3136088197192015, 0.2928940681918704, 0.26673244960812154, 
                0.2353074980169952, 0.19992067343051304, 0.15880430151099598, 0.10971764568250511, 0.05632249825820437, 
                -0.0004074601824117023, -0.06413467905262543, -0.12839540612248465, -0.19030379239400008, -0.24707200966128542, 
                -0.30089930818176497, -0.3478263507581108, -0.39536568735703526, -0.4391756143595774, -0.46848901399841025, 
                -0.4913428977173283, -0.5187249022321764, -0.5463355789725005, -0.5690579373548724, -0.593141520535162, 
                -0.6240961960519943, -0.6433187733058012, -0.6566478266659063, -0.6657193859875322, -0.6839118927032792, 
                -0.6893244808403804]
    row_stds = [0.49246307024383446, 0.49405656280171656, 0.507085685216817, 0.5245600552724622, 0.5350295319757017, 
                0.5365905670330141, 0.5294542901977626, 0.5130673419474796, 0.4828893926356937, 0.4477007684080995, 
                0.414881271223064, 0.3899308261911562, 0.37387463047156627, 0.3572497337695616, 0.3451941250734744, 
                0.3293750562678042, 0.3112405845502191, 0.2969425673643146, 0.2742529598521361, 0.25716046876386184, 
                0.24682608424400834]
    col_stds = [0.3292779134456069, 0.3037172778671089, 0.2752989283872806, 0.2456309628042609, 0.21041759247166178, 
                0.18130588701036307, 0.17001223349348243, 0.16648744903733437, 0.16020344516547988, 0.15496306111854943, 
                0.14573622095762365, 0.1332052525316638, 0.11851568875878994, 0.09915832280549554, 0.08090672051068432, 
                0.06714929272929468, 0.05449353251914492, 0.045268018382996045, 0.04196659461601065, 0.04365496359999668, 
                0.05074680353261589, 0.06058841548334946, 0.07143102621614877, 0.08104087347672886, 0.09023734648587958, 
                0.09728842413642955, 0.10480104680993095, 0.10953665127058994, 0.11233511107620539, 0.115440783714841, 
                0.11885181049580529, 0.11989455400196129, 0.11613416706975489, 0.110530474554263, 0.10488397283062961, 
                0.09183112135375353, 0.08419581245232298, 0.08183690016391228, 0.08322525216768564, 0.08377754572311669, 
                0.08869085905437202]
    for i in range(0, len(row_sums)):
        assert abs(tb.value('a[%d,:].mean()' % i) - row_sums[i]) < 1e-3, "Row %d does not have the right mean." % i
        assert abs(tb.value('a[%d,:].std()' % i) - row_stds[i]) < 1e-3, "Row %d does not have the right standard deviation." % i
    for j in range(0, len(col_sums)):
        assert abs(tb.value('a[:,%d].mean()' % j) - col_sums[j]) < 1e-3, "Column %d does not have the right mean." % j
        assert abs(tb.value('a[:,%d].std()' % j) - col_stds[j]) < 1e-3, "Column %d does not have the right standard deviation." % j
    