import maxlikespy.util as util
import sys


cell_range = sys.argv[-2:]
path = sys.argv[1]
cell_range = list(map(int, cell_range))
cell_range = range(cell_range[0], cell_range[1]+1)

util.collect_data(cell_range, "log_likelihoods_even", path)
util.collect_data(cell_range, "model_comparisons_even", path)
util.collect_data(cell_range, "cell_fits_even", path)

util.collect_data(cell_range, "log_likelihoods_odd", path)
util.collect_data(cell_range, "model_comparisons_odd", path)
util.collect_data(cell_range, "cell_fits_odd", path)