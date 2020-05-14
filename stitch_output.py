import maxlikespy.util as util
import sys


cell_range = sys.argv[-2:]
path = sys.argv[1]
cell_range = list(map(int, cell_range))
cell_range = range(cell_range[0], cell_range[1]+1)

util.collect_data(cell_range, "log_likelihoods", path)
util.collect_data(cell_range, "model_comparisons", path)
util.collect_data(cell_range, "cell_fits", path)