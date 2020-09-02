# NOTE: tf_upgrade_v2 does not successfully convert all these files to tf2
tf_upgrade_v2 --infile rnn_cell.py --outfile rnn_cell.py
tf_upgrade_v2 --infile rnn.py --outfile rnn.py
tf_upgrade_v2 --infile rnn_ops.py --outfile rnn_ops.py
tf_upgrade_v2 --infile tf_utils.py --outfile tf_utils.py
