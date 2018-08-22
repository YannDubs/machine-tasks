import sys

sys.path.append("../tasks")

from reporter import generate_multireport
from tasks import get_task

IS_RETRAIN = True
OUTPUT_DIR = "../reports/"

all_tasks = [get_task("long lookup"),
             get_task("long lookup reverse"),
             get_task("noisy long lookup single"),
             get_task("long lookup intermediate noise"),
             get_task("noisy long lookup multi"),
             get_task("scan"),
             get_task("symbol rewriting", is_small=True)]

# EXAMPLE USAGE
print("0")
models, others0 = generate_multireport(all_tasks,
                                       OUTPUT_DIR,
                                       name="test_allData",
                                       eval_batch_size=256,
                                       is_retrain=IS_RETRAIN,
                                       is_value=True,
                                       is_position_attn=True,
                                       is_posrnn=True,
                                       embedding_size=32,
                                       is_clamp_mu=True,
                                       is_highway=True,
                                       is_plot_train=True,
                                       is_add_all_controller=True,
                                       use_attention="pre-rnn",
                                       anneal_min_sigma=0.1,
                                       anneal_mid_dropout=0.1,

                                       anneal_kq_noise_output=0.1,
                                       is_content_attn=True,
                                       content_method='dot',
                                       key_size=16)
