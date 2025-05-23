# gammon/cli/post.py

from pathlib import Path

from . import PostProcessor


def post_process(outfile, mu):
    print("Performing post-processing...")
    path = Path(outfile).absolute()
    pp = PostProcessor(path, mu)

    pp.do_graphE()
    pp.do_graph_nH()
    pp.acceptation_rate()
    pp.do_mean()
    pp.do_abs_graph()
