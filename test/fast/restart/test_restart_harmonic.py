import sys
import os
import shutil
import copy
from pathlib import Path

from gammon import GCMC, Structure
from gammon.calculator import HarmonicCalc
from gammon.utilities import get_nd3_data
from ...conftest import find_data_path
from ... import context  # noqa


def test_20_restart_harmonic(root):
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"

    # Prepare the parameters
    rcut = 0.2
    rcut_swit = 1
    nstate = 4
    nstep = 100
    nrestart = 4
    prob = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
    e, x, f = get_nd3_data()
    irr_sites = [x[5], x[11]]
    harmonic_coef = [[e[5], f[5]], [e[11], f[11]]]
    chem_pot = -15
    T = 300

    # Prepare the object
    calc = HarmonicCalc(coef_irrsites=harmonic_coef,
                        rcut_swit=rcut_swit)
    struct = Structure(struct_fn,
                       irr_sites,
                       site_rcut=rcut)

    # Run the full simulation
    if Path("gcmc1").exists():
        shutil.rmtree("gcmc1")
    Path("gcmc1").mkdir()
    os.chdir("gcmc1")

    gcmc1 = GCMC(struct=copy.deepcopy(struct), nstate=nstate, calc=calc,
                 prob=prob, mu=chem_pot, T=T, buffer_size=1)
    gcmc1.run(nstep=nstep, extend=False)
    with open(gcmc1.result_fn / "GCMC.out", encoding="utf-8") as f:
        lines1 = f.readlines()

    os.chdir("..")

    # Run the simulation restarts
    if Path("gcmc2").exists():
        shutil.rmtree("gcmc2")
    Path("gcmc2").mkdir()
    os.chdir("gcmc2")
    gcmc2 = GCMC(struct=copy.deepcopy(struct), nstate=nstate, calc=calc,
                 prob=prob, mu=chem_pot, T=T, buffer_size=1)

    gcmc2.run(nstep=nstep//nrestart, extend=False)
    for i in range(nrestart-1):
        gcmc2 = GCMC(struct=copy.deepcopy(struct), nstate=nstate, calc=calc,
                     prob=prob, mu=chem_pot, T=T, buffer_size=1)
        gcmc2.run(nstep=nstep//nrestart, extend=True)

    with open(gcmc2.result_fn / "GCMC.out", encoding="utf-8") as f:
        lines2 = f.readlines()
    os.chdir("..")

    # Parse the data into a table
    def read_outlines(lines, table):
        for line in lines:
            if line.startswith("-"):
                continue
            nstep, nproc = int(line.split()[0])-1, int(line.split()[1])
            res = f"{line.split()[2]} {line.split()[3]} {line.split()[4]}"
            E = float(line.split()[5])
            table[nstep][nproc][0] = res
            table[nstep][nproc][1] = E
        return table

    table1 = [[[None for _ in range(2)] for _ in range(nstate)]
              for _ in range(nstep)]
    table2 = [[[None for _ in range(2)] for _ in range(nstate)]
              for _ in range(nstep)]
    table1 = read_outlines(lines1, table1)
    table2 = read_outlines(lines2, table2)

    # Compare the table
    for nstep in range(len(table1)):
        for nproc in range(len(table1[nstep])):
            l1 = table1[nstep][nproc]
            l2 = table2[nstep][nproc]
            if (l1[0] != l2[0]) or (float(l1[1]) - float(l2[1]) > 1e-6):
                e = f"Step{nstep}, State{nproc}, something went wrong :"
                e += f"{table1[nstep][nproc]} {table2[nstep][nproc]}"
                print(e, file=sys.stderr)
                print(table1[nstep][nproc][0],
                      table2[nstep][nproc][0], file=sys.stderr)
                print(table1[nstep][nproc][1],
                      table2[nstep][nproc][1], file=sys.stderr)
                raise RuntimeError(e)

    # Cleanup
    shutil.rmtree("gcmc1")
    shutil.rmtree("gcmc2")
