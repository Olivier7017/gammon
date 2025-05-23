import os
import sys
import shutil
import copy
from pathlib import Path
import pytest

from ase.data import atomic_numbers
from ase.io import read

from gammon import GCMC, Structure
from gammon.calculator import SnapCalc
from gammon.utilities import get_nd3_data
from ...conftest import find_data_path, has_lammps
from ... import context  # noqa


def generate_pair_coeff(atoms):
    symbols = sorted(set(atoms.get_chemical_symbols()))
    z_list = [atomic_numbers[x] for x in symbols]
    pc_list = []  # i.e. [[a1,a2],[b1,b2]]
    for t1 in range(len(z_list)):
        for t2 in range(t1+1):
            pc = f"pair_coeff {t2+1} {t1+1} zbl {z_list[t2]} {z_list[t1]}"
            pc_list.append(pc)
    if "H" not in symbols:
        symbols.append("H")
    pc_list.append("pair_coeff * * mliap "+" ".join(symbols))

    return pc_list


@pytest.mark.skipif(not has_lammps(), reason="No LAMMPS executable")
def test_restart_snap(root):
    datapath = find_data_path(root)
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    struct_fn = datapath / "Nd3MgNi14-2H.cif"
    model_fn = datapath / "Potential" / "modified_Nd3MgNi14H05_snap.model"
    descriptor_fn = model_fn.with_suffix(".descriptor")

    # Prepare the parameters
    nstate = 4
    nstep = 40
    nrestart = 2
    rcut = 0.25
    prob = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
    e, x, _ = get_nd3_data()
    irr_sites = [x[3], x[12]]

    T = 300

    ps = "pair_style hybrid/overlay zbl 0.4 0.9 mliap model linear " +\
         f"{model_fn} descriptor sna {descriptor_fn}"

    pc = generate_pair_coeff(read(struct_fn))

    # Prepare the object
    chem_pot = -124.549150/2
    calc = SnapCalc(model_fn=model_fn,
                    descriptor_fn=descriptor_fn,
                    pair_style=ps,
                    pair_coeff=pc,
                    lmp_cmd="lmp_serial")

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

    # Run the simulation with restarts
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
                raise AssertionError(e)
    # Cleanup
    shutil.rmtree("gcmc1")
    shutil.rmtree("gcmc2")
    shutil.rmtree("LAMMPS")
