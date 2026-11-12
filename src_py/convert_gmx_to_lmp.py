#!/usr/bin/env python3
"""Build a TIP3P + tert-butyl fluoride LAMMPS (real units) system.

Reads the exact TBF.itp, tip3p.itp and PDB templates in tbfwater_gmx.tar.gz.
Uses numpy for random rigid-molecule placement, no external packing program.

TBF atomtypes, including HTB, are read from the selected TBF ITP.
Use --tbf-itp to override the TBF.itp inside the archive.
TIP3P OW/HW LJ parameters are read from ffnonbonded.itp inside the TAR.
LAMMPS Atoms # full includes ix,iy,iz PBC image flags.
The PDB keeps each molecule whole, including box-straddling molecules.
The source charges remain those in the molecule ITP, not the atomtype defaults.
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path
import math
import tarfile
import numpy as np

KJ_TO_KCAL = 1.0 / 4.184


def read_archive(path):
    ans = {}
    with tarfile.open(path, 'r:*') as tar:
        for member in tar:
            if member.isfile():
                name = Path(member.name).name.lower()
                if name in ('tbf.itp', 'tip3p.itp', 'tbf.pdb', 'tip3p.pdb', 'tip3p.gro', 'ffnonbonded.itp'):
                    if name in ans:
                        raise ValueError('Duplicate file in archive: ' + name)
                    ans[name] = tar.extractfile(member).read().decode()
    for name in ('tbf.itp', 'tip3p.itp', 'tbf.pdb', 'ffnonbonded.itp'):
        if name not in ans:
            raise ValueError('Missing archive member ' + name)
    if 'tip3p.gro' not in ans and 'tip3p.pdb' not in ans:
        raise ValueError('Need tip3p.gro or tip3p.pdb')
    return ans


def itp_sections(src):
    sec = None
    result = defaultdict(list)
    # tip3p contains #ifndef FLEXIBLE: retain only rigid-water main branch.
    branch = []
    for raw in src.splitlines():
        line = raw.split(';', 1)[0].strip()
        if line.startswith('#ifndef FLEXIBLE'):
            branch.append(True); continue
        if line.startswith('#else') and branch:
            branch[-1] = not branch[-1]; continue
        if line.startswith('#endif') and branch:
            branch.pop(); continue
        if line.startswith('#') or (branch and not all(branch)):
            continue
        if line.startswith('[') and ']' in line:
            sec = line.split(']', 1)[0].strip('[ ]').lower()
        elif sec and line:
            result[sec].append(line.split())
    return result


def read_nonbonded_atomtypes(content):
    """Read GROMACS [atomtypes] as type -> (sigma_nm, epsilon_kJ).

    Supports standard GROMACS atomtypes with 6, 7 or 8 fields;
    the last two fields are sigma and epsilon for combination rule 3.
    """
    sections = itp_sections(content)
    result = {}
    for row in sections.get('atomtypes', []):
        if len(row) < 6:
            raise ValueError(f'Invalid [atomtypes] record: {row}')
        key = row[0]
        values = (float(row[-2]), float(row[-1]))
        if key in result and result[key] != values:
            raise ValueError(f'Conflicting [atomtypes] values for {key}')
        result[key] = values
    return result


def read_pdb(text):
    names, elems, coords = [], [], []
    for l in text.splitlines():
        if l.startswith(('ATOM  ', 'HETATM')):
            name = l[12:16].strip()
            element = l[76:78].strip().upper() or name[0].upper()
            names.append(name); elems.append(element)
            coords.append([float(l[i:i+8]) for i in (30, 38, 46)])
    return names, elems, np.asarray(coords, dtype=float)


def read_gro(text):
    lines = text.splitlines()
    n = int(lines[1])
    names = [l[10:15].strip() for l in lines[2:2+n]]
    coords = np.array([[float(l[i:i+8])*10 for i in (20, 28, 36)]
                       for l in lines[2:2+n]], dtype=float)
    elems = ['O' if x.upper().startswith('O') else 'H' for x in names]
    return names, elems, coords


def normalize_template(itp, pdb):
    atoms = itp['atoms']
    names, elements, xyz = pdb
    expected = [x[4] for x in atoms]
    if len(atoms) != len(xyz) or names != expected:
        raise ValueError(f'ITP/PDB mismatch: {expected} versus {names}')
    xyz = xyz - xyz.mean(axis=0)
    return atoms, elements, xyz


def read_bonded(sec, label, expected_func):
    result = []
    for row in sec.get(label, []):
        n = {'bonds':2, 'angles':3, 'dihedrals':4}[label]
        if int(row[n]) != expected_func:
            raise ValueError(f'Unsupported {label} function: {row}')
        result.append((tuple(int(x)-1 for x in row[:n]), tuple(float(x) for x in row[n+1:])))
    return result


def calc_dist3_pairs(n, bonds):
    adj = [set() for _ in range(n)]
    for indices, _ in bonds:
        a, b = indices
        adj[a].add(b); adj[b].add(a)
    pairs = set()
    for start in range(n):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if j not in dist:
                    dist[j] = dist[i] + 1
                    queue.append(j)
        pairs.update((start,j) for j, d in dist.items() if d == 3 and start < j)
    return pairs


def rb_to_opls(c):
    if len(c) != 6:
        raise ValueError('Ryckaert-Bellemans dihedral requires 6 C coefficients')
    c0,c1,c2,c3,c4,c5 = c
    return (-2*c1-1.5*c3-1.25*c5, -c2-c4,
            -0.5*c3-0.625*c5, -0.25*c4)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def pack(templates, counts, box, seed, cutoff, max_tries):
    """PBC-aware random insertion based on distances between non-H atoms.

    Designed for initialization, NOT a final density/energy equilibration.
    """
    rng = np.random.default_rng(seed)
    centers = []
    placed_heavy = np.empty((0,3), dtype=float)
    instances = []
    # Big molecules first; water afterwards efficiently fills remaining gaps.
    for kind in ('TBF', 'SOL'):
        if counts[kind] == 0:
            continue
        atom_rows, elems, xyz = templates[kind]
        heavy = np.array([i for i,e in enumerate(elems) if e != 'H'], dtype=int)
        for index in range(counts[kind]):
            for attempt in range(max_tries):
                trial = xyz @ random_rotation(rng).T
                trial += rng.random(3)*box
                points = np.mod(trial[heavy], box)
                if len(placed_heavy):
                    # Chunk to keep memory controlled for larger builds.
                    ok = True
                    for p in points:
                        delta = placed_heavy-p
                        delta -= box*np.rint(delta/box)
                        if np.any(np.einsum('ij,ij->i', delta,delta) < cutoff**2):
                            ok = False; break
                    if not ok:
                        continue
                instances.append((kind,trial))
                placed_heavy = np.concatenate((placed_heavy, points),axis=0)
                break
            else:
                raise RuntimeError('Packing failed: increase --box, reduce --min-heavy, '
                                   'or increase --max-tries. Try Packmol for dense systems.')
        print(f'Placed {counts[kind]} {kind} molecules')
    return instances



def unwrap_by_mic(wrapped, bond_list, box):
    """Reconstruct a contiguous molecule by BFS along its bond graph using MIC.

    Returns unwrapped positions whose root atom has image (0,0,0).
    Assumes every individual bond is shorter than half the box length.
    """
    n = len(wrapped)
    adjacency = [[] for _ in range(n)]
    for pair, _ in bond_list:
        i, j = pair
        adjacency[i].append(j)
        adjacency[j].append(i)
    if not n:
        return wrapped.copy()
    xyz = np.empty_like(wrapped)
    visited = np.zeros(n, dtype=bool)
    for root in range(n):
        if visited[root]:
            continue
        xyz[root] = wrapped[root]
        visited[root] = True
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j in adjacency[i]:
                delta = wrapped[j] - wrapped[i]
                delta -= box * np.rint(delta / box)
                candidate = xyz[i] + delta
                if not visited[j]:
                    xyz[j] = candidate
                    visited[j] = True
                    queue.append(j)
                elif np.linalg.norm(xyz[j] - candidate) > 1.e-5:
                    raise ValueError('Inconsistent MIC unwrap: bond graph crosses box ambiguously')
    return xyz


def wrap_with_images(unwrapped, box):
    """LAMMPS x = x_wrapped + nx*L (similarly y,z)."""
    images = np.floor(unwrapped / box).astype(np.int64)
    wrapped = unwrapped - images * box
    # Protect against roundoff on the upper boundary.
    wrapped = np.mod(wrapped, box)
    return wrapped, images


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--nwater', '-n', type=int, required=True)
    p.add_argument('--ntbf', '-m', type=int, required=True)
    p.add_argument('--box', type=float, required=True, help='cubic box length, nm')
    p.add_argument('--out', type=Path, default=Path('tbf_tip3p'))
    p.add_argument('--tbf-itp', type=Path, default=None,
                   help='Updated TBF ITP; overrides TBF.itp in archive')
    p.add_argument('--ffnonbonded', type=Path, default=None,
                   help='Optional standalone [atomtypes] file override; default: ffnonbonded.itp INSIDE --archive')
    p.add_argument('--htb-sigma', type=float, help='TBF HTB LJ sigma in nm (if not in force field)')
    p.add_argument('--htb-epsilon', type=float, help='TBF HTB LJ epsilon in kJ/mol (if not in force field)')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--min-heavy', type=float, default=2.65, help='minimum heavy-heavy spacing in Angstrom')
    p.add_argument('--max-tries', type=int, default=10000)
    a = p.parse_args()
    if a.nwater < 0 or a.ntbf < 0 or a.nwater+a.ntbf == 0 or a.box <= 0:
        p.error('Counts must be nonnegative, nonzero total, and box positive')
    if a.box*10 <= 2*a.min_heavy:
        p.error('Box too small compared to --min-heavy')

    files = read_archive(a.archive)
    if a.tbf_itp is not None:
        if not a.tbf_itp.is_file():
            p.error(f'Cannot find updated TBF ITP: {a.tbf_itp}')
        files['tbf.itp'] = a.tbf_itp.read_text()
    t = itp_sections(files['tbf.itp'])
    w = itp_sections(files['tip3p.itp'])
    if t['moleculetype'][0][0] != 'TBF' or w['moleculetype'][0][0] != 'SOL':
        raise ValueError('Unexpected molecule names in source ITPs')
    wt = read_gro(files['tip3p.gro']) if 'tip3p.gro' in files else read_pdb(files['tip3p.pdb'])
    templates = {
        'TBF': normalize_template(t, read_pdb(files['tbf.pdb'])),
        'SOL': normalize_template(w, wt)
    }
    if len(templates['TBF'][0]) != 14 or len(templates['SOL'][0]) != 3:
        raise ValueError('Unexpected molecule sizes')
    # TBF defaults specify OPLS geometric combination and 0.5 1-4 scales.
    defaults = t.get('defaults',[])
    if len(defaults)!=1 or defaults[0][:5]!=['1','3','yes','0.5','0.5']:
        raise ValueError(f'Unexpected TBF nonbonded defaults: {defaults}')

    lj = {}
    for row in t.get('atomtypes',[]):
        # type bonded-type mass charge ptype sigma_nm epsilon_kJ
        lj[row[0]] = (float(row[-2]),float(row[-1]))
    if a.ffnonbonded is not None:
        if not a.ffnonbonded.is_file():
            p.error(f'Cannot find nonbonded override: {a.ffnonbonded}')
        nonbonded_content = a.ffnonbonded.read_text()
        ff_source = str(a.ffnonbonded)
    else:
        nonbonded_content = files['ffnonbonded.itp']
        ff_source = 'ffnonbonded.itp (inside archive)'
    nonbonded = read_nonbonded_atomtypes(nonbonded_content)
    for key, val in nonbonded.items():
        if key in lj and any(abs(x-y)>1e-10 for x,y in zip(lj[key],val)):
            raise ValueError(f'Conflicting LJ coefficients for {key} in TBF.itp and {ff_source}')
        lj[key] = val
    if (a.htb_sigma is None) != (a.htb_epsilon is None):
        p.error('Specify both --htb-sigma and --htb-epsilon, or neither')
    if a.htb_sigma is not None:
        lj['HTB'] = (a.htb_sigma, a.htb_epsilon)
    if a.ntbf and 'HTB' not in lj:
        p.error('HTB LJ parameters absent from selected TBF ITP and ffnonbonded.itp')
    for k in ('OW', 'HW'):
        if a.nwater and k not in lj:
            p.error(f'{k} LJ parameters missing from {ff_source}')
    atom_types = list(dict.fromkeys(row[1] for sec in (t,w) for row in sec['atoms']))
    if any(k not in lj for k in atom_types):
        raise ValueError('Missing LJ types: ' + str(set(atom_types)-set(lj)))
    tid = {v:i+1 for i,v in enumerate(atom_types)}
    masses = {}
    for sec in (t,w):
        for row in sec['atoms']:
            key = row[1]
            if len(row)>7:
                mass = float(row[7])
            else:
                mass = {'OW':15.9994, 'HW':1.008}[key]
            if key in masses and abs(masses[key]-mass)>1e-3:
                raise ValueError('Conflicting masses for '+key)
            masses[key] = mass

    tbonds = read_bonded(t,'bonds',1)
    tangles = read_bonded(t,'angles',1)
    tdihed = read_bonded(t,'dihedrals',3)
    # For the provided rigid TIP3P SETTLE geometry, include harmonic-style
    # water bonds/angle and constrain them in input.run with fix shake.
    settle = w.get('settles',[])
    if len(settle)!=1:
        raise ValueError('Expected one [settles] entry in supplied tip3p.itp')
    oh_nm,hh_nm = map(float,settle[0][2:4])
    theta = math.degrees(2*math.asin(hh_nm/(2*oh_nm)))
    # Typical flexible TIP3P harmonic values from the ITP's #else block,
    # used for bookkeeping only; fix shake enforces SETTLE geometry.
    wbonds = [((0,1),(oh_nm,502416.)),((0,2),(oh_nm,502416.))]
    wangles = [((1,0,2),(theta,628.02))]

    pair_expected = calc_dist3_pairs(14,tbonds)
    pair_actual = {tuple(sorted((int(r[0])-1,int(r[1])-1))) for r in t.get('pairs',[])}
    if pair_expected != pair_actual:
        raise ValueError('Explicit TBF [pairs] differ from all topological 1-4 pairs; '
                         'special_bonds 0 0 0.5 would not reproduce the supplied force field')
    if 'dihedrals' not in t or any(len(x)!=6 for _,x in tdihed):
        raise ValueError('Missing/unsupported RB dihedrals')

    inst = pack(templates,{'TBF':a.ntbf,'SOL':a.nwater},a.box*10,a.seed,
                a.min_heavy,a.max_tries)
    a.out.mkdir(parents=True,exist_ok=True)
    # Coefficient types are keyed by physical coefficient values.
    bd,ad,dd = {},{},{}
    def registry(reg, coeff):
        key = tuple(round(float(v),10) for v in coeff)
        if key not in reg: reg[key] = len(reg)+1
        return reg[key]
    atoms, bonds, angles, dihedrals = [],[],[],[]
    pdb_atoms = []
    atomid = 0
    L=a.box*10.
    for molid,(kind,xyz) in enumerate(inst,1):
        sec = t if kind=='TBF' else w
        local_bonds = tbonds if kind=='TBF' else wbonds
        local_angles = tangles if kind=='TBF' else wangles
        local_dihed = tdihed if kind=='TBF' else []
        start=atomid
        # Reconstruct continuous molecules using MIC over the molecular bond graph.
        # This is robust when the random packing straddles periodic boundaries.
        wrapped_input = np.mod(xyz, L)
        contiguous = unwrap_by_mic(wrapped_input, local_bonds, L)
        # Shift whole molecule so its geometric center belongs to the primary box.
        contiguous -= np.floor(contiguous.mean(axis=0) / L) * L
        wrapped, images = wrap_with_images(contiguous, L)
        for row,point,im,whole in zip(sec['atoms'],wrapped,images,contiguous):
            atomid+=1
            typ=tid[row[1]]
            q=float(row[6]); atoms.append((atomid,molid,typ,q,*point,*(int(i) for i in im)))
            # Whole molecular positions for readable PDB without broken bonds.
            pdb_atoms.append((atomid,molid,row[4],kind,whole))
        for ix,par in local_bonds:
            eq,k = par[:2]
            btype=registry(bd,(k*KJ_TO_KCAL/200.,eq*10.))
            bonds.append((len(bonds)+1,btype,*(start+i+1 for i in ix)))
        for ix,par in local_angles:
            eq,k=par[:2]
            atype=registry(ad,(k*KJ_TO_KCAL/2.,eq))
            angles.append((len(angles)+1,atype,*(start+i+1 for i in ix)))
        for ix,par in local_dihed:
            coeff=tuple(x*KJ_TO_KCAL for x in rb_to_opls(par))
            dtype=registry(dd,coeff)
            dihedrals.append((len(dihedrals)+1,dtype,*(start+i+1 for i in ix)))

    data = a.out/'mixture.data'
    def fmt_section(f,title,items):
        f.write('\n'+title+'\n\n')
        for r in items:f.write(' '.join(str(v) if isinstance(v,int) else f'{v:.10f}' for v in r)+'\n')
    with data.open('w') as f:
        f.write('TIP3P + tert-butyl fluoride, units real; generated from supplied ITP/PDB\n\n')
        for s,v in [('atoms',len(atoms)),('bonds',len(bonds)),('angles',len(angles)),('dihedrals',len(dihedrals)),('impropers',0)]:
            f.write(f'{v} {s}\n')
        f.write('\n')
        for s,v in [('atom types',len(tid)),('bond types',len(bd)),('angle types',len(ad)),('dihedral types',len(dd)),('improper types',0)]:
            f.write(f'{v} {s}\n')
        f.write(f'\n0.0 {L:.8f} xlo xhi\n0.0 {L:.8f} ylo yhi\n0.0 {L:.8f} zlo zhi\n')
        fmt_section(f,'Masses',[(tid[k],masses[k]) for k in atom_types])
        # LAMMPS epsilon (kcal/mol), sigma (Angstrom)
        fmt_section(f,'Pair Coeffs # lj/cut/coul/long',[(tid[k],lj[k][1]*KJ_TO_KCAL,lj[k][0]*10) for k in atom_types])
        fmt_section(f,'Bond Coeffs # harmonic',[(v,*k) for k,v in bd.items()])
        fmt_section(f,'Angle Coeffs # harmonic',[(v,*k) for k,v in ad.items()])
        if dd:fmt_section(f,'Dihedral Coeffs # opls',[(v,*k) for k,v in dd.items()])
        f.write('\n# Atoms: id molecule-ID type q x y z nx ny nz\n')
        fmt_section(f,'Atoms # full',atoms)
        fmt_section(f,'Bonds',bonds)
        fmt_section(f,'Angles',angles)
        if dd:fmt_section(f,'Dihedrals',dihedrals)

    pdb=a.out/'mixture.pdb'
    if len(pdb_atoms)>99999 or len(inst)>9999:
        print('WARNING: PDB atom/residue serials exceed classical format limits; '
              'PDB numbers will wrap. LAMMPS data IDs remain correct.')
    with pdb.open('w') as f:
        f.write(f'CRYST1{L:9.3f}{L:9.3f}{L:9.3f}{90.:7.2f}{90.:7.2f}{90.:7.2f} P 1           1\n')
        for idx,molid,name,kind,pos in pdb_atoms:
            elem = 'F' if name.startswith('F') else name[0]
            res='TBF' if kind=='TBF' else 'SOL'
            x,y,z=pos
            f.write(f'ATOM  {idx%100000:5d} {name:<4s} {res:>3s} {molid%10000:4d}    '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {elem:>2s}\n')
        f.write('END\n')

    run=a.out/'in.minimize'
    wbtype=registry(bd,(502416.*KJ_TO_KCAL/200.,oh_nm*10.))
    watype=registry(ad,(628.02*KJ_TO_KCAL/2.,theta))
    run.write_text('''# LAMMPS real units / OPLS 1-4 scaling.\nunits real\natom_style full\nboundary p p p\npair_style lj/cut/coul/long 10.0\npair_modify mix geometric\nbond_style harmonic\nangle_style harmonic\ndihedral_style opls\nspecial_bonds lj/coul 0.0 0.0 0.5\nread_data mixture.data\nkspace_style pppm 1.0e-5\n\n'''+(f'group water type {tid["OW"]} {tid["HW"]}\n'
         f'fix rigidwater water shake 1.0e-5 20 0 b {wbtype} a {watype}\n' if a.nwater else '')+'''
neighbor 2.0 bin\nneigh_modify every 1 delay 0 check yes\nthermo 100\nthermo_style custom step temp pe etotal press density\n# Verify initial structure with a minimization before MD.\nmin_style cg\nminimize 1.0e-4 1.0e-6 2000 10000\nwrite_data minimized.data\n''')

    ffnotes = a.out/'forcefield_notes.txt'
    ffnotes.write_text('LAMMPS units real; atom_style full; pair_style lj/cut/coul/long\n'
      'TIP3P water is rigid via fix shake, geometry from supplied SETTLE.\n'
      'TBF RB dihedrals converted exactly (up to a constant) to OPLS style.\n'
      'Geometric mixing (OPLS combination rule 3).\n'
      'special_bonds lj/coul 0 0 0.5; explicit TBF [pairs] checked against graph.\n'
      f'Water LJ values read from {ff_source}.\n'
      f'OW sigma_nm={lj.get("OW")}\n'
      f'HW sigma_nm={lj.get("HW")}\n'
      f'HTB sigma_nm / epsilon_kJ = {lj.get("HTB")}\n'
      'Data file contains x y z nx ny nz for each atom; MIC-based bond graph\n'
      'unwrapping ensures molecules are contiguous when image flags are applied.\n'
      'PDB writes whole-molecule coordinates (some may lie outside CRYST1).\n'
      'Validate force-field energies against reference GROMACS before production.\n')
    print(f'Created {data}, {pdb}, {run}')
    print(f'Counts: {len(inst)} molecules; {len(atoms)} atoms, {len(bonds)} bonds, '
          f'{len(angles)} angles, {len(dihedrals)} dihedrals')
    print(f'Loaded TIP3P LJ parameters from {ff_source}: OW={lj.get("OW")}, HW={lj.get("HW")}')

if __name__=='__main__':
    main()
