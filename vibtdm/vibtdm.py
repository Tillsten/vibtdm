# -*- coding: utf-8 -*-
from pathlib import Path
from collections import defaultdict
import numpy as np
import attr
import matplotlib
import Bio.PDB as PDB

from typing import Optional, Tuple, Dict, Set
Residue = PDB.Residue
from Bio.PDB.vectors import Vector
from geometry import project_on_plane


class PDBFile:
    def __init__(self, fname: str, id: str = None):
        """
        Helper class to work with the pdb file. Takes a file name to a pdb
        file.
        """
        if fname[-3:] == 'pdb':
            self.PDB = PDB.PDBParser()
        elif fname[-3:] == 'cif':
            self.PDB = PDB.FastMMCIFParser()
        if id is None:
            id = fname

        self.struct = self.PDB.get_structure(
            id, fname)  #.child_list[0].child_dict['A']
        self.res = PDB.Selection.unfold_entities(self.struct, "R")
        self.kd = PDB.NeighborSearch(list(self.struct.get_atoms()))

    @classmethod
    def from_pdb_id(cls, pdb_id: str, save_dir=None):
        """Download PDB file from ID, will saved in the homedir"""
        if save_dir is None:
            pdbl = PDB.PDBList()
            save_dir = str(Path.home())
        fname = pdbl.retrieve_pdb_file(pdb_id,
                                       file_format="mmCif",
                                       pdir=save_dir)
        return cls(fname)

    def get_res(self, res_name: str):
        match_res = filter(lambda r: r.get_resname() == res_name, self.res)
        return next(match_res)

    def find_near(self,
                  res_name: str,
                  radius: float = 5,
                  res_type=None,
                  atom_names=None,
                  exclude_self=False):
        """Find residues near given residue, currently assumes the residues is
        unique"""
        match_res = filter(lambda r: r.get_resname() == res_name, self.res)
        center_res = next(match_res)
        nearby_res: Set[PDB.Residue] = set()
        for atom in center_res:
            if atom_names is not None:
                if atom.name not in atom_names:
                    continue
            nearby_res |= set(self.kd.search(atom.coord, radius, level="R"))
        if exclude_self:
            out = list(nearby_res - {center_res})
        else:
            out = list(nearby_res)
        if res_type is not None:
            out = [i for i in out if i.get_resname() == res_type]
        return out

    def calculate_angles(self,
                         residue: PDB.Residue,
                         other_tdm: PDB.Vector = None,
                         do_project: bool = True):
        res = ResModes(residue)
        if other_tdm is not None:
            for v in res.modes.values():
                v.angle = np.rad2deg(v.tdm.angle(other_tdm))
                assert v.angle is not None
                if do_project:
                    if v.angle > 90.:
                        v.angle = 180. - v.angle
        return res

    def atom_vec_angles(self,
                        a: PDB.Atom,
                        b: PDB.Atom,
                        tdm: PDB.Atom,
                        do_project: bool = True):
        vec = b.get_vector() - a.get_vector()
        angle = np.rad2deg(tdm.angle(vec))

        if do_project:
            if angle > 90.:
                angle = 180. - angle
        return angle


def make_coord_array(res: PDB.Residue) -> np.ndarray:
    out = np.empty((3, len(res.child_list)))
    for i, a in enumerate(res.get_atoms()):
        out[:, i] = a.get_coord()
    return out


@attr.s(auto_attribs=True)
class Vibration:
    """Single Vibration"""
    name: str
    freq_range: Tuple[float, float] = attr.ib()
    strength: float = attr.ib(0)
    tdm: Vector = attr.ib(None)
    origin: Optional[Vector] = attr.ib(None)
    angle: Optional[float] = attr.ib(None)


WATER_NAMES = ("HOH", "WAT", "TIP3W", "TIP", "LYS")


@attr.s(auto_attribs=True)
class ResModes:
    residue: PDB.Residue
    modes: Dict[str, Vibration] = attr.Factory(dict)
    use_protonation: bool = False

    def vec(self, a: Optional[str], b=None) -> PDB.Vector:
        "Helper function to extract atom-vectors and atom-atom vectors"
        if b is not None:
            return self.residue[a].get_vector() - self.residue[b].get_vector()
        else:
            return self.residue[a].get_vector()

    def get_modes(self, kind: str):
        iter = filter(lambda x: x.startswith(kind), self.modes)
        return list(iter)

    def __attrs_post_init__(self):
        "Calculate all known modes"
        resname = self.residue.get_resname()
        #if resname in WATER_NAMES:  # Dont handle water
        #    return
        if not self.residue.id[0] == ' ':
            return
        self.amide_one()
        if resname in ["ASP", "GLU"]:
            is_protonated = any(k in res for k in ("HE1", "HE2", "HD1", "HD2"))
            if not is_protonated or not self.use_protonation:
                self.carboxylate_antisym(resname)
            if is_protonated or not self.use_protonation:
                self.carbonyl_stretch(resname)

    def amide_one(self):
        """Estimates the direction of the Amide I vibration by taking
        the C=O vector of the petide group and rotate it accordingly."""
        own_id = self.residue.id[1]
        next_N = self.residue.parent[self.residue.id[1] + 1]['N'].get_vector()
        vec_cn = -(self.vec('C') - next_N)
        vec_co = self.vec('C', 'O')
        normal = (vec_cn**vec_co).normalized()
        for angle, name in zip((0, 10, 20), ("No rot", "D2O", "H2O")):
            rot_mat = PDB.rotaxis(np.deg2rad(-angle), normal)
            vec = vec_co.left_multiply(rot_mat)
            vib = Vibration(name='Amide 1 %s' % name,
                            freq_range=(1640, 1660),
                            strength=3,
                            tdm=vec,
                            angle=np.nan,
                            origin=self.vec('C'))
            self.modes[vib.name] = vib

    def carboxylate_antisym(self, resname="ASP"):
        """Returns an estimate of the antisymetric COO- vibration by
        returning the O-O vector."""
        if resname == "ASP":
            vec = self.vec('OD1', 'OD2')
            freq_range = (1565, 1600)
            origin = self.vec('OD1')
        elif resname == "GLU":
            vec = self.vec('OE1', 'OE2')
            freq_range = (1555, 1600)
            origin = self.vec('OE1')

        vib = Vibration(
            name="COO- AS ",
            freq_range=freq_range,
            strength=4,
            tdm=vec,
            angle=np.nan,
            origin=origin,
        )
        self.modes[vib.name] = vib

    def carbonyl_stretch(self, resname):
        """Returns the carbonyl stretching by returning the C=O vector"""

        if resname == "ASP":
            origin = self.vec('CG')
            freq_range = (1690, 1740)
            if "HD1" in res or not self.use_protonation:
                vec = self.vec('OD2', 'CG')
                vib = Vibration(
                    name="CO (OD2)",
                    freq_range=freq_range,
                    tdm=vec,
                    origin=origin,
                )
                self.modes[vib.name] = vib
            if "HD2" in res or not self.use_protonation:
                vec = self.vec('OD1', 'CG')
                vib = Vibration(
                    name="CO (OD1)",
                    freq_range=freq_range,
                    tdm=vec,
                    origin=origin,
                )
                self.modes[vib.name] = vib

        if resname == "GLU":
            origin = self.vec('CD')
            freq_range = (1690, 1740)
            if "HE1" in res or not self.use_protonation:
                vec = self.vec('OE2', 'CD')
                vib = Vibration(
                    name="CO (OE2)",
                    freq_range=freq_range,
                    tdm=vec,
                    origin=origin,
                )
                self.modes[vib.name] = vib
            if "HE2" in res or not self.use_protonation:
                vec = self.vec('OE1', 'CD')
                vib = Vibration(name="CO (OE1)",
                                freq_range=freq_range,
                                origin=origin,
                                tdm=vec)
                self.modes[vib.name] = vib

    def ring_breathmode(self, res_name):
        if res_name == 'TRP':
            v1 = self.residue["CD1"].get_vector()
            vec = v1 - self.residue["CG"].get_vector()
            freq_range = (1615, 1625)
            origin = self.vec("CG")

        if res_name == 'TYR':
            vec = self.vec("CZ", "CG")
            freq_range = (1490, 1510)
            origin = self.vec("CG")

        vib = Vibration(
            name="CC",
            freq_range=freq_range,
            strength=4,
            tdm=vec,
            origin=origin,
        )
        self.modes[vib.name] = vib


if __name__ == "__main__":

    struct = PDBFile.from_pdb_id('5ZIM')
    ns = struct.find_near("RET", 10, atom_names=('C15', 'C13'))

    retinal = struct.get_res('RET')
    vec = retinal['C14'].get_vector() - retinal['C10'].get_vector()

    by_type: Dict[str, Dict[str, float]] = defaultdict(dict)
    by_res = {}
    for res in ns:
        r = struct.calculate_angles(res, vec)
        resname = res.get_resname() + str(res.id[1])
        resname = resname[:1] + resname[1:3].lower() + resname[3:]
        for i in r.modes:
            by_type[i][resname] = r.modes[i].angle
        by_res[resname] = r

    waters = 404, 406, 407
    chain = struct.struct[0]['A']
    chain_dict = {res.id[1]: res for res in chain}
    print("Water angles\n-----------------")
    for i in waters:
        for j in waters:
            if i == j:
                pass
            else:
                a = chain_dict[i]['O']
                b = chain_dict[j]['O']
                print(i, j,  a-b, struct.atom_vec_angles(a, b, vec))

    print("CO stretch angles\n-----------------")
    for r in by_res:
        for m in by_res[r].get_modes('CO'):
            print(f"{r} {m} {by_res[r].modes[m].angle: .1f}")

