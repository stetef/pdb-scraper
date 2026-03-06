#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
from larch import Group
from larch.xafs import xftf


def read_chik_k_chi(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    k_vals: list[float] = []
    chi_vals: list[float] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            try:
                k_vals.append(float(parts[0]))
                chi_vals.append(float(parts[1]))
            except ValueError:
                continue

    if not k_vals:
        raise ValueError(f"No numeric chi(k) data found in {file_path}")

    return np.asarray(k_vals), np.asarray(chi_vals)

def xftf_larch(k, chi, kmin, kmax, dk, kweight, kstep, rmax_out):
    grp = Group()
    grp.k = k
    grp.chi = chi
    xftf(
        grp.k,
        grp.chi,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        kweight=kweight,
        kstep=kstep,
        rmax_out=rmax_out,
        window="hanning",
        group=grp,
    )
    return grp.r, grp.chir

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert chi(k) from ATHENA to chi(R) by calling Larch's xfft."
        )
    )
    parser.add_argument(
        "chi_k_path",
        type=Path,
        help=(
            "Path to file with chi(k). Assumes from ATHENA."
        ),
    )
    parser.add_argument("--kmin", type=float, default=3.0)
    parser.add_argument("--kmax", type=float, default=11.0)
    parser.add_argument("--dk", type=float, default=1.0)
    parser.add_argument("--kweight", type=int, default=2)
    parser.add_argument("--kstep", type=float, default=0.05)
    parser.add_argument("--rmax", type=float, default=6.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    k, chi = read_chik_k_chi(args.chi_k_path)
    
    r, chir = xftf_larch(
            k,
            chi,
            kmin=args.kmin,
            kmax=args.kmax,
            dk=args.dk,
            kweight=args.kweight,
            kstep=args.kstep,
            rmax_out=args.rmax,
        )
    
    chir_mag = np.abs(chir)
    chir_re = chir.real
    chir_im = chir.imag

    out_dir = args.chi_k_path.parent
    fname = args.chi_k_path.stem
    out_dat = out_dir / f"{fname}_chi_R.dat"
    header = "r  chir_mag  chir_re  chir_im"
    np.savetxt(out_dat, np.column_stack([r, chir_mag, chir_re, chir_im]), header=header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())