#!/usr/bin/env python3
"""Score Gate 2 arm A dumps.

Theta is rebuilt from the plan (rho, V, dt, outgoing area-weighted flux,
1-4096 eps). It is never read from the code. O3 uses YAML u0_R, cv_R, Tref
and R = 8.31446 only.
"""

import math
import sys
from pathlib import Path

import numpy as np

R_GAS = 8.31446
EPS = np.finfo(np.float64).eps
MARGIN = 4096.0 * EPS
DT = 1.0

EXPECT = {
    "adv-lmars": ("x1", 5),
    "adv-hllc": ("x1", 5),
    "settling": ("x1", 5),
    "x2": ("x2", 30),
    "donors": ("x1", 4),
    "mixed": ("x1", 5),
    "vapor-u0-lmars": ("x1", 5),
    "vapor-u0-hllc": ("x1", 5),
}

# mutation -> hook species that must miss O3 ("v" vapor, "c" cloud)
AFFECTED = {
    1: ("v", "c"),
    2: ("v", "c"),
    3: ("c",),
    4: ("c",),
    5: ("c",),
}


def within(err, abs_tol, rel_tol, scale):
    return abs(err) <= abs_tol + rel_tol * scale


def load_dump(path):
    meta = {"species": [], "bad": []}
    arrays = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# species "):
            p = line.split()
            # # species IDX NAME M_code VAL M_yaml VAL cv_R VAL u0_R VAL cloud VAL
            meta["species"].append(
                {
                    "i": int(p[2]),
                    "name": p[3],
                    "M_code": float(p[5]),
                    "M_yaml": float(p[7]),
                    "cv_R": float(p[9]),
                    "u0_R": float(p[11]),
                    "cloud": int(p[13]),
                }
            )
        elif line.startswith("# bad "):
            p = line.split()
            meta["bad"].append((int(p[2]), int(p[3]), int(p[4])))
        elif line.startswith("# index "):
            p = line.split()
            meta["index"] = {p[j]: int(p[j + 1]) for j in range(2, len(p), 2)}
        elif line.startswith("# il "):
            p = line.split()
            meta.update({p[j]: int(p[j + 1]) for j in range(1, len(p), 2)})
        elif line.startswith("# "):
            p = line[2:].split()
            if len(p) == 2:
                key, val = p
                try:
                    meta[key] = int(val) if key not in (
                        "i2_hke_over_s",
                        "max_abs_b2_off",
                        "Tref",
                        "Rgas",
                        "kintera_Rgas",
                    ) else float(val)
                except ValueError:
                    meta[key] = val
            elif len(p) >= 2:
                meta[p[0]] = " ".join(p[1:])
        elif line.startswith("BEGIN "):
            tok = line.split()
            name = tok[1]
            dims = [int(x) for x in tok[2:]]
            n = 1
            for d in dims:
                n *= d
            if not dims or n == 0:
                arrays[name] = None
                i += 1
                if i >= len(lines) or lines[i] != "END":
                    raise RuntimeError(f"{path}: {name} missing END")
            else:
                vals = [float(lines[i + 1 + k]) for k in range(n)]
                arrays[name] = np.array(vals, dtype=np.float64).reshape(dims)
                i += n + 1
                if lines[i] != "END":
                    raise RuntimeError(f"{path}: {name} END at {lines[i]!r}")
        i += 1
    meta["path"] = str(path)
    return meta, arrays


def load_summary(path):
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] in ("census_a", "census_b2") and len(p) == 5:
            out[p[0]] = [int(x) for x in p[1:]]
        elif p[0] == "b2_bad":
            out.setdefault("b2_bad", []).append(tuple(int(x) for x in p[1:]))
        elif len(p) == 2:
            try:
                out[p[0]] = float(p[1]) if "." in p[1] or "e" in p[1].lower() else int(p[1])
            except ValueError:
                out[p[0]] = p[1]
        else:
            out[p[0]] = p[1:]
    return out


def interior_slice(meta):
    return (
        slice(meta["kl"], meta["ku"] + 1),
        slice(meta["jl"], meta["ju"] + 1),
        slice(meta["il"], meta["iu"] + 1),
    )


def composition(w, idx):
    """Total density, dry/vapor/cloud partial densities, mass fractions."""
    rho = w[idx["IDN"]]
    ny = w.shape[0] - idx["ICY"]
    y = w[idx["ICY"] : idx["ICY"] + ny]
    ycloud = y[1:].sum(axis=0) if ny > 1 else 0.0
    yv = y[0]
    yd = 1.0 - y.sum(axis=0)
    return rho, rho * yd, rho * yv, rho * y[1] if ny > 1 else np.zeros_like(rho), yd, yv


def ke_of(w, idx):
    return 0.5 * (w[idx["IVX"]] ** 2 + w[idx["IVY"]] ** 2 + w[idx["IVZ"]] ** 2)


def yaml_T(p, rho_d, rho_v, species):
    md, mv = species[0]["M_code"], species[1]["M_code"]
    return p / (R_GAS * (rho_d / md + rho_v / mv))


def h_o3(species, T, ke):
    """Per-species specific enthalpy, dry included. Shape (3, ...)."""
    out = []
    for sp in species:
        z = 0.0 if sp["cloud"] else 1.0
        u = (R_GAS / sp["M_code"]) * (
            sp["u0_R"] + sp["cv_R"] * (T - sp_tref(sp)) + z * T
        )
        out.append(u + ke)
    return np.stack(out, axis=0)


def sp_tref(sp):
    return sp["Tref"]


def attach_tref(species, tref):
    for sp in species:
        sp["Tref"] = tref
    return species


def o3_check(meta, arr):
    idx = meta["index"]
    w = arr["W"]
    species = attach_tref(meta["species"], float(meta["Tref"]))
    rho, rd, rv, rc, yd, yv = composition(w, idx)
    ke = ke_of(w, idx)
    T = yaml_T(w[idx["IPR"]], rd, rv, species)
    h = h_o3(species, T, ke)
    sl = interior_slice(meta)
    tcode = arr["TCODE"]
    dT = np.abs(T[sl] - tcode[sl])
    scale_T = np.maximum(np.abs(T[sl]), np.abs(tcode[sl]))
    t_ok = bool(np.all(dT <= 1e-12 + 7e-14 * scale_T)) if dT.size else True
    hook = arr["H"]  # (ny, nc3, nc2, nc1), dry dropped
    # species 1 vapor -> hook 0, species 2 cloud -> hook 1
    h_bad = []
    max_rel = []
    for s, hs in ((1, 0), (2, 1)):
        err = np.abs(hook[hs][sl] - h[s][sl])
        scale = np.maximum(np.abs(hook[hs][sl]), np.abs(h[s][sl]))
        tol = 1e-9 + 1e-12 * scale
        h_bad.append(float(np.max(err / tol)) if err.size else 0.0)
        max_rel.append(float(np.max(err / np.maximum(scale, 1e-300))) if err.size else 0.0)
    # I1: dry from O3, vapor and cloud from the hook
    U = np.zeros_like(T)
    for sp, rho_n in zip(species, (rd, rv, rc)):
        U = U + rho_n * (R_GAS / sp["M_code"]) * (
            sp["u0_R"] + sp["cv_R"] * (T - sp["Tref"])
        )
    lhs = rd * h[0] + rv * hook[0] + rc * hook[1]
    rhs = U + w[idx["IPR"]] + rho * ke
    err = np.abs(lhs[sl] - rhs[sl])
    scale = np.maximum(np.abs(lhs[sl]), np.abs(rhs[sl]))
    i1 = float(np.max(err / (1e-6 + 1e-12 * scale))) if err.size else 0.0
    m_void = False
    m_worst = 0.0
    for sp in species:
        den = max(abs(sp["M_code"]), abs(sp["M_yaml"]), 1e-300)
        rel = abs(sp["M_code"] - sp["M_yaml"]) / den
        m_worst = max(m_worst, rel)
        if rel > 2e-5:
            m_void = True
    return {
        "T_ok": t_ok,
        "T_max": float(dT.max()) if dT.size else 0.0,
        "h_over": h_bad,
        "h_rel": max_rel,
        "i1_over": i1,
        "M_rel": m_worst,
        "M_void": m_void,
        "h": h,
        "T": T,
        "ke": ke,
        "rho": (rho, rd, rv, rc),
    }


def face_axes(meta, flux, area):
    """Return the flux axis and the area axis for this direction."""
    # flux (nvar, x3, x2, x1-or-face); area (x3, x2, face) or (x3, face, x1)
    if flux.shape[-1] == area.shape[-1] and flux.shape[2] == area.shape[1]:
        return 3, 2  # x1
    if flux.shape[2] == area.shape[1]:
        return 2, 1  # x2
    raise RuntimeError(f"cannot align flux {flux.shape} area {area.shape}")


def accumulate_out(rho_s, flux_s, area, lo, hi, flux_axis):
    """Outgoing area-weighted flux on the conserved grid. rho_s/flux_s lack no species dim."""
    nf = hi - lo + 2
    # Spatial tensors: the face axis is the same index as the cell axis.
    area_axis = flux_axis
    sl_f = [slice(None)] * flux_s.ndim
    sl_f[flux_axis] = slice(lo, lo + nf)
    sl_a = [slice(None)] * area.ndim
    sl_a[area_axis] = slice(lo, lo + nf)
    f = flux_s[tuple(sl_f)]
    a = area[tuple(sl_a)]
    af = a * f
    upper = np.maximum(af[tuple(_span(af.ndim, flux_axis, 1, nf))], 0.0)
    lower = np.maximum(-af[tuple(_span(af.ndim, flux_axis, 0, nf - 1))], 0.0)
    out = np.zeros_like(rho_s)
    dest = [slice(None)] * out.ndim
    dest[flux_axis] = slice(lo, hi + 1)
    out[tuple(dest)] = upper + lower
    return out


def _span(ndim, axis, start, stop):
    sl = [slice(None)] * ndim
    sl[axis] = slice(start, stop)
    return tuple(sl)


def theta_of(rho_s, vol, out):
    avail = np.maximum(rho_s, 0.0) * vol * (1.0 - MARGIN)
    drain = DT * out
    th = np.ones_like(rho_s)
    m = drain > 0
    th[m] = np.minimum(1.0, avail[m] / np.maximum(drain[m], 1e-300))
    return th


def _at(t, s, k, j, i, axis, face):
    """Index a (species, x3, x2, x1) tensor. `face` replaces the axis index."""
    if axis == 3:
        return t[s, k, j, face]
    if axis == 2:
        return t[s, k, face, i]
    raise RuntimeError(axis)


def _cell(t, s, k, j, i):
    return t[s, k, j, i]


def score_faces(meta, ref, on, o3):
    """Carry residuals. Spatial axis 2 is x1, axis 1 is x2, on rank-3 fields."""
    idx = meta["index"]
    w = ref["W"]
    u = ref["U"]
    icy = idx["ICY"]
    ny = 2
    rho_s = u[icy : icy + ny]
    vol = ref["VOL"]
    counts = {"x1": 0, "x2": 0}
    nface = 0
    worst = {"Rs": 0.0, "RE": 0.0, "RM": 0.0}
    vel_keys = ("IVX", "IVY", "IVZ")

    def take(flux, s, k, j, i, dim, face):
        if dim == "x1":
            return flux[s, k, j, face]
        return flux[s, k, face, i]

    for dim, flux_name, area_name, lo_key, hi_key, axis in (
        ("x1", "FLUX1", "AREA1", "il", "iu", 2),
        ("x2", "FLUX2", "AREA2", "jl", "ju", 1),
    ):
        flux_r = ref[flux_name]
        flux_o = on[flux_name]
        area = ref[area_name]
        if flux_r is None or area is None:
            continue
        lo, hi = meta[lo_key], meta[hi_key]
        fsed = ref["FSED1"] if dim == "x1" else None
        outs = []
        for s in range(ny):
            outs.append(
                accumulate_out(rho_s[s], flux_r[icy + s], area, lo, hi, axis)
            )
        th = theta_of(rho_s, vol, np.stack(outs, axis=0))
        k0, k1 = meta["kl"], meta["ku"]
        j0, j1 = meta["jl"], meta["ju"]
        i0, i1 = meta["il"], meta["iu"]
        faces = range(lo, hi + 2)
        for f in faces:
            if dim == "x1":
                coords = [(k, j, 0) for k in range(k0, k1 + 1) for j in range(j0, j1 + 1)]
            else:
                coords = [(k, 0, i) for k in range(k0, k1 + 1) for i in range(i0, i1 + 1)]
            for k, j, i in coords:
                limited = False
                sum_e = 0.0
                sum_m = [0.0, 0.0, 0.0]
                for s in range(ny):
                    Fref = take(flux_r, icy + s, k, j, i, dim, f)
                    Fon = take(flux_o, icy + s, k, j, i, dim, f)
                    sed = 0.0
                    if fsed is not None:
                        sed = take(fsed, s, k, j, i, dim, f)
                    adv = Fref - sed
                    donor_net = (f - 1) if Fref > 0.0 else f
                    if dim == "x1":
                        th_d = th[s, k, j, donor_net]
                    else:
                        th_d = th[s, k, donor_net, i]
                    if th_d < 1.0 and Fref != 0.0:
                        limited = True
                    share = 1.0 - th_d
                    Rs = Fon - th_d * Fref
                    scale = max(abs(Fon), abs(th_d * Fref))
                    worst["Rs"] = max(worst["Rs"], abs(Rs) / (1e-15 + 1e-13 * scale))
                    for part, dm in ((adv, share * adv), (sed, share * sed)):
                        dcell = (f - 1) if part > 0.0 else f
                        if dim == "x1":
                            kk, jj, ii = k, j, dcell
                        else:
                            kk, jj, ii = k, dcell, i
                        sum_e += dm * o3["h"][s + 1, kk, jj, ii]
                        for c, key in enumerate(vel_keys):
                            sum_m[c] += dm * w[idx[key], kk, jj, ii]
                Fre = take(flux_r, idx["IPR"], k, j, i, dim, f)
                Foe = take(flux_o, idx["IPR"], k, j, i, dim, f)
                RE = (Fre - Foe) - sum_e
                worst["RE"] = max(
                    worst["RE"], abs(RE) / (1e-9 + 1e-12 * max(abs(Fre), abs(sum_e)))
                )
                for c, key in enumerate(vel_keys):
                    Fm = take(flux_r, idx[key], k, j, i, dim, f)
                    Fo = take(flux_o, idx[key], k, j, i, dim, f)
                    RM = (Fm - Fo) - sum_m[c]
                    worst["RM"] = max(
                        worst["RM"], abs(RM) / (1e-12 + 1e-12 * max(abs(Fm), abs(sum_m[c])))
                    )
                nface += 1
                if limited:
                    counts[dim] += 1
    return counts, worst, nface


def column_over(meta, on, ref):
    idx = meta["index"]
    sl = interior_slice(meta)
    rows = [idx["IDN"], idx["IVX"], idx["IVY"], idx["IVZ"], idx["IPR"]]
    icy = idx["ICY"]
    ny = on["DU"].shape[0] - icy
    rows += list(range(icy, icy + ny))
    worst = 0.0
    for r in rows:
        a = on["DU"][r][sl]
        b = ref["DU"][r][sl]
        err = abs(float(a.sum() - b.sum()))
        tol = 1e-12 + 1e-12 * float(np.abs(b).sum())
        worst = max(worst, err / tol if tol else 0.0)
    return worst


def mutation_over(meta, mut, base_o3, ref_flux_pack):
    """How many times tolerance the mutated hook and its carry miss."""
    idx = meta["index"]
    w = mut["W"]
    species = attach_tref(meta["species"], float(meta["Tref"]))
    # Recompute O3 on the mutated run's own primitive state (same IC).
    rho, rd, rv, rc, _, _ = composition(w, idx)
    ke = ke_of(w, idx)
    T = yaml_T(w[idx["IPR"]], rd, rv, species)
    h = h_o3(species, T, ke)
    sl = interior_slice(meta)
    hook = mut["H"]
    which = AFFECTED[int(meta["mutation"])]
    h_over = 0.0
    for tag, s, hs in (("v", 1, 0), ("c", 2, 1)):
        if tag not in which:
            continue
        err = np.abs(hook[hs][sl] - h[s][sl])
        scale = np.maximum(np.abs(hook[hs][sl]), np.abs(h[s][sl]))
        tol = 1e-9 + 1e-12 * scale
        h_over = max(h_over, float(np.max(err / tol)))
    # Carry redness is scored by the caller with score_faces against B2.
    return h_over


def g(x):
    return "%.17g" % float(x)


def clamp_split(meta, arr, floor=1e-10):
    """Why a cell is in the pre-repair census. Floor is the EOS default."""
    idx = meta["index"]
    upd = arr["U"] + arr["DU"]
    sl = interior_slice(meta)
    icy = idx["ICY"]
    ny = upd.shape[0] - icy
    idn = upd[idx["IDN"]][sl]
    species = upd[icy : icy + ny][(slice(None),) + sl]
    neg = np.any(species < 0.0, axis=0)
    low = idn < floor
    return {
        "idn_only": int(np.sum(low & ~neg)),
        "species_only": int(np.sum(neg & ~low)),
        "both": int(np.sum(low & neg)),
        "min_idn": float(idn.min()) if idn.size else 0.0,
        "min_species": float(species.min()) if species.size else 0.0,
    }


def export_faces(path, meta, on, ref, o3):
    """Full-slab face table. Theta is rebuilt; it is not read from the code."""
    idx = meta["index"]
    icy = idx["ICY"]
    ny = 2
    rows = ["IDN", "IVX", "IVY", "IVZ", "IPR", "ICY", "ICY+1"]
    row_index = {
        "IDN": idx["IDN"],
        "IVX": idx["IVX"],
        "IVY": idx["IVY"],
        "IVZ": idx["IVZ"],
        "IPR": idx["IPR"],
        "ICY": icy,
        "ICY+1": icy + 1,
    }
    rho_s = ref["U"][icy : icy + ny]
    vol = ref["VOL"]
    lines = []
    lines.append(
        "# eos %s case %s arm %s mutation %s"
        % (meta.get("eos"), meta.get("case"), meta.get("arm"), meta.get("mutation"))
    )
    lines.append(
        "# use_nasa9_cp %s use_h2_cp %s Tref %s Rgas %s"
        % (meta.get("use_nasa9_cp"), meta.get("use_h2_cp"), g(meta["Tref"]), g(meta["Rgas"]))
    )
    for sp in meta["species"]:
        lines.append(
            "# species %d %s M_code %s M_yaml %s cv_R %s u0_R %s cloud %d"
            % (
                sp["i"],
                sp["name"],
                g(sp["M_code"]),
                g(sp["M_yaml"]),
                g(sp["cv_R"]),
                g(sp["u0_R"]),
                sp["cloud"],
            )
        )
    lines.append(
        "# F_ref is the B2 flux (limiter on, flux cut skipped). "
        "F_on is this arm. theta_s is rebuilt from rho_s, V, dt=1, "
        "outgoing area-weighted F_ref, margin 1-4096eps."
    )
    lines.append(
        "# j,i on a row are the left cell of that face (face-1 along the flux axis)."
    )
    lines.append(
        "axis,k,j,i,face,row,F_ref,F_on,fsed1,theta_s,hook_h,T_donor"
    )
    k0, k1 = meta["kl"], meta["ku"]
    j0, j1 = meta["jl"], meta["ju"]
    i0, i1 = meta["il"], meta["iu"]

    def take(flux, s, k, j, i, dim, face):
        if dim == "x1":
            return flux[s, k, j, face]
        return flux[s, k, face, i]

    for dim, flux_name, area_name, lo_key, hi_key, axis in (
        ("x1", "FLUX1", "AREA1", "il", "iu", 2),
        ("x2", "FLUX2", "AREA2", "jl", "ju", 1),
    ):
        flux_r = ref[flux_name]
        flux_o = on[flux_name]
        area = ref[area_name]
        if flux_r is None or area is None or flux_o is None:
            continue
        lo, hi = meta[lo_key], meta[hi_key]
        fsed = ref["FSED1"] if dim == "x1" else None
        outs = [
            accumulate_out(rho_s[s], flux_r[icy + s], area, lo, hi, axis)
            for s in range(ny)
        ]
        th = theta_of(rho_s, vol, np.stack(outs, axis=0))
        for f in range(lo, hi + 2):
            if dim == "x1":
                coords = [(k, j, i0) for k in range(k0, k1 + 1) for j in range(j0, j1 + 1)]
            else:
                coords = [(k, j0, i) for k in range(k0, k1 + 1) for i in range(i0, i1 + 1)]
            for k, j, i in coords:
                for row in rows:
                    srow = row_index[row]
                    Fref = float(take(flux_r, srow, k, j, i, dim, f))
                    Fon = float(take(flux_o, srow, k, j, i, dim, f))
                    if row.startswith("ICY"):
                        s = 0 if row == "ICY" else 1
                        sed = float(take(fsed, s, k, j, i, dim, f)) if fsed is not None else ""
                        Fsp = float(take(flux_r, icy + s, k, j, i, dim, f))
                        donor = (f - 1) if Fsp > 0.0 else f
                        if dim == "x1":
                            th_d = float(th[s, k, j, donor])
                            hh = float(on["H"][s, k, j, donor])
                            Td = float(o3["T"][k, j, donor])
                        else:
                            th_d = float(th[s, k, donor, i])
                            hh = float(on["H"][s, k, donor, i])
                            Td = float(o3["T"][k, donor, i])
                        sed_s = g(sed) if sed != "" else ""
                        extra = "%s,%s,%s,%s" % (sed_s, g(th_d), g(hh), g(Td))
                    else:
                        extra = ",,,"
                    # Left cell of this face. The transverse index is the real one.
                    rec_j = j if dim == "x1" else (f - 1)
                    rec_i = (f - 1) if dim == "x1" else i
                    lines.append(
                        "%s,%d,%d,%d,%d,%s,%s,%s,%s"
                        % (dim, k, rec_j, rec_i, f, row, g(Fref), g(Fon), extra)
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "study/236/gate2/raw")
    eos_list = ["ideal-moist", "moist-mixture"]
    lines = []
    clamp_lines = []
    any_fail = False
    any_void = False
    for eos in eos_list:
        for case, (direction, expect_n) in EXPECT.items():
            stem = raw / f"{eos}__{case}"
            summary = load_summary(Path(str(stem) + ".summary"))
            a_path = Path(str(stem) + "__a__m0.csv")
            b_path = Path(str(stem) + "__b2__m0.csv")
            if not a_path.is_file() or not b_path.is_file():
                lines.append(f"{eos} {case}: MISSING dump")
                any_fail = True
                continue
            ma, aa = load_dump(a_path)
            mb, ab = load_dump(b_path)
            if abs(float(ma["Rgas"]) - R_GAS) > 0.0 or abs(float(ma["kintera_Rgas"]) - R_GAS) > 0.0:
                lines.append(f"{eos} {case}: FAIL Rgas {ma.get('Rgas')}")
                any_fail = True
            if int(float(ma["use_nasa9_cp"])) or int(float(ma["use_h2_cp"])):
                lines.append(f"{eos} {case}: FAIL cp flag")
                any_fail = True
            o3 = o3_check(ma, aa)
            void = False
            reasons = []
            if int(summary.get("b2_eq_off", ma.get("b2_eq_off", 0))) != 1:
                void = True
                reasons.append(
                    f"B2 flux != limiter off (max abs {summary.get('max_abs_b2_off', ma.get('max_abs_b2_off'))})"
                )
            cens = summary.get("census_b2", [
                int(float(mb["nan_interior"])),
                int(float(mb["nan_ghost"])),
                int(float(mb["clamp_interior"])),
                int(float(mb["clamp_ghost"])),
            ])
            # Rescore: B2 is the unlimited reference, so a negative drained
            # cell does not void it. Valid means bitwise = limiter:false.
            if o3["M_void"]:
                void = True
                reasons.append(f"M rel {o3['M_rel']:.3e}")
            cens_a = summary.get("census_a", [
                int(float(ma["nan_interior"])),
                int(float(ma["nan_ghost"])),
                int(float(ma["clamp_interior"])),
                int(float(ma["clamp_ghost"])),
            ])
            try:
                counts, worst, nface = score_faces(ma, ab, aa, o3)
            except Exception as exc:
                lines.append(f"{eos} {case}: SCORE ERROR {exc}")
                any_fail = True
                continue
            col = column_over(ma, aa, ab)
            ca = clamp_split(ma, aa)
            cb = clamp_split(mb, ab)
            clamp_lines.append(
                f"{eos} {case} A idn_only {ca['idn_only']} species_only {ca['species_only']} "
                f"both {ca['both']} min_idn {ca['min_idn']:.6e} min_species {ca['min_species']:.6e} "
                f"| B2 idn_only {cb['idn_only']} species_only {cb['species_only']} "
                f"both {cb['both']} min_idn {cb['min_idn']:.6e} min_species {cb['min_species']:.6e} "
                f"repair_threw_b2 {summary.get('repair_threw_b2', 'n/a')} "
                f"repair_threw_a {summary.get('repair_threw_a', 'n/a')}"
            )
            face_dir = raw.parent / "faces"
            export_faces(face_dir / f"{eos}__{case}__m0.csv", ma, aa, ab, o3)
            i2_bit = int(summary.get("i2_neighbor_bitwise", ma.get("i2_neighbor_bitwise", -1)))
            i2_ratio = float(summary.get("i2_hke_over_s", ma.get("i2_hke_over_s", -1)))
            four_eps = 4.0 * EPS
            bars = {
                "O3h_v": o3["h_over"][0] <= 1.0,
                "O3h_c": o3["h_over"][1] <= 1.0,
                "T": o3["T_ok"],
                "I1": o3["i1_over"] <= 1.0,
                "I2_bit": i2_bit == 1,
                "I2_hke": 0.0 <= i2_ratio <= four_eps,
                "count": counts[direction] == expect_n,
                "IDN": int(summary.get("idn_a_b2", 0)) == 1,
                "Rs": worst["Rs"] <= 1.0,
                "RE": worst["RE"] <= 1.0,
                "RM": worst["RM"] <= 1.0,
                "column": col <= 1.0,
                # Species rows only (ICY..). An IDN undershoot is reported
                # beside main; it is not an arm A clamp.
                "A_census0": ca["species_only"] + ca["both"] == 0,
            }
            status = []
            for name, ok in bars.items():
                if name in ("Rs", "RE", "RM", "count", "IDN", "column") and void:
                    status.append(f"{name}=VOID")
                    any_void = True
                elif ok:
                    status.append(f"{name}=PASS")
                else:
                    status.append(f"{name}=FAIL")
                    any_fail = True
            if void:
                any_void = True
            mut_txt = ""
            if eos == "moist-mixture":
                mut_bits = []
                for m in range(1, 6):
                    mp = Path(str(stem) + f"__a__m{m}.csv")
                    if not mp.is_file():
                        mut_bits.append(f"m{m}=MISSING")
                        any_fail = True
                        continue
                    mm, am = load_dump(mp)
                    h_over = mutation_over(mm, am, o3, ab)
                    try:
                        # score mutated A against the same B2 reference and O3
                        mo = o3_check(mm, am)
                        _, mw, _ = score_faces(mm, ab, am, mo)
                    except Exception as exc:
                        mut_bits.append(f"m{m}=ERR")
                        any_fail = True
                        continue
                    red_h = h_over >= 1e3
                    red_e = mw["RE"] >= 1e3
                    ok = red_h and red_e
                    mut_bits.append(
                        f"m{m}={'RED' if ok else 'NOTRED'} h={h_over:.3e} RE={mw['RE']:.3e}"
                    )
                    if not ok:
                        any_fail = True
                    export_faces(
                        (raw.parent / "faces") / f"{eos}__{case}__m{m}.csv",
                        mm,
                        am,
                        ab,
                        mo,
                    )
                mut_txt = " " + " ".join(mut_bits)
            lines.append(
                f"{eos} {case}: faces {counts} expect {direction} {expect_n} "
                f"nface {nface} over Rs {worst['Rs']:.3e} RE {worst['RE']:.3e} "
                f"RM {worst['RM']:.3e} col {col:.3e} "
                f"h {o3['h_over']} i1 {o3['i1_over']:.3e} "
                f"Tmax {o3['T_max']:.3e} i2 {i2_bit} {i2_ratio:.3e} "
                f"M {o3['M_rel']:.3e} censusA {cens_a} censusB2 {cens} "
                + ("VOID " + "; ".join(reasons) if void else "ref-ok")
                + " | "
                + " ".join(status)
                + mut_txt
            )
    text = "\n".join(lines) + "\n"
    out_path = raw.parent / "score_lines.txt"
    out_path.write_text(text)
    (raw.parent / "clamp_split.txt").write_text("\n".join(clamp_lines) + "\n")
    print(text)
    if any_fail or any_void:
        print("GATE FAIL" if any_fail else "GATE VOID")
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
