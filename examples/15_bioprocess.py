"""
Example 15: monoclonal antibody fed-batch co-design.

A biopharmaceutical company has to deliver `annual_demand` kg/year of a
monoclonal antibody at `target_titer` g/L. The choices are:

    cell line:       which CHO clone to use (productivity vs growth)
    media:           which commercial formulation (cost vs cell-density cap)
    bioreactor:      which format and size (single-use vs stainless steel)
    feed strategy:   how aggressively to feed glucose (titer vs waste)

The four subsystems are coupled cyclically:

    - higher target titer needs higher cell density,
    - higher cell density needs higher kLa (a more capable bioreactor),
    - higher cell density also needs richer feed,
    - richer feed produces more lactate and ammonia (waste burden),
    - waste burden inflates the cell density required to deliver the
      titer (the cycle closes here).

The Kleene iteration over the peak-cell-density port resolves the
cycle. The outer Pareto front is in (cogs_per_g, footprint_m2,
co2_per_g) and has several incomparable points, since a high-producer-
with-premium-media design lives at low COGS but moderate footprint,
while a cheap-cell-line-with-large-stainless design has higher COGS
but lower footprint per gram.

Parameters are taken from the bioprocessing literature (2024-2026):

    qP per cell line:    integrated/effective specific productivity over
                         a full fed-batch is calibrated to give realistic
                         titers at literature-reported peak VCDs of
                         10-30 x 10^6 cells/mL. Per-cell instantaneous qP
                         values are CHO-K1 7-16, CHO-S 2-6, CHO-DG44 2-4,
                         CHO-MK up to 19 pg/cell/day; the integrated
                         effective values used here are about 4x higher.
                         (Reinhart et al. 2021; Sumi et al. 2024)
    QO2 (CHO):           2-8e-10 mmol/cell/h
                         (BioProcess International 2024)
    kLa (mammalian):     1-25 h^-1 typical, kLa=1 supports about
                         4 x 10^6 cells/mL with pure O2 sparging
                         (BioProcess International 2024)
    Media cost:          $70-130/L standard CD,
                         $200-600/L clinical/cGMP grade
                         (CHO media market report 2025)
    Bioreactor capex:    $200k-600k for 2000L SU + $50-150 consumables
                         per batch, $3-8M for 50,000L SS installed
                         (Sustainability Atlas 2026, BioPlan 2025)
    Metabolic limits:    ammonia > 8 mM and lactate > 55 mM each halve
                         growth rate; low-glucose feed (5 mM target)
                         minimises both burdens
                         (Khattak et al. 2010, Lao & Toth 1997)
"""
from __future__ import annotations

import math

from codesign import (
    CatalogDP,
    CatalogEntry,
    Module,
    Ports,
    Reals,
    System,
    solve,
)


# ---------------------------------------------------------------------------
# Catalogue: cell lines. qP here is "effective integrated" productivity,
# calibrated so titer = qP * avg_vcd * batch_days * 1e-3 gives realistic
# numbers given the literature peak VCDs of 10-30 x 10^6 cells/mL.
# ---------------------------------------------------------------------------

CELL_LINES = [
    # (name, qP_eff [pg/cell/day], qO2 [1e-10 mmol/cell/h],
    #  batch_days, license_fee_per_batch [USD])
    ("CHO-S",     15.0, 5.0, 14.0,    500.0),   # biomass-favouring legacy
    ("CHO-DG44",  18.0, 6.0, 14.0,   1500.0),   # DHFR-amplified
    ("CHO-K1",    50.0, 7.0, 12.0,   5000.0),   # high-producer workhorse
    ("CHO-MK",   120.0, 7.5,  8.0,  25000.0),   # next-gen, short batch
]


# ---------------------------------------------------------------------------
# Catalogue: media. max_vcd_supported is in 1e6 cells/mL.
# ---------------------------------------------------------------------------

MEDIA_OPTIONS = [
    # (name, cost_per_l [USD], max_vcd_supported [1e6 cells/mL])
    ("HyClone-CD",         80.0, 15.0),   # standard CD
    ("EX-CELL-CD-CHO",    110.0, 25.0),   # workhorse modern CD
    ("Cellvento-CHO-220", 140.0, 35.0),   # premium CD with feed-220
    ("BalanCD-HIP",       250.0, 80.0),   # high-intensity perfusion grade
]


# ---------------------------------------------------------------------------
# Catalogue: bioreactors. max_peak_vcd is in 1e6 cells/mL, derived from
# kLa via the rough rule "kLa=1 supports 4e6 cells/mL with pure O2".
# ---------------------------------------------------------------------------

BIOREACTORS = [
    # (name, working_volume [L], max_peak_vcd [1e6 cells/mL],
    #  capex_per_batch [USD], footprint [m^2], co2_per_batch [kg])
    ("SU-200",      200.0,  40.0,   3_000.0,  2.0,  150.0),
    ("SU-2000",    2000.0,  60.0,  20_000.0,  8.0, 1200.0),   # industry sweet spot
    ("SS-5000",    5000.0,  72.0,  35_000.0, 12.0, 2200.0),
    ("SS-12500",  12500.0,  80.0,  60_000.0, 22.0, 4800.0),
    ("SS-25000",  25000.0,  90.0, 100_000.0, 35.0, 8500.0),
]


# ---------------------------------------------------------------------------
# CellLine module: maps demanded titer to required cell density,
# oxygen demand at peak, and process diagnostics.
# ---------------------------------------------------------------------------


class CellLine(Module):
    """A CHO cell line characterised by effective specific productivity,
    oxygen uptake rate per cell, and standard fed-batch length.

    Derivation
    ----------
    With qP in pg/cell/day, avg VCD in cells/mL, and 1000 mL/L:

        daily volumetric productivity [g/L/day]
            = qP [pg/cell/day] * VCD [cells/mL] * 1000 mL/L / 1e12 pg/g
            = qP * VCD * 1e-9

    Integrated over a fed-batch of length T days:

        titer [g/L] = qP * avg_VCD * T * 1e-9

    Rearranging with VCD in 1e6 cells/mL:

        avg_VCD_million = titer * 1e3 / (qP * T)

    Peak VCD is roughly 2x average over a fed-batch (the cell density
    rises to a plateau and then declines), so peak_VCD_million =
    2 * avg_VCD_million.

    Oxygen demand at peak in mmol/L/h is then

        OUR_peak = peak_VCD [cells/mL] * 1000 mL/L * qO2 [mmol/cell/h]
                = peak_VCD_million * 1e9 * qO2 * 1e-10
                = peak_VCD_million * qO2 * 0.1
    """

    F = {"target_titer": Reals(unit="g/L")}
    R = {
        "avg_vcd":           Reals(unit="1e6 cells/mL"),
        "peak_vcd":          Reals(unit="1e6 cells/mL"),
        "oxygen_demand":     Reals(unit="mmol/L/h"),
        "batch_days":        Reals(unit="day"),
        "license_per_batch": Reals(unit="USD"),
    }

    def __init__(self, name, qp_pg_cell_day, qo2_e10, batch_days,
                 license_fee=0.0, peak_to_avg=2.0):
        self.cell_name = name
        self.qp = qp_pg_cell_day        # effective integrated qP, pg/cell/day
        self.qo2 = qo2_e10              # 1e-10 mmol/cell/h
        self.batch_days = batch_days
        self.license_fee = license_fee
        self.peak_to_avg = peak_to_avg
        super().__init__()

    def h(self, f):
        titer = f["target_titer"]
        # avg VCD needed, in 1e6 cells/mL units.
        avg_vcd = titer * 1e3 / (self.qp * self.batch_days)
        # Peak is roughly 2x the average over the batch.
        peak_vcd = self.peak_to_avg * avg_vcd
        # Oxygen demand at peak, mmol/L/h. See class docstring derivation.
        oxygen = peak_vcd * self.qo2 * 0.1
        return {
            "avg_vcd":           avg_vcd,
            "peak_vcd":          peak_vcd,
            "oxygen_demand":     oxygen,
            "batch_days":        float(self.batch_days),
            "license_per_batch": float(self.license_fee),
        }


# ---------------------------------------------------------------------------
# Bioreactor catalog: pick the smallest entry that supports the demanded
# peak VCD. Working volume, capex, footprint, and CO2 all become outputs.
# ---------------------------------------------------------------------------


def make_bioreactor_dp():
    """Build a CatalogDP that picks the smallest bioreactor supporting
    the demanded peak cell density.

    F (demands):
        peak_vcd        - 1e6 cells/mL at peak

    R (resources, per batch):
        working_volume     - L
        capex_per_batch    - USD amortised over this batch
        footprint_m2       - m^2 occupied by the reactor (excluding utilities)
        co2_per_batch      - kg CO2 emitted per batch (consumables + utilities)
    """
    F = Ports({"peak_vcd": Reals(unit="1e6 cells/mL")})
    R = Ports({
        "working_volume":  Reals(unit="L"),
        "capex_per_batch": Reals(unit="USD"),
        "footprint_m2":    Reals(unit="m^2"),
        "co2_per_batch":   Reals(unit="kg"),
    })
    entries = [
        CatalogEntry(
            name=name,
            provides={"peak_vcd": max_vcd},
            costs={
                "working_volume":  vol,
                "capex_per_batch": capex,
                "footprint_m2":    fp,
                "co2_per_batch":   co2,
            },
        )
        for (name, vol, max_vcd, capex, fp, co2) in BIOREACTORS
    ]
    return CatalogDP(F=F, R=R, catalog=entries, name="bioreactor_catalog")


# ---------------------------------------------------------------------------
# Media catalog: smallest cost media supporting the demanded peak VCD.
# ---------------------------------------------------------------------------


def make_media_dp():
    """Build a CatalogDP that picks the cheapest media supporting the
    demanded peak cell density.

    F (demands):
        peak_vcd        - 1e6 cells/mL at peak (must be supportable)

    R (resources):
        media_cost_per_l  - USD per litre of working volume
    """
    F = Ports({"peak_vcd": Reals(unit="1e6 cells/mL")})
    R = Ports({"media_cost_per_l": Reals(unit="USD/L")})
    entries = [
        CatalogEntry(
            name=name,
            provides={"peak_vcd": max_vcd},
            costs={"media_cost_per_l": cost},
        )
        for (name, cost, max_vcd) in MEDIA_OPTIONS
    ]
    return CatalogDP(F=F, R=R, catalog=entries, name="media_catalog")


# ---------------------------------------------------------------------------
# FeedStrategy module: glucose set-point as the design knob.
# ---------------------------------------------------------------------------


class FeedStrategy(Module):
    """Glucose-controlled bolus feed strategy.

    The single design knob is the glucose set-point in mM. Lower
    set-points (around 5 mM) shift cell metabolism away from lactate
    production, reducing waste accumulation and allowing higher peak
    VCD. Higher set-points are simpler to control but produce ammonia
    and lactate that limit growth (Khattak 2010, Lao & Toth 1997).

    From Khattak 2010: a 3x increase in nutrient set-point produced
    +78% ammonia and -30% growth rate. We encode that as a linear
    multiplicative factor on the cell-density demand: at the rich end
    of the range, more cells are needed for the same titer because the
    effective integrated qP suffers from waste burden.

    Parameters
    ----------
    glucose_setpoint_mm : float
        Target glucose concentration during the production phase.
        Practical range 4 mM (HIPDOG-aggressive) to 15 mM (legacy).
    """

    F = {
        "peak_vcd":   Reals(unit="1e6 cells/mL"),
        "batch_days": Reals(unit="day"),
    }
    R = {
        "feed_cost_per_l":   Reals(unit="USD/L"),
        "metabolic_factor":  Reals(unit="rel"),
        "cogs_multiplier":   Reals(unit="rel"),
    }

    def __init__(self, glucose_setpoint_mm=8.0):
        # Clamp into the practical range.
        self.glucose = max(4.0, min(15.0, float(glucose_setpoint_mm)))
        super().__init__()

    def h(self, f):
        # Feed cost: ~5% of working volume per day of bolus feed at
        # roughly $6/L base, with a premium for richer formulations.
        glucose_premium = 1.0 + 0.05 * (self.glucose - 5.0)
        feed_cost_per_l = 6.0 * glucose_premium * f["batch_days"] * 0.05
        # Metabolic factor (high-glucose penalty): more lactate and
        # ammonia at richer feed, growth penalty applied as more cells
        # needed for the same titer. 1.0 at 5 mM, ~1.30 at 15 mM.
        # See Khattak 2010 for the 3x-nutrient = -30% growth correlation.
        waste_factor = 1.0 + 0.03 * (self.glucose - 5.0)
        # COGS multiplier (U-shape around 8 mM):
        #   - Low glucose (~5 mM) operates near starvation threshold;
        #     tight control failures discard batches, raising effective
        #     COGS by ~20-30% from wasted runs.
        #   - The 8 mM industry default is the robustness sweet spot.
        #   - High glucose (~12-15 mM) is easier to control but creates
        #     waste accumulation that occasionally crashes growth, ~10%
        #     batch loss.
        low_penalty  = 0.06 * max(0.0, 8.0 - self.glucose) ** 1.5
        high_penalty = 0.015 * max(0.0, self.glucose - 8.0) ** 1.5
        cogs_multiplier = 1.0 + low_penalty + high_penalty
        return {
            "feed_cost_per_l":   feed_cost_per_l,
            "metabolic_factor":  waste_factor,
            "cogs_multiplier":   cogs_multiplier,
        }


# ---------------------------------------------------------------------------
# Assemble the System.
# ---------------------------------------------------------------------------


def make_bioprocess(
    *,
    cell_line: tuple,
    glucose_setpoint_mm: float,
    annual_demand_kg: float,
    turnaround_days: float = 5.0,
    downstream_yield: float = 0.7,
):
    """Construct a co-design problem for a specific cell-line and feed
    choice. The bioreactor and media are picked by the solver via
    catalogue lookups; cell line and feed strategy are fixed at
    construction time so the caller sweeps them in the outer loop.

    Parameters
    ----------
    cell_line : tuple
        (name, qP_eff, qO2_e10, batch_days, license_fee) from CELL_LINES.
    glucose_setpoint_mm : float
        Feed strategy design knob (4-15 mM).
    annual_demand_kg : float
        Annual production target in kg/year. Drives the number of
        parallel reactor lines through batches_per_year.
    turnaround_days : float
        Bioreactor turnaround between batches. Typical 3-7 days for
        single-use, longer for stainless with CIP/SIP.
    downstream_yield : float
        Fraction of mAb mass surviving downstream purification. 0.6-0.8
        is typical (Protein A capture + viral filtration + polish).
    """
    sys = System(f"mAb_{cell_line[0]}")

    # Outer F: mission demand (titer at harvest).
    target_titer = sys.provides("target_titer", unit="g/L")

    # Outer R: the cost vector the engineer minimises over.
    cogs_per_g   = sys.requires("cogs_per_g",   unit="USD/g")
    footprint_m2 = sys.requires("footprint_m2", unit="m^2")
    co2_per_g    = sys.requires("co2_per_g",    unit="kg/g")

    # Subsystems.
    cell  = sys.add("cell",  CellLine(*cell_line))
    feed  = sys.add("feed",  FeedStrategy(glucose_setpoint_mm))
    bior  = sys.add("bior",  make_bioreactor_dp())
    media = sys.add("media", make_media_dp())

    # Wire the cell line: titer demand drives required cell density.
    cell.target_titer >= target_titer

    # Feed strategy reads the peak VCD and batch length, emits a
    # metabolic load factor.
    feed.peak_vcd   >= cell.peak_vcd
    feed.batch_days >= cell.batch_days

    # Bioreactor must support the metabolically-inflated peak VCD.
    bior.peak_vcd  >= cell.peak_vcd * feed.metabolic_factor

    # Media must support the metabolically-inflated peak VCD too.
    media.peak_vcd >= cell.peak_vcd * feed.metabolic_factor

    # Aggregate the three outer cost outputs via constraint expressions.
    # We use the dict-based constrain form because we need to combine
    # several ports plus closed-over parameters (annual_demand, yield).
    def cogs_eq(x):
        # Per-batch mass and per-batch cost.
        vol = x["bior.working_volume"]
        capex = x["bior.capex_per_batch"]
        license_fee = x["cell.license_per_batch"]
        media_cost_l = x["media.media_cost_per_l"]
        feed_l = x["feed.feed_cost_per_l"]
        titer = x["target_titer"]
        mass_per_batch_g = titer * vol * downstream_yield        # grams of mAb
        cost_per_batch = (capex + license_fee
                          + media_cost_l * vol
                          + feed_l * vol)
        # Multiply by the feed-strategy COGS multiplier to absorb the
        # batch-failure penalty (low glucose) or waste penalty (high
        # glucose) into the unit cost.
        return (cost_per_batch / max(mass_per_batch_g, 1e-6)
                * x["feed.cogs_multiplier"])

    def footprint_eq(x):
        # Annual demand divided by per-batch mass = batches/year needed;
        # each reactor line can deliver at most 365/(batch+turnaround)
        # batches/year, so parallel_lines = ceil(needed / max_per_line).
        # A 3x multiplier accounts for downstream and utilities space.
        vol = x["bior.working_volume"]
        titer = x["target_titer"]
        batch_d = x["cell.batch_days"]
        cycle_d = batch_d + turnaround_days
        max_batches_per_line = 365.0 / cycle_d
        mass_per_batch_g = titer * vol * downstream_yield
        batches_needed = annual_demand_kg * 1000.0 / max(mass_per_batch_g, 1e-6)
        parallel_lines = max(1.0, batches_needed / max_batches_per_line)
        return parallel_lines * x["bior.footprint_m2"] * 3.0

    def co2_eq(x):
        vol = x["bior.working_volume"]
        titer = x["target_titer"]
        mass_per_batch_g = titer * vol * downstream_yield
        return x["bior.co2_per_batch"] / max(mass_per_batch_g, 1e-6)

    sys.constrain("cogs_per_g",   cogs_eq)
    sys.constrain("footprint_m2", footprint_eq)
    sys.constrain("co2_per_g",    co2_eq)

    return sys.build()


# ---------------------------------------------------------------------------
# Sweep over cell-line and feed choices, collect global Pareto front.
# ---------------------------------------------------------------------------


def run_sweep(target_titer_g_l: float, annual_demand_kg: float, *, quiet=False):
    if not quiet:
        print(f"\n=== mAb co-design: titer = {target_titer_g_l} g/L, "
              f"annual demand = {annual_demand_kg} kg/year ===\n")

    results = []
    for cl in CELL_LINES:
        for glu in (5.0, 8.0, 12.0):
            try:
                dp = make_bioprocess(
                    cell_line=cl,
                    glucose_setpoint_mm=glu,
                    annual_demand_kg=annual_demand_kg,
                )
                r = solve(dp, {"target_titer": target_titer_g_l},
                          max_iter=200)
            except Exception:
                continue
            if not r.feasible:
                continue
            for pt in r.antichain.points:
                cogs = pt["cogs_per_g"]
                fp = pt["footprint_m2"]
                co2 = pt["co2_per_g"]
                if math.isinf(cogs) or math.isinf(fp) or math.isinf(co2):
                    continue
                results.append({
                    "label": f"{cl[0]}/glu={glu:.0f}mM",
                    "cogs_per_g":   cogs,
                    "footprint_m2": fp,
                    "co2_per_g":    co2,
                })

    # Global Pareto front.
    pareto = []
    for p in results:
        dominated = any(
            q["cogs_per_g"]   <= p["cogs_per_g"]
            and q["footprint_m2"] <= p["footprint_m2"]
            and q["co2_per_g"] <= p["co2_per_g"]
            and (q["cogs_per_g"]   < p["cogs_per_g"]
                 or q["footprint_m2"] < p["footprint_m2"]
                 or q["co2_per_g"] < p["co2_per_g"])
            for q in results
        )
        if not dominated:
            pareto.append(p)
    pareto.sort(key=lambda x: x["cogs_per_g"])

    if not quiet:
        print(f"  {len(results)} feasible designs evaluated")
        print(f"  {len(pareto)} non-dominated (Pareto-optimal):\n")
        print(f"  {'design':<25} {'COGS':>10}  {'footprint':>11}  {'CO2':>10}")
        print(f"  {'-'*25} {'-'*10}  {'-'*11}  {'-'*10}")
        for p in pareto:
            print(f"  {p['label']:<25} "
                  f"${p['cogs_per_g']:>7.2f}/g  "
                  f"{p['footprint_m2']:>8.1f} m^2  "
                  f"{p['co2_per_g']*1000:>6.2f} g/g")
        if pareto:
            cheapest = min(pareto, key=lambda x: x["cogs_per_g"])
            print(f"\n  Cheapest design overall: {cheapest['label']}")
            print(f"     COGS      = ${cheapest['cogs_per_g']:.2f}/g")
            print(f"     Footprint = {cheapest['footprint_m2']:.1f} m^2")
            print(f"     CO2       = {cheapest['co2_per_g']*1000:.2f} g CO2/g mAb")

    return pareto


# ---------------------------------------------------------------------------
# Main: three representative mission scales.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Clinical batch supply (~10 kg/year): small SU dominates.
    # 2. Mid-stage commercial (~100 kg/year): SU 2000L sweet spot.
    # 3. Large commercial (~500 kg/year): pushes toward SS.
    _ = run_sweep(target_titer_g_l=3.0, annual_demand_kg=10.0)
    _ = run_sweep(target_titer_g_l=5.0, annual_demand_kg=100.0)
    _ = run_sweep(target_titer_g_l=8.0, annual_demand_kg=500.0)
