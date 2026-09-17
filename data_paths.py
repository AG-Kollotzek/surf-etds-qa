"""Where the raw measurement data live: the surf-etds-data submodule.

Every campaign of a linac is a folder surf-etds-data/campaigns/<YYYY-MM-DD>_L<n>/ with the
terminal files in phantom/ and the ExacTrac exports in etd/. A time stamp from
etds_qa_2026_config.json identifies one file; it is searched across all campaigns of the
linac and must match exactly once.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CAMPAIGNS_DIR = PROJECT_DIR / "surf-etds-data" / "campaigns"


def campaign_dirs(linac):
    """All campaign folders of one linac, oldest first."""
    return sorted(p for p in CAMPAIGNS_DIR.glob(f"*_L{linac}") if p.is_dir())


def _find_one(linac, subdir, pattern, keep=lambda p: True):
    if not CAMPAIGNS_DIR.is_dir():
        raise FileNotFoundError(f"{CAMPAIGNS_DIR} is missing; run 'git submodule update --init'.")
    hits = [p for d in campaign_dirs(linac) for p in sorted((d / subdir).glob(pattern)) if keep(p)]
    if len(hits) != 1:
        found = ", ".join(str(p.relative_to(CAMPAIGNS_DIR)) for p in hits) or "none"
        raise FileNotFoundError(f"expected exactly one file {subdir}/{pattern} for linac {linac}, found: {found}")
    return hits[0]


def etd_stamp_dashed(stamp):
    """161959 -> 16-19-59 (as in the export file names); dashed stamps pass unchanged."""
    stamp = str(stamp)
    return f"{stamp[:2]}-{stamp[2:4]}-{stamp[4:]}" if len(stamp) == 6 and "-" not in stamp else stamp


def surf_csv(linac, surf_stamp):
    """Terminal CSV of a measurement (never the sphere-detection *_QA.csv)."""
    return _find_one(linac, "phantom", f"*{surf_stamp}.csv", keep=lambda p: not p.name.endswith("_QA.csv"))


def surf_qa_csv(linac, surf_stamp):
    """Sphere-detection file *_<stamp>_QA.csv of a measurement."""
    return _find_one(linac, "phantom", f"*{surf_stamp}_QA.csv")


def etd_json(linac, etds_stamp):
    """ExacTrac export TrackingResult_<date>_<HH-MM-SS>.json of a measurement."""
    return _find_one(linac, "etd", f"TrackingResult_*_{etd_stamp_dashed(etds_stamp)}.json")
