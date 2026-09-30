"""Identity of the routed network, so a state file cannot be loaded onto another one.

Waterbody ids (``arange`` over the routable lake set) and routing link ids (one ``arange``
over the long links in row order, ``nhf_discretize._discretize_links``) are positional. A
changed lake set or discretization leaves an id present and naming another segment, which
a presence or coverage check passes.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping

import pandas as pd

FINGERPRINT_KEY = "network_fingerprint"
_LINK_IDENTITY_COLS = ("fp_id", "segment_order", "downstream")


def _canonical(value: object) -> str:
    """Render a value so the hash ignores its dtype.

    ``fp_id`` floats once any row is null, so one network can present it as int64 or
    float64, and hashing the repr would refuse that network's own restarts.
    """
    if value is None or value is pd.NA:
        return "~"
    if isinstance(value, float):
        if math.isnan(value):
            return "~"
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def network_fingerprint(
    dataframe: pd.DataFrame,
    waterbody_dataframe: pd.DataFrame,
    record_id_field: str,
    connections: Mapping[int, Iterable[int]],
) -> str:
    """Hash which links exist, what each one is, and how the routing graph joins them.

    Each link id is hashed with its ``fp_id``, ``segment_order`` and ``downstream``, so a
    relabeled id changes the hash. The graph comes from ``connections``, the only record
    of a reservoir's inlets and outlet, and each lake is its hydrofabric id paired with
    its routed id.
    """
    h = hashlib.sha256()
    h.update(b"links\n")
    if not dataframe.empty:
        cols = [c for c in _LINK_IDENTITY_COLS if c in dataframe.columns]
        ident = dataframe[cols].copy() if cols else pd.DataFrame(index=dataframe.index)
        ident.insert(0, "_id", dataframe.index.to_numpy())
        for row in ident.sort_values("_id").itertuples(index=False, name=None):
            h.update("|".join(_canonical(v) for v in row).encode())
            h.update(b"\n")
    h.update(b"edges\n")
    for node in sorted(connections):
        downstream = ",".join(_canonical(d) for d in sorted(connections[node]))
        h.update(f"{_canonical(node)}|{downstream}\n".encode())
    h.update(b"lakes\n")
    if not waterbody_dataframe.empty and record_id_field in waterbody_dataframe.columns:
        ids = pd.to_numeric(waterbody_dataframe[record_id_field], errors="coerce")
        # A lake demoted to a channel after allocation leaves its slot empty, so one
        # lake set can carry two allocations.
        pairs = sorted(
            (int(og), _canonical(routed))
            for routed, og in zip(waterbody_dataframe.index, ids, strict=True)
            if pd.notna(og)
        )
        for og, routed in pairs:
            h.update(f"{og}|{routed}\n".encode())
    return h.hexdigest()


def check_fingerprint(saved: str | None, current: str, source: str) -> None:
    """Refuse a state file built on a different network.

    ``None`` means the file predates the stamp, so its keys cannot be verified at all.
    """
    if saved == current:
        return
    if saved is None:
        msg = (
            f"{source} carries no {FINGERPRINT_KEY}, so the network it was written for "
            "cannot be verified. Waterbody and routing link ids are positional, so a "
            "state file from a different lake set or discretization loads onto the "
            "wrong segments without error. Start this configuration cold once."
        )
        raise ValueError(msg)
    msg = (
        f"{source} was written for a different network ({saved[:12]}, this run is "
        f"{current[:12]}). Waterbody and routing link ids are positional, so its state "
        "would load onto the wrong segments. A changed lake set, discretization length "
        "or hydrofabric all do this. Start this configuration cold once."
    )
    raise ValueError(msg)
