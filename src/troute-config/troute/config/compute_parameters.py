import logging
import math

from pydantic import BaseModel, DirectoryPath, FilePath, Field, field_validator, model_validator, ConfigDict
from datetime import datetime

from typing import Annotated, Any, Dict, Optional, List, Union
from typing_extensions import Literal, Self

from ._validators import coerce_datetime

LOG = logging.getLogger("TROUTE")


# ---------------------------- Compute Parameters ---------------------------- #

ParallelComputeMethod = Literal[
    "serial",
    "by-network",
    "by-subnetwork-jit",
    "by-subnetwork-jit-clustered",
    "by-subnetwork-diffusive",
    "bmi",
]

ComputeKernel = Literal["V02-structured", "diffusive", "diffusice_cnt"]


# TODO: determine how to handle context specific required fields
# TODO: consider other ways to handle wrf hydro fields (i.e. subclass)
class RestartParameters(BaseModel):
    """
    Parameters specifying warm-state simulation conditions.
    """
    start_datetime: Optional[datetime] = None
    """
    Time of model initialization (timestep zero). Datetime format should be %Y-%m-%d_%H:%M, e.g., 2023-04-25_00:00
    This start time will control which forcing files and TimeSlice files are required for the simulation. 
    If the start time is erroneously enertered, such that there are no available forcing files, then the simulation will fail. 
    Likewise, if there are no TimeSlice files available, then data assimilation will not occur.
    NOTE: The default is 'None' because the start date can be determined from restart files
    such as 'lite_channel_restart_file' or 'wrf_hydro_channel_restart_file'. But if no restart
    file is provided, this parameter is required.
    """
    lite_channel_restart_file: Optional[FilePath] = None
    """
    Filepath to a 'lite' channel restart file create by a previous t-route simulation. If a file is specified, then it will be 
    given preference over WRF restart files for a simulation restart.
    """
    lite_waterbody_restart_file: Optional[FilePath] = None
    """
    Filepath to a 'lite' waterbody restart file create by a previous t-route simulation. If a file is specified, then it will be 
    given preference over WRF restart files for a simulation restart.
    """

    wrf_hydro_channel_restart_file: Optional[FilePath] = None
    """
    Filepath to WRF Hydro HYDRO_RST file. This file does not need to be timed with start_datetime, which allows initial states
    from one datetime to initialize a simulation with forcings starting at a different datetime. However, if the start_datetime 
    parameter is not specified, then the time attribute in the channel restart file will be used as the starting time of the simulation.
    """
    wrf_hydro_channel_ID_crosswalk_file: Optional[FilePath] = None
    """
    Filepath to channel geometry file.
    NOTE: if `wrf_hydro_channel_restart_file` is given, `wrf_hydro_channel_ID_crosswalk_file` is required
    """
    wrf_hydro_channel_ID_crosswalk_file_field_name: Optional[str] = None
    """
    Field name of segment IDs in restart file.
    """
    wrf_hydro_channel_restart_upstream_flow_field_name: Optional[str] = None
    """
    Field name of upstream flow in restart file.
    """
    wrf_hydro_channel_restart_downstream_flow_field_name: Optional[str] = None
    """
    Field name of downstream flow in restart file.
    """
    wrf_hydro_channel_restart_depth_flow_field_name: Optional[str] = None
    """
    Field name of depth in restart file.
    """

    wrf_hydro_waterbody_restart_file: Optional[FilePath] = None
    """
    Filepath to waterbody restart file. This is often the same as wrf_hydro_channel_restart_file.
    """
    wrf_hydro_waterbody_ID_crosswalk_file: Optional[FilePath] = None
    """
    Filepath to lake parameter file.
    NOTE: required if `wrf_hydro_waterbody_restart_file`
    """
    wrf_hydro_waterbody_ID_crosswalk_file_field_name: Optional[str] = None
    """
    Field name of waterbody ID.
    """
    wrf_hydro_waterbody_crosswalk_filter_file: Optional[FilePath] = None
    """
    Filepath to channel geometry file.
    """
    wrf_hydro_waterbody_crosswalk_filter_file_field_name: Optional[str] = None
    """
    Fieldname of waterbody IDs in channel geometry file.
    """
    

    _coerce_datetime = field_validator("start_datetime", mode="before")(coerce_datetime)


# TODO: determine how to handle context specific required fields
class HybridParameters(BaseModel):
    """
    Parameters controlling the use of MC/diffusive hybrid simulations. Only include/populate these parameters if an 
    MC/diffusive hybrid simulations is desired.
    """
    run_hybrid_routing: bool = False
    """
    Boolean parameter whether or not hybrid routing is actived. If it is set to True, the hybrid routing is activated. 
    If false, MC is solely used for channel flow routing.
    NOTE: required for hybrid simulations
    """
    diffusive_domain: Optional[FilePath] = None
    """
    Filepath to diffusive domain dictionary file. This file can be either JSON or yaml and contain a dictionary
    of diffusive network segments, organized by tailwater ID (keys). This is a file such as: 
    https://github.com/NOAA-OWP/t-route/blob/master/test/LowerColorado_TX_v4/domain/coastal_domain_tw.yaml
    This file defines tailwater and head water flowpath IDs for the diffusive domain. See file for more info.
    NOTE: required for hybrid simulations
    """
    use_natl_xsections: bool = False
    """
    Boolean parameter whether or not natural cross section data is used. If it is set to True, diffusive model 
    uses natural cross section data. If False, diffusive model uses synthetic cross section defined by RouteLink.nc
    """
    topobathy_domain: Optional[FilePath] = None
    """
    Filepath to topobathy data for channel cross sections. Currently (June 25, 2024), 3D cross section data
    is contained in a separate file, which this parameter should point to. In the future this data may simply be
    included in the hydrofabric.
    Topobathy data of a channel cross section is defined by comid.
    NOTE: Required for diffusive routing for natural cross sections. 
    """
    run_refactored_network: bool = False
    """
    Boolean parameter whether or not to run the diffusive module on a refactored network. This was necessary on
    the NHD network due to short segments causing computational issues. Not needed for HYFeatures.
    """
    refactored_domain: Optional[FilePath] = None
    """
    A file with refactored flowpaths to eliminate short segments.
    NOTE: Only needed for NHD network. 
    """
    refactored_topobathy_domain: Optional[FilePath] = None
    """
    A file with refactored topobathy data.
    NOTE: Only needed for NHD network.
    """
    coastal_boundary_domain: Optional[FilePath] = None
    """
    File containing crosswalk between diffusive tailwater segment IDs and coastal model output node IDs. 
    This is needed if t-route will use outputs from a coastal model as the downstream boundary condition for
    the diffusive module. See example:
    https://github.com/NOAA-OWP/t-route/blob/master/test/LowerColorado_TX_v4/domain/coastal_domain_crosswalk.yaml
    NOTE: This is related to the ForcingParameters -> coastal_boundary_input_file parameter. 
    """


class QLateralForcingSet(BaseModel):
    """
    Forcing files and number of timesteps associated with each simulation loop. This is optional, only include if 
    explicitly listing the forcing files in each set. If this variable is not present, make sure nts, 
    qlat_file_pattern_filter, and max_loop_size variables are listed.
    NOTE: Using nts, qlat_input_folder, qlat_file_pattern_filter, and max_loop_size is the preferred method.
    """
    nts: "QLateralFiles"
    """
    Number of timesteps in loop iteration 1. This corresponds to the number of files listed in qlat_files.
    This parameter is repeated for as many iterations as are desired.
    """


class QLateralFiles(BaseModel):
    qlat_files: List[FilePath]
    """
    List of forcing file names to be used in a single iteration.
    """


class StreamflowDA(BaseModel):
    """
    Parameters controlling streamflow nudging DA
    """
    streamflow_nudging: bool = False
    """
    Boolean, determines whether or not streamflow nudging is performed.
    NOTE: Mandatory for streamflow DA
    """
    gage_segID_crosswalk_file: Optional[FilePath] = None
    """
    File relating stream gage IDs to segment links in the model domain. This is typically the RouteLink file.
    NOTE: Mandatory for streamflow DA on NHDNetwork. Not necessary on HYFeatures as this information is included
    in the hydrofabric.
    """
    crosswalk_gage_field: Optional[str] = 'gages'
    """
    Column name for gages in gage_segID_crosswalk_file.
    NOTE: Not necessary on HYFeatures.
    """
    crosswalk_segID_field: Optional[str] = 'link'
    """
    Column name for flowpaths/links in gage_segID_crosswalk_file.
    NOTE: Not necessary on HYFeatures.
    """
    lastobs_file: Optional[FilePath] = None
    """
    File containing information on the last streamflow observations that were assimilated from a previous t-route run. 
    This is used for a 'warm' restart. Mostly used for operational NWM settings.
    """
    diffusive_streamflow_nudging: bool = False
    """
    If True, enable streamflow data assimilation in diffusive module. 
    NOTE: Not yet implemented, leave as False. (June 25, 2024)
    """
    streamflow_scaling: bool = False
    """Enable the simple-scaling streamflow DA (NHF networks only): gage
    observations are injected into the MC kernel before routing (downstream
    propagation) and the recorded innovation is spread upstream post-routing,
    area-scaled over each gage's contributing tree. The correction enters the
    model state on the run's FINAL window only; only discharge is corrected, and
    the injected volume is an analysis increment, not mass-balanced. Mutually
    exclusive with ``streamflow_nudging``."""

    streamflow_scaling_parameters: "StreamflowScalingParams" = Field(
        default_factory=lambda: StreamflowScalingParams()
    )
    """Options for the simple-scaling DA; the defaults are a complete
    configuration, so the block is only needed to override them."""

    @model_validator(mode='after')
    def check_one_streamflow_da_method(self) -> Self:
        if self.streamflow_nudging and self.streamflow_scaling:
            raise ValueError(
                "streamflow_da.streamflow_nudging and "
                "streamflow_da.streamflow_scaling are mutually exclusive: both "
                "drive the same Muskingum-Cunge nudging override, and nudging's "
                "lastobs state would give the scaling arm cross-window decay "
                "continuity it is documented not to have."
            )
        return self


class ThetaRegionalization(BaseModel):
    """Region theta for the upstream area-scaling step (Ogden-Dawdy 2003).

    Theta must be UNIFORM within a gage tree: the linear step telescopes to the
    closed form dQ_o*(A_s/A_o)^theta only for a constant exponent, so one value per
    gage is the finest resolution the method admits. It is therefore resolved once
    per tree at build time, not carried per segment.

    Resolution order, most specific first: ``per_tree_csv`` (by gage id), then
    ``by_vpu`` (by the gage's hydrofabric VPU), then ``default``.

    Setting theta regionally is not optional detail. The source proposal derives
    0.77 from Ogden & Dawdy (2003) on a 0.3-21.2 km2 semi-humid watershed, and
    states that simple scaling "no longer holds" above roughly 50 km2, where the
    exponent becomes a function of drainage area and flood exceedance probability
    with regional values spanning 0.2 to 0.9. The delivered increment at a
    subarea scales as ``(A_s/A_o)^theta``, so relative to an applicable
    ``theta*`` the ratio of delivered to appropriate correction is
    ``(A_s/A_o)^(0.77-theta*)``: at a subarea 1% of the gage's area, 0.77
    delivers 7% of what theta*=0.2 implies and 1.8 times what theta*=0.9
    implies. The error grows with scale separation, which is why the exponent
    belongs to the operator's regional judgement rather than to a shipped
    constant.
    """
    model_config = ConfigDict(extra='forbid')

    default: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.77
    """Theta for any gage matched by neither ``per_tree`` nor ``by_vpu``. 0.77 is
    the Ogden-Dawdy value, fitted on a small semi-humid watershed; it is a
    regional calibration constant, not a segment-local constitutive law, and it
    is a fallback rather than a recommendation outside that setting."""

    per_tree_csv: str | None = None
    """Path to a CSV of per-gage-tree theta: two columns, gage id and theta.

    Ids are matched EXACTLY as they appear in the input TimeSlice files (the same
    ids used for ``holdout_sites_file``). A header is optional; ``gage_id,theta``
    is recognised, and a headerless two-column file is read positionally. One
    tree is one exponent, which is the finest resolution the closed form admits,
    so this is the most specific control the method can express. Wins over
    ``by_vpu``.

    A file rather than an inline mapping because a continental domain has
    thousands of gages, and because the exponent field is an INPUT an operator
    prepares from regional hydroclimatology, alongside the hydrofabric and the
    observations, rather than something hand-edited into a run config."""

    by_vpu: dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]] = {}
    """Per-VPU theta, keyed by the hydrofabric's ``vpu_id`` EXACTLY as it appears
    there (zero-padded, e.g. "01" not "1"). Unmatched keys are rejected rather than
    silently ignored, because a mistyped key would otherwise fall back to the
    default and look like it had been applied."""

    @field_validator("by_vpu")
    @classmethod
    def check_vpu_thetas(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject unusable theta values up front rather than mid-run.

        A bare ``> 0`` test lets NaN through (every comparison with NaN is False, so
        ``not (x > 0)`` is True but ``x <= 0`` is False), and a NaN theta silently
        turns the whole tree's correction into NaN.
        """
        for vpu, theta in v.items():
            if not vpu.strip():
                msg = f"streamflow_scaling_parameters.theta has an empty key: {vpu!r}"
                raise ValueError(msg)
            if not math.isfinite(theta) or theta <= 0:
                msg = (
                    f"streamflow_scaling_parameters.theta[{vpu!r}] must be a finite "
                    f"positive number, got {theta!r}"
                )
                raise ValueError(msg)
        return v


class StreamflowScalingParams(BaseModel):
    """
    Parameters controlling the simple-scaling streamflow DA: insert obs-minus-model
    at each gage and distribute the correction upstream (area-scaling along reaches,
    flow-ratio split at confluences). Applied per loop on NHF (-V5) runs.

    Both rules are always in play; which one applies at a given step is a property of
    the topology (a confluence splits by flow, a linear step scales by area), not a
    user choice.
    """
    # extra='forbid': a misspelled key here fails silently otherwise (e.g. a bad
    # holdout_sites_file assimilates the gages the run is scored against).
    model_config = ConfigDict(extra='forbid')

    theta: ThetaRegionalization = ThetaRegionalization()
    """Region theta, resolved once per gage tree: per_tree_csv, then by_vpu,
    then default."""

    synthetic_obs_factor: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    """If set, assimilate synthetic obs = factor * modeled at each gage (mechanics
    testing / end-to-end demo). Takes precedence over the TimeSlice observations."""

    synthetic_obs_baseline: str | None = None
    """In-kernel synthetic obs must be a FROZEN open-loop series (a no-DA run's
    output dir, feature_id=fp_id), not factor*live_model (which compounds under
    feedback). Required when synthetic_obs_factor is set (validated below); a
    missing or empty directory raises at first read rather than degrading to a
    silent no-assimilation run."""

    @model_validator(mode='after')
    def _synthetic_baseline_required(self) -> 'StreamflowScalingParams':
        if self.synthetic_obs_factor is not None and not self.synthetic_obs_baseline:
            raise ValueError(
                "streamflow_scaling_parameters.synthetic_obs_baseline is required "
                "when synthetic_obs_factor is set: synthetic observations must be "
                "factor * a FROZEN no-DA baseline, never factor * the live model."
            )
        return self

    holdout_sites_file: str | None = None
    """Text file (one site_no per line) of gages to EXCLUDE from injection, for
    held-out skill evaluation. Unknown ids are a hard error -- a typo that removes
    nothing would let the run assimilate the gages it claims to withhold."""

    min_flow_cms: float = Field(1e-6, gt=0, allow_inf_nan=False)
    """Floor (m^3/s) on the confluence flow-ratio split's denominator. A junction at
    or below this flow contributes no correction at that timestep, guarding the
    Q_branch/Q_parent division at near-dry junctions."""

    max_reach_km: float = Field(200.0, gt=0, allow_inf_nan=False)
    """How far upstream a gage's correction propagates, as distance ALONG the
    river network.

    Expressed in distance rather than travel time because reach length is known
    exactly while travel time is not. Every route to a defensible celerity was
    measured and rejected: a single constant cannot describe a continental
    network, and the router's own Courant number reports the KINEMATIC celerity,
    which runs faster than the wave peak where an event is passing and slower
    than any usable constant at ambient flow on the same reaches. Localizing an
    assimilation increment on network distance avoids estimating any of that;
    set the radius from the spacing of the gage network."""

    innovation_spread_h: float = Field(0.0, ge=0, allow_inf_nan=False)
    """Width, in hours, of the FORWARD window the gage innovation is averaged
    over before it is distributed upstream.

    An upstream segment's present flow reaches the gage later, so what corrects
    it is the innovation the gage will report on arrival, not the one it reports
    now. The window is one-sided for that reason: hours t to t+width, never
    before t, since an earlier innovation describes water that had already gone
    past.

    The width is NOT derived from ``max_reach_km``. Converting that distance to
    an arrival time needs a celerity, and no defensible one was found (200 km is
    69 h at 0.8 m/s, so the two defaults do not describe the same interval).
    Read it instead as how long a gage innovation is taken to stay informative,
    the same question ``da_decay_min`` answers for the downstream leg, and note
    that nothing in this work measures it: a held-out sweep at 0, 6, 12 and 24 h
    on the Ohio subset separated by 1.4 points of median absolute PBIAS, which
    four gages cannot resolve.

    0 applies the raw innovation. That is the honest default: it claims nothing
    about timing, it makes the confluence background exact rather than
    approximate, and it needs no data from the next forcing window, so results
    cannot depend on the window partition."""

    travel_time_lag: bool = False
    """Shift a gage innovation by the travel time before applying it upstream.

    The source proposal applies dQ(t) simultaneously at every subarea and says
    nothing about travel time. The shift is delivered as an agreed addition to
    that proposal, because a correction applied at one instant hundreds of km
    upstream is not physically meaningful.

    OFF by default (2026-08-17, measured record): the lag improves ANALYSIS
    timing (OSSE 2.00 h vs 6.00 h untimed), leaves held-out skill unchanged
    within resolution (13.1% vs 11.5%), and costs the FORECAST nothing by
    construction -- the hand-off instant is always seeded UNTIMED, so the
    forecast equals the untimed arm's. Off for runnability, not skill: the
    span must fit the opening update (fail-closed) and operational 3-28 h
    lookbacks cannot host the 48 h default, plus ~5% runtime. Enable where
    the cadence hosts the span and record timing matters.

    On, an upstream segment at time ``t`` is corrected by ``dQ_o(t + tau)``: a
    correction placed there routes down and reaches the gage at ``t + tau``, so
    this is the shift that makes the upstream field reproduce the innovation the
    gage actually observed. Off applies the innovation at the observation time,
    which is the proposal's own formulation.

    ``tau`` is traced BACKWARD along the characteristic from the router's own
    Courant number (``cn = ck*dt/dx``, so the trace accumulates ``cn`` to 1.0 and
    needs no reach length at all). Every read is therefore at a time already
    routed and held in memory, which is what makes the trace well posed; a
    forward trace would need celerity at ``t + tau``, and the attempt that froze
    the field at ``t`` instead read ambient celerity where an event was passing.
    What is traced is the KINEMATIC WAVE, not the water: by the Kleitz-Seddon law
    a discharge perturbation moves at ``c = dQ/dA = beta*V``, which is ``(5/3)V``
    for a wide Manning channel, so tracing the water velocity would over-estimate
    travel time by that factor.

    There is one estimator and one direction because there is one defensible
    answer for each. Every celerity that could be PICKED was measured and
    rejected (a constant cannot describe a continental network; the raw Courant
    number reports the AMBIENT wave speed, which gives 168 h over 39 km).
    Cross-correlating the routed hydrographs resolved 9.9% of segments against
    the trace's 87.7%, needed a significance gate to stop it fabricating lags
    from noise, and scored the same as applying no timing at all on the held-out
    gages. Applying the shift backward, ``dQ_o(t - tau)``, is ``2*tau`` late by
    construction and scored 12.00 h against 6.00 h for no timing on the OSSE.

    The trace follows the MODEL, not the physics: the solver clamps the
    Muskingum K at one timestep (``Km = max(dt, dx/Ck)``), so its perturbation
    crosses at most one segment per timestep however fast the physical celerity
    is, and the trace accumulates ``min(cn, 1)`` to match. Tracing the raw
    exported cn under-estimated tau by exactly that factor wherever ``cn > 1``,
    which at NHF reach lengths is most event flow.

    Unresolvable segments are excluded, never given a fallback speed; the
    DIFFUSIVE domain (no Courant export) drops with its subtrees deliberately.
    The run log counts unresolved segments by reason; only ``short`` resolves
    under a longer ``lag_window_h``. The drivers force ``return_courant`` on
    while the trace still needs it."""

    lag_window_h: float = Field(48.0, gt=0, allow_inf_nan=False)
    """Hours of routed flow the travel time is traced over, once per run.

    Deliberately NOT the forcing window. The span decides the answer, so a lag
    traced over each window would make ``max_loop_size``, a memory knob, change
    discharge: measured on the Ohio subset, per-window traces moved the median
    tau across 12.1 to 14.4 h in one run, and the resolved SET moved with it
    because the walk stops at the start of the record. Slicing a fixed span from
    the run's start makes every partition read the same data, and both drivers
    enlarge a short first window to cover it.

    It is therefore also the cap on what can be resolved at all: a segment whose
    accumulated Courant number does not cross its reach within the span, or that
    crosses only on the oldest sample the span holds, is excluded rather than
    given a lower bound dressed up as a measurement.

    The trade is a real one. A longer span resolves longer travel times but is
    anchored further from the event, where the trace reads AMBIENT celerity; a
    shorter span reads closer to the flow that matters but excludes more
    segments."""

    spread_chunk_timesteps: Optional[int] = Field(None, ge=0)
    """Time-chunking of the upstream spread's memory transients. Unset = auto
    (chunk only above ~0.5 GB per transient), 0 = never, N = fixed chunks.
    Bit-identical to unchunked."""


    # Real observations come from the shared usgs_timeslices_folder under the
    # shared qc_threshold / interpolation_limit_min; there is deliberately no
    # scaling-DA-specific observation source.

class ReservoirPersistenceDA(BaseModel):
    """
    Parameters controlling persistence reservoir DA. This if for USGS/USACE reservoirs.
    """
    reservoir_persistence_usgs: bool = False
    """
    If True, USGS reservoirs will perform data assimilation.
    """
    reservoir_persistence_usace: bool = False
    """
    If True, USACE reservoirs will perform data assimilation.
    """
    reservoir_persistence_greatLake: bool = False
    """
    If True, Great Lakes will perform data assimilation.
    """
    reservoir_persistence_usbr: bool = False
    """
    If True, USBR Reservoirs will perform data assimilation.
    """

    crosswalk_usgs_gage_field: str = "usgs_gage_id"
    """
    Column name designation in files for USGS gages.
    """
    crosswalk_usace_gage_field: str = "usace_gage_id"
    """
    Column name designation in files for USACE gages.
    """
    crosswalk_usgs_lakeID_field: str = "usgs_lake_id"
    """
    Column name designation in files for USGS lake IDs.
    """
    crosswalk_usace_lakeID_field: str = "usace_lake_id"
    """
    Column name designation in files for USACE lake IDs.
    """


class ReservoirRfcParameters(BaseModel):
    """
    Parameters controlling RFC reservoirs DA.
    """
    reservoir_rfc_forecasts: Literal[True] = True
    """
    If True, RFC reservoirs will perform data assimilation.
    """
    reservoir_rfc_forecasts_time_series_path: Optional[DirectoryPath] = None
    """
    Directory containing RFC timeseries files.
    NOTE: Required if reservoir_rfc_forecasts is True.
    """
    reservoir_rfc_forecasts_lookback_hours: int = 28
    """
    Hours to look back in time from simulation time for RFC timeseries files.
    """
    reservoir_rfc_forecasts_offset_hours: int = 28
    """
    Offset hours forward in time from simulation time to look for files. 
    This helps find the most recent RFC timeseries files for operational NWM use.
    """
    reservoir_rfc_forecast_persist_days: int = 11
    """
    Days to persist an observation when no new, good observations can be found.
    """


class ReservoirRfcParametersDisabled(BaseModel):
    reservoir_rfc_forecasts: Literal[False] = False


class DiversionDA(BaseModel):
    """
    Parameters controlling diversion DA: a managed transfer of flow out of one
    flowpath, measured by a streamgage, as at the Old River Control Structure.

    The observed discharge is subtracted from the donor flowpath in the routing
    kernel. The receiving river gains the same water through ordinary streamflow
    nudging at the diversion gage, which sits on a headwater flowpath of the
    receiving system, so ``streamflow_da.streamflow_nudging`` must be enabled for
    the transfer to conserve mass.
    """
    diversion_gage_crosswalk: Dict[int, str] = {}
    """
    Donor flowpath ``fp_id`` -> site number of the gage measuring the diverted flow.

    The full observed discharge is removed from the donor, which assumes the model
    routes no flow of its own down the diversion path. That holds where the diversion
    gage sits on a headwater flowpath, as it does at Old River. If it were ever mapped
    to a gage with upstream contributing area, the correction would need to be the
    observed discharge minus the simulated flow already leaving the donor, not the
    observed discharge alone.

    Mass is conserved except where the observed diversion exceeds the routed donor
    flow: the donor is then floored at zero and gives up less than the receiving river
    gains. Capping the transfer at the routed flow would require the donor's discharge
    inside the receiving reach's compute job, which crosses the parallel
    decomposition, so the condition is reported at WARNING with the volume involved
    rather than enforced. It has not been observed over the available record.
    """
    persist_historical_median: bool = False
    """
    Fill gaps in the diversion gage record with hardcoded monthly climatology, so
    forecast timesteps (which have no observations) still divert. Substituted values
    are logged; they are climatology, not observations.

    Despite the field name the stored values are monthly MEANS retrieved from NWIS
    (``_DIVERSION_MONTHLY_MEANS`` in ``troute.DataAssimilation``), and they exist for
    the Old River gage only. A diversion gage with no entry there is skipped with a
    warning rather than filled.
    """


class ReservoirDA(BaseModel):
    """
    Parameters controlling reservoir DA.
    """
    reservoir_persistence_da: Optional[ReservoirPersistenceDA] = None
    reservoir_rfc_da: Optional[
        Union[ReservoirRfcParameters, ReservoirRfcParametersDisabled]
    ] = Field(None, discriminator="reservoir_rfc_forecasts")
    reservoir_parameter_file: Optional[FilePath] = None
    """
    File conaining reservoir parameters (e.g., reservoir_index_AnA.nc).
    NOTE: Needed for NHDNetwork, but not HYFeatures as this information is included in the hydrofabric.
    """


class DataAssimilationParameters(BaseModel):
    model_config = ConfigDict(extra='ignore')
    """
    Parameters controlling data assimilation.
    """
    usgs_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to usgs timeslice files.
    NOTE: required for streamflow nudging and/or USGS reservoir DA
    """
    usace_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to usace timeslice files.
    NOTE: required for USACE reservoir DA
    """
    usbr_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to USBR timeslice files.
    NOTE: required for USBR reservoir DA
    """
    canada_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to canadian timeslice files. 
    NOTE: required for Lake Erie DA (and streamflow nudging using Canadian gages, though that has not been 
    implemented as of June 25, 2024).
    """
    LakeOntario_outflow: Optional[FilePath] = None
    """
    CSV file containing DA values for Lake Ontario. Needs to be obtained and pre-processed from https://ijc.org/en/loslrb/watershed/flows.
    NOTE: Required for Lake Ontario DA.
    """
    timeslice_lookback_hours: int = 24
    """
    Number of hours to look back in time (from simulation time) for USGS, USACE, and Canadian timeslice data assimilation files.
    """
    interpolation_limit_min: int = 59
    """
    Limit on how many missing values can be replaced by linear interpolation from timeslice files.
    """

    wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time: int = 0
    """
    Lead time of lastobs relative to simulation start time (secs).
    NOTE: Only relevant if using a WRF-Hydro lastobs restart file.
    """
    wrf_lastobs_type: str = "obs-based"
    
    streamflow_da: StreamflowDA = None
    reservoir_da: Optional[ReservoirDA] = None
    diversion_da: Optional[DiversionDA] = None

    qc_threshold: float = Field(1, ge=0, le=1)
    """
    Threshold for determining which observations are deemed acceptable for DA and which are not. If the values is set to 1, 
    then only the very best observations are retained. On the other hand, if the value is set to 0, then all observations will be 
    used for assimilation, even those markesd as very poor quality.
    """

    da_decay_coefficient: float = Field(120, gt=0)
    """Minutes over which an assimilated observation decays back toward the model
    once observations stop; governs the in-kernel decay for both nudging and the
    simple-scaling DA."""

    @model_validator(mode='before')
    @classmethod
    def coerce_none_to_default(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("qc_threshold") is None:
            values["qc_threshold"] = 1
        if values.get("timeslice_lookback_hours") is None:
            values["timeslice_lookback_hours"] = 24
        return values

    @model_validator(mode='after')
    def check_diversion_has_an_observation_source(self):
        """A diversion needs the gage's discharge to reach the observation frame.

        The kernel subtracts the observed diversion from the donor flowpath. Nothing
        adds it to the receiving river in code: the gage sits on a headwater of the
        receiving system, so imposing the observed discharge there routes the water
        down through the existing topology. Either source populates that row,
        ``streamflow_nudging`` from timeslices or ``persist_historical_median`` from
        climatology, and both conserve the transfer.

        With neither set the observation frame stays empty, the kernel map resolves
        to nothing and the diversion silently does not happen, which is worth saying
        out loud rather than leaving to be discovered in the output.
        """
        diversion = self.diversion_da
        if diversion is None or not diversion.diversion_gage_crosswalk:
            return self
        nudging = bool(self.streamflow_da and self.streamflow_da.streamflow_nudging)
        if not nudging and not diversion.persist_historical_median:
            LOG.warning(
                "diversion_da.diversion_gage_crosswalk is set but neither "
                "streamflow_da.streamflow_nudging nor "
                "diversion_da.persist_historical_median is enabled, so no discharge "
                "is available at the diversion gage and no flow will be diverted."
            )
        return self

    @model_validator(mode='after')
    def check_scaling_da_is_usable(self) -> Self:
        """Reject configurations that would run but silently assimilate nothing."""
        sda = self.streamflow_da
        if sda is None or not sda.streamflow_scaling:
            return self
        scaling = sda.streamflow_scaling_parameters

        if scaling.synthetic_obs_factor is None and not self.usgs_timeslices_folder:
            raise ValueError(
                "streamflow_da.streamflow_scaling is true but neither "
                "synthetic_obs_factor nor usgs_timeslices_folder is set, so there "
                "are no observations to assimilate and the run would be identical "
                "to a no-DA run. Point usgs_timeslices_folder at a TimeSlice "
                "directory."
            )

        if scaling.synthetic_obs_factor is not None and not scaling.synthetic_obs_baseline:
            raise ValueError(
                "streamflow_scaling_parameters.synthetic_obs_factor requires "
                "synthetic_obs_baseline, a frozen no-DA output directory. Scaling the "
                "live model instead is endogenous and compounds under feedback."
            )
        return self

    @model_validator(mode='before')
    @classmethod
    def reject_legacy_scaling_da_block(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """The pre-release spelling must fail loudly, not be silently ignored."""
        if isinstance(values, dict) and "scaling_da" in values:
            raise ValueError(
                "data_assimilation_parameters.scaling_da has moved: use "
                "streamflow_da.streamflow_scaling: true with options under "
                "streamflow_da.streamflow_scaling_parameters."
            )
        return values


class ForcingParameters(BaseModel):
    """
    Parameters controlling model forcing.
    """
    qts_subdivisions: int = 12
    """
    The number of routing simulation timesteps per qlateral time interval. For example, if dt_qlateral = 3600 secs, 
    and dt = 300 secs, then qts_subdivisions = 3600/300 = 12
    """
    dt: int = 300
    """
    Time step size (seconds). Default is 5 mintues
    """
    qlat_input_folder: Optional[DirectoryPath] = None
    et_input_folder: Optional[DirectoryPath] = None
    """
    Name of the directory where ET forcings are stored for channel loss
    """
    nts: Optional[int] = 288
    """
    Number of timesteps. This value, multiplied by 'dt', gives the total simulation time in seconds.
    """
    max_loop_size: int = 24
    """
    Value is in hours. To handle memory issues, t-route can divvy it's simulation time into chunks, reducing the amount 
    of forcing and data assimilation files it reads into memory at once. This is the size of those time loops.
    """
    qlat_file_index_col: str = "feature_id"
    """
    Name of column containing flowpath/nexus IDs
    """
    qlat_file_value_col: str = "q_lateral"
    """
    Name of column containing q_lateral data.
    """
    qlat_file_gw_bucket_flux_col: str = "qBucket"
    """
    Groundwater bucket flux (to channel) variable name in forcing file.
    NOTE: Only needed if using WRF-Hydro output files (CHRTOUT) as forcing files.
    """
    qlat_file_terrain_runoff_col: str = "qSfcLatRunoff"
    """
    Surface terrain runoff (to channel) variable name in forcing file.
    NOTE: Only needed if using WRF-Hydro output files (CHRTOUT) as forcing files.
    """
    qlat_file_pattern_filter: Optional[str] = "*NEXOUT"
    """
    Globbing file pattern to identify q_lateral forcing files.
    """
    et_file_pattern_filter: Optional[str] = None
    """
    Globbing file pattern to identify ET forcing files.
    """

    qlat_forcing_sets: Optional[List[QLateralForcingSet]] = None
    binary_nexus_file_folder: Optional[DirectoryPath] = None
    """
    Directory to save converted forcing files. Only needed if running t-route as part of ngen suite AND if t-route is having memory issues.
    NOTE: Exlpanation: Ngen outputs q_lateral files as 1 file per nexus containing all timesteps. t-route requires 1 file per timestep 
    containing all locations. If this parameter is omitted or left blank, t-route will simply read in all of ngen's output q_lateral files 
    into memory and will attempt routing. If the simulation is large (temporally and/or spatially), t-route might crash due to memory issues. 
    By providing a directory to this parameter, t-route will convert ngen's output q_lateral files into parquet files in the format t-route 
    needs. Then, during routing, t-route will only read the required parquet files as determined by 'max_loop_size', thus reducing memory.
    """
    coastal_boundary_input_file: Optional[FilePath] = None
    """
    File containing coastal model output.
    NOTE: Only used if running diffusive routing.
    """

    ssout: float = 0.0
    """
    Parameter SSOUT specifies the sub-surface loss and is defined as the rate in CMS of sub-surface outflow along the stream channel.
    """

    peadj: float = 1.0 
    """
    When ET data are used the daily ET values can be adjusted with a constant adjustment factor (PEADJ). This is calibratable
    """

class ComputeParameters(BaseModel):
    """
    Parameters specific to the routing simulation.
    """
    parallel_compute_method: ParallelComputeMethod = "by-network"
    """
    parallel computing scheme used during simulation, options below
    - "serial": no parallelization
    - "by-network": parallelization across independent drainage basins
    - "by-subnetwork-jit": parallelization across subnetworks 
    - "by-subnetwork-jit-clustered": parallelization across subnetworks, with clustering to optimize scaling
    """
    compute_kernel: ComputeKernel = "V02-structured"
    """
    routing engine used for simulation
    - "V02-structured" - Muskingum Cunge
    NOTE: There are two other options that were previously being developed for use with the diffusive kernel, 
    but they are now depricated:
    - "diffusive" - Diffusive with adaptive timestepping
    - "diffusice_cnt" - Diffusive with CNT numerical solution
    TODO: Remove these additional options? And this parameter altogether as there is only one option?
    """
    assume_short_ts: bool = False
    """
    If True the short timestep assumption used in WRF hyro is used. if False, the assumption is dropped.
    """
    subnetwork_target_size: int = 10000
    """
    The target number of segments per subnetwork, only needed for "by-subnetwork..." parallel schemes.
    The magnitude of this parameter affects parallel scaling. This is to improve efficiency. Default value has 
    been tested as the fastest for CONUS simultions. For smaller domains this can be reduced.
    """
    cpu_pool: Optional[int] = 1
    """
    Number of CPUs used for parallel computations
    If parallel_compute_method is anything but 'serial', this determines how many cpus to use for parallel processing.
    """
    return_courant: bool = False
    """
    If True, Courant metrics are returnd with simulations. This only works for MC simulations
    """

    restart_parameters: RestartParameters = Field(default_factory=RestartParameters)
    hybrid_parameters: HybridParameters = Field(default_factory=HybridParameters)
    forcing_parameters: ForcingParameters = Field(default_factory=ForcingParameters)
    data_assimilation_parameters: DataAssimilationParameters = Field(default_factory=DataAssimilationParameters)


ComputeParameters.model_rebuild()
QLateralForcingSet.model_rebuild()
ReservoirRfcParameters.model_rebuild()
