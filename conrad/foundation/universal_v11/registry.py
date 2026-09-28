"""Source-derived Universal OS-FM V1.1 modality registry.

The registry is intentionally larger than the trainable encoder set.  It is the
typed promise that OceanSense knows a modality, even when that modality is not
yet backed by data, licensing, and a qualified learned branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class EncoderFamily(str, Enum):
    VISUAL_IMAGE = "visual_image"
    SPECTRAL_PHOTONIC = "spectral_photonic"
    ACTIVE_ACOUSTIC = "active_acoustic"
    PASSIVE_ACOUSTIC_VIBRATION = "passive_acoustic_vibration"
    ULTRASONIC_NDT = "ultrasonic_ndt"
    ELECTROMAGNETIC_MAGNETIC = "electromagnetic_magnetic"
    GEOMETRY_SPATIAL = "geometry_spatial"
    MECHANICAL_STRUCTURAL_TIME_SERIES = "mechanical_structural_time_series"
    PHYSICAL_OCEANOGRAPHIC_TIME_SERIES = "physical_oceanographic_time_series"
    CHEMICAL_ELECTROCHEMICAL_SPECTRAL = "chemical_electrochemical_spectral"
    BIOLOGICAL_MOLECULAR = "biological_molecular"
    MICROSCOPIC_PARTICLE = "microscopic_particle"
    GEOPHYSICAL_SEISMIC = "geophysical_seismic"
    RADIOLOGICAL = "radiological"
    ENGINEERING_DOCUMENT_CONTEXT = "engineering_document_context"


class ModalityState(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    DEGRADED = "degraded"
    STALE = "stale"
    INVALID = "invalid"
    UNCALIBRATED = "uncalibrated"
    SATURATED = "saturated"

    @property
    def can_route(self) -> bool:
        return self in {self.AVAILABLE, self.DEGRADED, self.STALE, self.UNCALIBRATED, self.SATURATED}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    family: EncoderFamily
    category: str
    source_label: str
    tokenizer: str
    learned_active: bool
    exact_evidence: bool = False
    sample_or_lab_result: bool = False


_REGISTRY_SOURCE = """
## Optical / imaging
- RGB video
- RGB still
- mono
- low-light
- intensified imaging
- HDR
- high-speed
- macro
- microscopy
- stereo
- multi-camera arrays
- panoramic
- 360
- fisheye
- telephoto
- structured illumination
- photometric stereo
- shape-from-shading
- multi-view photogrammetry
- event cameras
- neuromorphic cameras
## Polarimetric
- linear polarization
- circular polarization
- full-Stokes
- polarization difference
- depolarization
## Spectral
- multispectral
- hyperspectral
- UV
- visible spectroscopy
- short-range NIR where viable
- narrow-band
- tunable filter
- snapshot HSI
## Fluorescence
- chlorophyll
- CDOM
- hydrocarbon
- PAH
- tracer dye
- laser-induced fluorescence
- organism fluorescence
- coral fluorescence
- coating fluorescence
- UV fluorescence
## Laser / photonic geometry
- laser line
- laser triangulation
- structured light
- laser profilometry
- blue-green LiDAR
- scanning LiDAR
- ToF optical
- laser speckle
- laser Doppler vibrometry
- interferometry
- holography
- OCT
- digital image correlation
- fringe projection
## Spectroscopy
- Raman
- resonance Raman
- SERS
- UV-Vis
- absorption
- reflectance
- fluorescence spectroscopy
- LIBS
- laser absorption
- XRF
- FTIR
- ATR-FTIR
- spectrophotometry
- colorimetry
## Active acoustic
- FLS
- scanning sonar
- electronic sonar
- imaging sonar
- multibeam
- single beam
- split beam
- side scan
- interferometric side scan
- SAS
- 3D sonar
- profiler
- obstacle sonar
- altimeter
- fisheries sonar
- multifrequency sonar
- broadband chirp
- biomass sonar
- bubble/plume sonar
- target strength
- acoustic tomography
- sub-bottom
- chirp profiler
- pinger
- boomer
- sparker
## Passive acoustic
- single hydrophone
- arrays
- vector hydrophone
- directional hydrophone
- broadband
- low-frequency
- high-frequency
- fish
- whales
- dolphins
- shrimp
- reef soundscape
- leaks
- cavitation
- valves
- bearings
- machinery
- propeller
- cable vibration
- moorings
- acoustic emissions
- resonance
- impact
- earthquakes
- landslides
- volcanism
- ice
- sediment motion
- vessel noise
- construction
- pile driving
- turbine noise
## Ultrasonic NDT
- pulse echo
- through transmission
- A/B/C scan
- thickness
- corrosion map
- PAUT
- TOFD
- FMC
- TFM
- guided wave
- LRUT
- Lamb
- shear-wave
- creeping-wave
- immersion
- acoustic microscopy
- ultrasonic tomography
- bolt tension
- weld
- UPV
- void detection
- delamination
- flooded member
- composites
- adhesive bond
- coating
- EMAT
- laser ultrasonics
## EM / magnetic
- ECT
- ECA
- PEC
- RFEC
- ACFM
- ACPD
- DCPD
- MFL
- MPI
- Barkhausen
- permeability
- magnetometer
- fluxgate
- atomic/optically pumped magnetometer
- gradiometer
- Hall
- conductivity
- resistivity
- dielectric
- impedance
- capacitance
- electrical tomography
- EM induction
- CSEM
- microwave/contact EM probes
## Electrochemical / corrosion
- CP potential
- Ag/AgCl reference
- zinc reference
- galvanic potential
- corrosion potential
- corrosion current
- LPR
- EIS
- potentiodynamic polarization
- cyclic polarization
- electrical resistance probes
- galvanic probes
- coupons
- weight-loss coupons
- hydrogen permeation
- hydrogen
- chloride activity
- ORP
- pitting potential
- coating impedance
- anode current
- anode depletion
- current density
## Mechanical/contact
- force
- torque
- shear
- pressure
- contact pressure
- tactile
- force-torque
- haptics
- hardness
- microindentation
- rebound hardness
- stiffness
- compliance
- modulus
- scratch
- friction
- roughness
- adhesion
- coating pull-off
- crack-opening displacement
## Structural
- strain
- rosettes
- displacement
- extensometers
- LVDT
- tilt
- acceleration
- gyro
- seismometers
- vibration
- modal response
- natural frequencies
- damping
- load
- mooring tension
- anchor load
- cable tension
- preload
## Pipeline geometry
- caliper pig
- geometry pig
- deformation pig
- ovality
- dents
- buckles
- bore
- pipe movement
## Fiber optics
- FBG strain
- FBG temperature
- DAS
- DTS
- DSS
- distributed vibration
- Brillouin
- Rayleigh
- Raman distributed sensing
- OTDR
- coherent OTDR
- fiber bend loss
- fiber attenuation
- break localization
## Radiological
- X-ray
- gamma radiography
- digital radiography
- neutron radiography
- backscatter
- Geiger
- scintillation
- gamma spectroscopy
- neutron detector
- alpha/beta sample analysis
- dosimetry
- radionuclide analysis
- Cs-137
- Co-60
- Sr-90
- tritium
- uranium series
- radon
- neutron activation
- radioactive tracers
## Physical oceanography
- temperature
- conductivity
- salinity
- pressure
- depth
- density
- sound speed
- sound-speed profile
- refractive index
- point current
- EM current
- acoustic current
- ADCP
- ADV
- velocity
- vertical velocity
- shear
- turbulence
- dissipation
- microstructure
- Reynolds stresses
- eddy covariance
- waves
- significant wave height
- period
- spectrum
- direction
- orbital velocity
- tides
- surge
- water level
- bed shear
- friction velocity
- near-bed flow
- wake
- vortex shedding
## Water optical properties
- turbidity
- nephelometry
- optical backscatter
- transmissometry
- attenuation
- absorption
- scattering
- backscatter
- CDOM
- visibility
- PAR
- irradiance
- UV
- water color
- radiance
- reflectance
## General chemistry
- pH
- pCO2
- CO2
- DIC
- alkalinity
- bicarbonate
- carbonate
- saturation states
- dissolved oxygen
- oxygen saturation
- ORP
- sulfide
- H2S
- nitrate
- nitrite
- ammonium
- ammonia
- phosphate
- silicate
- urea
- total N
- total P
- chloride
- sulfate
- bromide
- fluoride
- sodium
- potassium
- calcium
- magnesium
- DOC
- POC
- TOC
- DOM
- BOD
- COD
## Trace elements
- Fe
- Mn
- Cu
- Zn
- Pb
- Hg
- Cd
- Ni
- Cr
- As
- Al
- Co
- Se
- Mo
- Ag
- Sn
- rare-earth elements
- isotope ratios
- stripping voltammetry
- potentiometry
- ion-selective electrodes
- ICP-MS
- ICP-OES
- atomic absorption
- XRF
- LIBS
## Gases/leaks
- methane
- ethane
- propane
- butane
- H2
- H2S
- CO2
- O2
- N2
- N2O
- noble gases
- VOCs
- membrane-inlet MS
- underwater MS
- TDLAS
- optical absorption
- electrochemical
- GC sample analysis
- bubble acoustics
- bubble imagery
- dissolved-gas extraction
## Pollution / industrial chemistry
- TPH
- PAH
- BTEX
- VOC
- oil
- hydraulic fluid
- lubricants
- drilling mud
- produced water
- antifouling chemicals
- organotins
- pesticides
- herbicides
- solvents
- surfactants
- detergents
- phenols
- PFAS
- pharmaceuticals
- endocrine disruptors
- explosives
- munition residues
- dyes
- chemical tracers
## Biological/ecological imaging
- fish
- mammals
- crustaceans
- coral
- sponge
- mollusk
- echinoderm
- macroalgae
- seagrass
- biofouling
- invasives
- species
- abundance
- biomass
- size
- morphology
- behavior
- health
- bleaching
- disease
- lesions
- growth
- recruitment
- grazing
- predator/prey behavior
## Plankton/microorganism imaging
- microscopy
- holographic microscopy
- flow imaging
- imaging cytometry
- flow cytometry
- plankton camera
- optical plankton counters
- particle counters
## Molecular biology
- eDNA
- eRNA
- metabarcoding
- qPCR
- ddPCR
- metagenomics
- transcriptomics
- metatranscriptomics
- genomes
- proteomics
- metabolomics
- lipidomics
- epigenomics
- ATP
- enzymes
- respiration
- bacterial counts
- fecal indicators
- pathogen assays
- toxins
- immunoassay
- antibody
- aptamer
- CRISPR sensor
## Pigments/productivity
- chlorophyll-a
- chlorophyll-b
- chlorophyll-c
- phycocyanin
- phycoerythrin
- carotenoids
- photosynthetic efficiency
- PAM
- Fv/Fm
- primary productivity
- PAR
- oxygen evolution
- HAB markers
- cyanobacterial toxins
- domoic acid
- saxitoxin
- brevetoxin
- microcystin
## Particles/microplastics
- particle count
- concentration
- size distribution
- morphology
- settling velocity
- diffraction
- optical counters
- Coulter counters
- acoustic particle sizing
- microplastic imaging
- fluorescent staining
- Raman microscopy
- FTIR microscopy
- HSI plastics
- pyrolysis GC-MS
- nets
- litter
- ghost gear
- wreck debris
## Geological/geotechnical
- bathymetry
- acoustic backscatter
- sediment texture
- ripple morphology
- scour
- erosion
- deposition
- grain size
- density
- porosity
- permeability
- moisture
- shear strength
- cohesion
- friction angle
- pore pressure
- consolidation
- CPT
- piezocone
- penetrometer
- vane shear
- pressuremeter
- cores
- box cores
- grabs
- seismic
- CSEM
- resistivity
- IP
- self-potential
- magnetics
- gravity
- gravimetry
- microgravity
- heat flow
- geothermal gradient
- OBS
## Hydrothermal/seep
- methane
- H2S
- thermal anomalies
- pH
- ORP
- Fe/Mn
- plume turbidity
- bubble acoustics
- bubble imagery
- MS
- sampling
- vent chemistry
- heat flux
## Concrete/civil
- UPV
- impact echo
- pulse echo
- surface wave
- acoustic tomography
- resistivity
- half-cell
- rebar
- cover depth
- contact GPR
- corrosion potential
- crack width
- delamination
- void
- chloride
- carbonation
- strength proxies
- rebound
- concrete cores
- permeability
- acoustic emission
- strain
- vibration
## Power / telecom cables
- voltage
- current
- power
- phase
- insulation resistance
- partial discharge
- dielectric loss
- electrical field
- magnetic field
- cable temperature
- DTS
- strain
- DAS
- OTDR
- fault location
- sheath current
- burial depth
- cable position
- tension
- bend radius
- free span
- scour
- thermal plume
- connector health
## Machinery/process
- vibration spectrum
- acoustics
- motor current
- voltage
- torque
- RPM
- bearing vibration
- bearing temperature
- lubricant condition
- pressure
- flow
- valve position
- actuator force
- cavitation
- shaft displacement
- alignment
- magnetic flux
- insulation
- partial discharge
## Pipeline operational evidence
- pressure
- temperature
- flow
- mass flow
- fluid composition
- valve state
- pump state
- production
- water cut
- gas fraction
- sand production
- chemical injection
- inhibitor dosage
- pigging
- vibration
- slugging
- leak alarms
## Navigation/platform
- GNSS
- USBL
- LBL
- SBL
- DVL
- INS
- IMU
- accelerometer
- gyro
- magnetometer
- depth
- altitude
- pressure
- velocity
- heading
- pitch
- roll
- heave
- SLAM
- VO
- sonar odometry
- terrain-relative navigation
- dead reckoning
- acoustic positioning
- beacon range
- clock synchronization
- covariance
- pose uncertainty
## Sensor/platform health
- sensor temp
- electronics temp
- voltage
- current
- battery state
- battery health
- BMS
- leak detector
- housing humidity
- housing pressure
- thruster current
- thruster RPM
- motor temp
- vibration
- camera gain
- exposure
- focus
- sonar gain
- acoustic frequency
- pulse length
- calibration state
- dropped packets
- timing jitter
- sync uncertainty
- saturation
- noise floor
- bad/dead pixels/elements
- bandwidth
- compression
- communication quality
## Sampling
- Niskin
- syringe
- pump
- filtration
- in-situ filtration
- grab
- box core
- piston core
- push core
- tissue
- scrape
- biofilm
- plankton net
- settlement plate
- microbial filter
- corrosion product
- coating chip
- deposit
- metal coupon
- biofouling
- chromatography
- MS
- sequencing
- microscopy
- elemental analysis
- culturing
- mechanical testing
## Historical/digital
- CAD
- BIM
- P&ID
- FEA
- CFD
- weld maps
- fabrication drawings
- materials
- coatings
- nominal thickness
- design load
- fatigue
- serial/model data
- commissioning
- repair
- inspection history
- UT history
- defects
- maintenance
- operation history
- age
- failure history
- manufacturing QC
- design standards
- digital twin
- environment history
- tides
- ocean models
- habitat
- bathymetry
## Exotic / rare but valid registry entries
- underwater gravimeter
- quantum magnetometer
- SQUID
- atomic magnetometer
- advanced timing / atomic-clock observations
- neutron sensors
- muon tomography where applicable
- NMR contact/sample systems
- microwave resonant sensors
- THz/contact material probes with effectively eliminated water path
- plasmonic biosensors
- surface plasmon resonance
- fiber-optic chemical sensors
- optodes
- microfluidics
- lab-on-chip
- nanopore sequencing
- in-situ wet chemistry
- ion chromatography
- capillary electrophoresis
- electrochemical aptamer sensors
- microbial fuel-cell biosensors
- electronic noses
- electronic tongues
- quantum sensors
- MEMS chemical arrays
- synthetic-biology biosensors
- resonant mass sensors
- quartz-crystal microbalance
- surface-acoustic-wave chemical sensors
"""


_CATEGORY_FAMILY = {
    "Optical / imaging": EncoderFamily.VISUAL_IMAGE,
    "Polarimetric": EncoderFamily.SPECTRAL_PHOTONIC,
    "Spectral": EncoderFamily.SPECTRAL_PHOTONIC,
    "Fluorescence": EncoderFamily.SPECTRAL_PHOTONIC,
    "Laser / photonic geometry": EncoderFamily.GEOMETRY_SPATIAL,
    "Spectroscopy": EncoderFamily.SPECTRAL_PHOTONIC,
    "Active acoustic": EncoderFamily.ACTIVE_ACOUSTIC,
    "Passive acoustic": EncoderFamily.PASSIVE_ACOUSTIC_VIBRATION,
    "Ultrasonic NDT": EncoderFamily.ULTRASONIC_NDT,
    "EM / magnetic": EncoderFamily.ELECTROMAGNETIC_MAGNETIC,
    "Electrochemical / corrosion": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Mechanical/contact": EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES,
    "Structural": EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES,
    "Pipeline geometry": EncoderFamily.GEOMETRY_SPATIAL,
    "Fiber optics": EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES,
    "Radiological": EncoderFamily.RADIOLOGICAL,
    "Physical oceanography": EncoderFamily.PHYSICAL_OCEANOGRAPHIC_TIME_SERIES,
    "Water optical properties": EncoderFamily.SPECTRAL_PHOTONIC,
    "General chemistry": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Trace elements": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Gases/leaks": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Pollution / industrial chemistry": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Biological/ecological imaging": EncoderFamily.VISUAL_IMAGE,
    "Plankton/microorganism imaging": EncoderFamily.MICROSCOPIC_PARTICLE,
    "Molecular biology": EncoderFamily.BIOLOGICAL_MOLECULAR,
    "Pigments/productivity": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Particles/microplastics": EncoderFamily.MICROSCOPIC_PARTICLE,
    "Geological/geotechnical": EncoderFamily.GEOPHYSICAL_SEISMIC,
    "Hydrothermal/seep": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
    "Concrete/civil": EncoderFamily.ULTRASONIC_NDT,
    "Power / telecom cables": EncoderFamily.ELECTROMAGNETIC_MAGNETIC,
    "Machinery/process": EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES,
    "Pipeline operational evidence": EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES,
    "Navigation/platform": EncoderFamily.GEOMETRY_SPATIAL,
    "Sensor/platform health": EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT,
    "Sampling": EncoderFamily.BIOLOGICAL_MOLECULAR,
    "Historical/digital": EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT,
    "Exotic / rare but valid registry entries": EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
}


_FAMILY_TOKENIZER = {
    EncoderFamily.VISUAL_IMAGE: "image",
    EncoderFamily.SPECTRAL_PHOTONIC: "spectra",
    EncoderFamily.ACTIVE_ACOUSTIC: "sonar",
    EncoderFamily.PASSIVE_ACOUSTIC_VIBRATION: "waveform",
    EncoderFamily.ULTRASONIC_NDT: "ndt_scan",
    EncoderFamily.ELECTROMAGNETIC_MAGNETIC: "em_map",
    EncoderFamily.GEOMETRY_SPATIAL: "point_cloud",
    EncoderFamily.MECHANICAL_STRUCTURAL_TIME_SERIES: "time_series",
    EncoderFamily.PHYSICAL_OCEANOGRAPHIC_TIME_SERIES: "time_series",
    EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL: "spectra",
    EncoderFamily.BIOLOGICAL_MOLECULAR: "molecular_sequence",
    EncoderFamily.MICROSCOPIC_PARTICLE: "microscopy",
    EncoderFamily.GEOPHYSICAL_SEISMIC: "seismic",
    EncoderFamily.RADIOLOGICAL: "radiological",
    EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT: "document_context",
}


_ACTIVE_NAMES = {
    "rgb_video",
    "rgb_still",
    "imaging_sonar",
    "structured_light",
    "tof_optical",
    "cad",
    "inspection_history",
}


_EXACT_CATEGORIES = {
    "Electrochemical / corrosion",
    "Mechanical/contact",
    "Structural",
    "Physical oceanography",
    "General chemistry",
    "Trace elements",
    "Gases/leaks",
    "Radiological",
    "Power / telecom cables",
    "Machinery/process",
    "Pipeline operational evidence",
    "Navigation/platform",
    "Sensor/platform health",
}


_SAMPLE_CATEGORIES = {
    "Sampling",
    "Molecular biology",
    "Pigments/productivity",
    "Particles/microplastics",
    "Trace elements",
    "Gases/leaks",
    "Pollution / industrial chemistry",
}


def canonical_name(label: str) -> str:
    lowered = label.strip().lower().replace("°", "deg")
    lowered = lowered.replace("&", "and").replace("+", "plus")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered or "unnamed"


def _parse_registry() -> tuple[ModalitySpec, ...]:
    specs: list[ModalitySpec] = []
    category: str | None = None
    seen: dict[str, int] = {}
    for raw in _REGISTRY_SOURCE.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            category = line[3:].strip()
            if category not in _CATEGORY_FAMILY:
                raise ValueError(f"unmapped modality category {category!r}")
        elif line.startswith("- ") and category is not None:
            label = line[2:].strip()
            base_name = canonical_name(label)
            count = seen.get(base_name, 0)
            seen[base_name] = count + 1
            name = base_name if count == 0 else f"{base_name}_{count + 1}"
            family = _CATEGORY_FAMILY[category]
            specs.append(
                ModalitySpec(
                    name=name,
                    family=family,
                    category=category,
                    source_label=label,
                    tokenizer=_FAMILY_TOKENIZER[family],
                    learned_active=name in _ACTIVE_NAMES,
                    exact_evidence=category in _EXACT_CATEGORIES,
                    sample_or_lab_result=category in _SAMPLE_CATEGORIES,
                )
            )
    return tuple(specs)


_SUPPLEMENTAL_COMPATIBILITY_SPECS = (
    ModalitySpec("rgb_camera", EncoderFamily.VISUAL_IMAGE, "V1 compatibility alias", "RGB camera", "image", True),
    ModalitySpec("sonar_image", EncoderFamily.ACTIVE_ACOUSTIC, "V1 compatibility alias", "sonar image", "sonar", True),
    ModalitySpec("ctd_profile", EncoderFamily.PHYSICAL_OCEANOGRAPHIC_TIME_SERIES, "V1.1 prompt alias", "CTD profile", "time_series", False, exact_evidence=True),
    ModalitySpec("ph_probe", EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, "V1.1 prompt alias", "pH probe", "time_series", False, exact_evidence=True),
    ModalitySpec("paut_scan", EncoderFamily.ULTRASONIC_NDT, "V1.1 prompt alias", "PAUT scan", "ndt_scan", False),
    ModalitySpec("tofd_scan", EncoderFamily.ULTRASONIC_NDT, "V1.1 prompt alias", "TOFD scan", "ndt_scan", False),
    ModalitySpec("tfm_scan", EncoderFamily.ULTRASONIC_NDT, "V1.1 prompt alias", "TFM scan", "ndt_scan", False),
    ModalitySpec("em_ndt_eddy_current", EncoderFamily.ELECTROMAGNETIC_MAGNETIC, "V1.1 prompt alias", "EM NDT eddy current", "em_map", False),
    ModalitySpec("edna_sequence", EncoderFamily.BIOLOGICAL_MOLECULAR, "V1.1 prompt alias", "eDNA sequence", "molecular_sequence", False, sample_or_lab_result=True),
    ModalitySpec("gamma_spectrometer", EncoderFamily.RADIOLOGICAL, "V1.1 prompt alias", "gamma spectrometer", "radiological", False, exact_evidence=True),
    ModalitySpec("synthetic_aperture_sonar", EncoderFamily.ACTIVE_ACOUSTIC, "V1.1 prompt alias", "synthetic aperture sonar", "sonar", False),
    ModalitySpec("seismic_reflection", EncoderFamily.GEOPHYSICAL_SEISMIC, "V1.1 prompt alias", "seismic reflection", "seismic", False),
)


MODALITY_REGISTRY: tuple[ModalitySpec, ...] = _parse_registry() + _SUPPLEMENTAL_COMPATIBILITY_SPECS


def registry_by_name() -> dict[str, ModalitySpec]:
    return {spec.name: spec for spec in MODALITY_REGISTRY}


def inactive_modalities() -> tuple[str, ...]:
    return tuple(spec.name for spec in MODALITY_REGISTRY if not spec.learned_active)


def registry_categories() -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.category for spec in MODALITY_REGISTRY))


def validate_registry() -> None:
    names = [spec.name for spec in MODALITY_REGISTRY]
    if len(names) != len(set(names)):
        raise ValueError("duplicate V1.1 modality registry names")
    missing = set(EncoderFamily) - {spec.family for spec in MODALITY_REGISTRY}
    if missing:
        raise ValueError(f"missing encoder families: {sorted(item.value for item in missing)}")
    if len(MODALITY_REGISTRY) < 800:
        raise ValueError("V1.1 registry lost source-prompt modalities")
