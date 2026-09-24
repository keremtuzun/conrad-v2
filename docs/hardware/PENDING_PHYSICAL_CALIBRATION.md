# Pending physical calibration

No robot or structural inspection payload has been measured. The values below
are synthetic engineering estimates or unknowns. The software mechanisms may
be tested, but a physical detection or endurance claim requires measurements.

| Physical parameter | Software placeholder and status | Measurement required | Configuration entry point |
|---|---|---|---|
| Footprint versus standoff | `StructuralSensorModelV2.footprint_width_m`, `footprint_height_m`; ENGINEERING_ESTIMATE in tests | Measure visible surface width and height across the operating range | Structural sensor model, then mission sensor options |
| Field of view | 100 by 80 deg; ENGINEERING_ESTIMATE | Measure usable horizontal and vertical FOV | `StructuralSensorOptions.hfov_deg`, `vfov_deg` |
| Working range and standoff | 0.3 to 4.0 m; ENGINEERING_ESTIMATE | Establish detection quality versus standoff | `StructuralSensorOptions.min_range_m`, `max_range_m`; structural sensor model |
| Crack detectability | Patch size, crack length, and crack depth limits are separate `ENGINEERING_ESTIMATE` fields | Detection probability versus crack length, depth, orientation and surface | `StructuralSensorModelV2.minimum_resolvable_crack_m`, `minimum_detectable_crack_length_m`, `minimum_detectable_crack_depth_m` |
| Corrosion spatial resolution | `minimum_resolvable_corrosion_m`, `axial_resolution_m`, `lateral_resolution_m`; ENGINEERING_ESTIMATE in tests | Minimum resolvable patch and pixel footprint across range | `StructuralSensorModelV2` |
| Structural measurement noise | `noise_sigma_m`; ENGINEERING_ESTIMATE in tests | Repeatability and bias by condition and material | Structural sensor model |
| Range noise versus range | 0.05 m constant; ENGINEERING_ESTIMATE | Range error curve across standoff | `StructuralSensorOptions.range_sigma_m` |
| Noise versus turbidity | T0 degradation and 20 NTU full scale; ENGINEERING_ESTIMATE | Response and uncertainty across turbidity | `StructuralSensorOptions.degradation`, `turbidity_ntu_full_scale` |
| Orientation sensitivity | 0.02 rad angular noise; ENGINEERING_ESTIMATE | Detection and geometry error versus incidence angle | `StructuralSensorOptions.angle_sigma_rad`; later response parameters |
| Pose and support uncertainty | Optional; UNKNOWN when absent | Localization covariance and footprint-edge error in water | Pose covariance and `CapsuleSurfaceSupport` uncertainty fields |
| Capsule survey endpoint registration | Exact axis or clipped synthetic endpoint bound; ENGINEERING_ESTIMATE | Measure or independently certify the maximum endpoint registration error for a surveyed asset; a Gaussian sigma alone cannot supply a hard bound | `MissionWorldOptions.survey_endpoint_bound_m`, `DesignComponent.survey_endpoint_bound_m`, `spatial_registration.py` |
| Measurement latency | No calibrated value; UNKNOWN | Capture-to-observation delay distribution | Sensor scheduling and timestamp configuration |
| Sensor power | 5 W candidate placeholder; ENGINEERING_ESTIMATE | Measure draw by active mode and duty cycle | `CandidateSensor.power_w` and hardware sensor configuration |
| Thruster power | Synthetic coefficient 6 W/N^1.5; ENGINEERING_ESTIMATE | Thrust-to-power curve and battery voltage/current | `conrad/sim/kernel/params.py` |
| Vehicle drag/current response | Synthetic dynamics; ENGINEERING_ESTIMATE | Tow/current trials and vehicle motion response | Kernel dynamics parameters |
| Communication throughput/loss | Simulated link profiles; ENGINEERING_ESTIMATE | Throughput, outage, latency and loss distributions | `conrad/orchestration/mission_config.py` link and communication profile |
