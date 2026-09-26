"""Run the prospectively declared Spatial V1.1 Unity parity matrix once."""

import scripts.validate_spatial_unity_parity as parity

parity.SEEDS = (2001, 2002)

if __name__ == "__main__":
    parity.main()
