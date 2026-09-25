"""Run the second, separately declared Spatial V1 Unity parity partition."""

import scripts.validate_spatial_unity_parity as parity

parity.SEEDS = (1501, 1502)

if __name__ == "__main__":
    parity.main()
