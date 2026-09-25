"""Run the third, separately declared Spatial V1 Unity parity partition."""

import scripts.validate_spatial_unity_parity as parity

parity.SEEDS = (1701, 1702)

if __name__ == "__main__":
    parity.main()
