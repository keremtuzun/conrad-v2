"""Run the prospectively declared Spatial V1.1 controlled-view matrix once."""

import scripts.validate_spatial_case_matrix as matrix

matrix.SEEDS = (2001, 2002)

if __name__ == "__main__":
    matrix.main()
