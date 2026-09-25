"""Run the fourth, separately declared Spatial V1 controlled-view partition."""

import scripts.validate_spatial_case_matrix as matrix

matrix.SEEDS = (1801, 1802)

if __name__ == "__main__":
    matrix.main()
