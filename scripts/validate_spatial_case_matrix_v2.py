"""Run the second, separately declared Spatial V1 controlled-view partition."""

import scripts.validate_spatial_case_matrix as matrix

matrix.SEEDS = (1501, 1502)

if __name__ == "__main__":
    matrix.main()
