# V1.1 P4 long run: hourly progress

Run `v11-p4-kerem-long-a1-20260930T001158Z`, 775,000 steps on 1x NVIDIA L4 (conrad-p4-l4b, us-central1-a).
Rank gate: 75. The meaning score is the held-out temporal nearest-neighbour hit rate (chance is about 0.05; checkpoints need >= 0.3).
Cost: TRY spent on this machine so far / projected total for this machine. About 280 TRY more was spent earlier on other machines.

| Time (Turkey) | Step (%) | Steps/s | Sonar rank / meaning | Camera rank / meaning | Projected finish (Turkey) | TRY spent / projected | Status |
|---|---|---|---|---|---|---|---|
| Wed 30 Sep 09:24 | 74,000 (9.5%) | 3.42 | 43.8 / 0.65 | 95.0 / 0.70 | Fri 02 Oct 19:01 | 410 / 2874 | training |
| Wed 30 Sep 12:44 | 115,000 (14.8%) | 3.42 | 45.2 / 0.69 | 92.6 / 0.68 | Fri 02 Oct 18:46 | 553 / 2863 | training |
| Wed 30 Sep 16:08 | 156,000 (20.1%) | 3.25 | 46.8 / 0.72 | 90.1 / 0.69 | Fri 02 Oct 19:04 | 699 / 2876 | training |
| Wed 30 Sep 22:34 | 235,000 (30.3%) | 3.43 | 48.4 / 0.74 | 101.0 / 0.67 | Fri 02 Oct 18:48 | 973 / 2864 | training |
| Thu 01 Oct 13:26 | 416,000 (53.7%) | 3.26 | 61.4 / 0.76 | 94.2 / 0.65 | Fri 02 Oct 18:49 | 1607 / 2865 | training |
| Thu 01 Oct 16:25 | 453,000 (58.5%) | 3.43 | 51.0 / 0.73 | 100.8 / 0.68 | Fri 02 Oct 18:54 | 1736 / 2869 | training |
| Fri 02 Oct 18:44 | 775,000 (100%) | 3.43 | 55.6 / 0.77 | 100.2 / 0.65 | finished | about 2,865 / 2,865 | finished: NO-GO on rank gate (best: sonar 61.4, camera 94.2) |

## Round 3: v11-p4-kerem-sonar5-a1-20261004T165839Z (730,000 steps, pilot A)

| Time (TR) | Step | it/s | Sonar rank / nn (pooled) | Camera rank / nn | Sonar rank per source | ETA | Spent / projected TRY | State |
|---|---|---|---|---|---|---|---|---|
| Sun 04 Oct 21:31 | 18,000 (2.5%) | 3.36 | 95.3 / 0.35 | 83.9 / 0.69 | aquascan 36 catalunya 57 china_offshore 69 subpipe 20 uatd 40 | Wed 07 Oct 12:39 | 228 / 3007 | training |
| Sun 04 Oct 22:20 | 27,000 (3.7%) | 3.19 | 101.3 / 0.37 | 87.1 / 0.69 | aquascan 40 catalunya 57 china_offshore 74 subpipe 20 uatd 40 | Wed 07 Oct 11:11 | 265 / 2943 | training |
| Sun 04 Oct 23:20 | 39,000 (5.3%) | 3.37 | 101.5 / 0.36 | 88.0 / 0.68 | aquascan 46 catalunya 54 china_offshore 73 subpipe 21 uatd 39 | Wed 07 Oct 10:27 | 309 / 2910 | training |
| Mon 05 Oct 02:20 | 75,000 (10.3%) | 3.39 | 101.6 / 0.38 | 91.6 / 0.68 | aquascan 47 catalunya 49 china_offshore 79 subpipe 20 uatd 38 | Wed 07 Oct 09:40 | 441 / 2876 | training |
| Mon 05 Oct 03:20 | 87,000 (11.9%) | 3.36 | 105.9 / 0.37 | 92.4 / 0.66 | aquascan 50 catalunya 51 china_offshore 84 subpipe 19 uatd 38 | Wed 07 Oct 09:33 | 485 / 2871 | training |
| Mon 05 Oct 04:20 | 99,000 (13.6%) | 3.36 | 111.1 / 0.38 | 97.4 / 0.67 | aquascan 52 catalunya 49 china_offshore 78 subpipe 19 uatd 34 | Wed 07 Oct 09:28 | 529 / 2867 | training |
| Mon 05 Oct 05:20 | 111,000 (15.2%) | 3.18 | 106.6 / 0.37 | 94.2 / 0.68 | aquascan 45 catalunya 45 china_offshore 83 subpipe 21 uatd 37 | Wed 07 Oct 09:24 | 573 / 2864 | training |
| Mon 05 Oct 06:20 | 123,000 (16.8%) | 3.39 | 102.5 / 0.38 | 93.6 / 0.66 | aquascan 50 catalunya 52 china_offshore 82 subpipe 19 uatd 35 | Wed 07 Oct 09:21 | 617 / 2862 | training |
| Mon 05 Oct 07:20 | 135,000 (18.5%) | 3.39 | 111.1 / 0.36 | 93.8 / 0.65 | aquascan 48 catalunya 47 china_offshore 83 subpipe 22 uatd 35 | Wed 07 Oct 09:18 | 661 / 2860 | training |
| Mon 05 Oct 08:20 | 147,000 (20.1%) | 3.21 | 104.2 / 0.38 | 95.6 / 0.68 | aquascan 46 catalunya 49 china_offshore 81 subpipe 20 uatd 33 | Wed 07 Oct 09:16 | 705 / 2858 | training |
| Mon 05 Oct 09:20 | 159,000 (21.8%) | 3.38 | 108.8 / 0.38 | 97.2 / 0.64 | aquascan 49 catalunya 49 china_offshore 78 subpipe 23 uatd 35 | Wed 07 Oct 09:14 | 749 / 2857 | training |
| Mon 05 Oct 10:20 | 171,000 (23.4%) | 3.21 | 107.4 / 0.37 | 96.7 / 0.69 | aquascan 47 catalunya 46 china_offshore 89 subpipe 20 uatd 37 | Wed 07 Oct 09:12 | 793 / 2856 | training |
| Mon 05 Oct 12:20 | 195,000 (26.7%) | 3.39 | 108.6 / 0.36 | 93.5 / 0.72 | aquascan 45 catalunya 46 china_offshore 89 subpipe 22 uatd 39 | Wed 07 Oct 09:10 | 881 / 2854 | training |
| Mon 05 Oct 13:20 | 207,000 (28.4%) | 3.37 | 111.8 / 0.36 | 96.9 / 0.68 | aquascan 51 catalunya 42 china_offshore 88 subpipe 24 uatd 33 | Wed 07 Oct 09:08 | 925 / 2853 | training |
| Mon 05 Oct 14:20 | 219,000 (30.0%) | 3.39 | 105.6 / 0.37 | 100.8 / 0.68 | aquascan 50 catalunya 44 china_offshore 89 subpipe 21 uatd 36 | Wed 07 Oct 09:07 | 969 / 2852 | training |
| Mon 05 Oct 15:20 | 231,000 (31.6%) | 3.21 | 115.3 / 0.37 | 101.1 / 0.69 | aquascan 49 catalunya 44 china_offshore 89 subpipe 25 uatd 38 | Wed 07 Oct 09:07 | 1013 / 2851 | training |
| Mon 05 Oct 16:20 | 243,000 (33.3%) | 3.39 | 109.0 / 0.38 | 97.5 / 0.67 | aquascan 44 catalunya 47 china_offshore 91 subpipe 24 uatd 35 | Wed 07 Oct 09:06 | 1057 / 2851 | training |
| Mon 05 Oct 17:20 | 255,000 (34.9%) | 3.37 | 103.7 / 0.37 | 100.9 / 0.64 | aquascan 49 catalunya 43 china_offshore 85 subpipe 20 uatd 39 | Wed 07 Oct 09:05 | 1101 / 2850 | training |
| Mon 05 Oct 18:20 | 267,000 (36.6%) | 3.38 | 114.7 / 0.37 | 99.8 / 0.69 | aquascan 54 catalunya 47 china_offshore 87 subpipe 24 uatd 35 | Wed 07 Oct 09:04 | 1145 / 2850 | training |
| Mon 05 Oct 19:20 | 279,000 (38.2%) | 3.39 | 106.4 / 0.36 | 99.8 / 0.68 | aquascan 51 catalunya 44 china_offshore 87 subpipe 23 uatd 36 | Wed 07 Oct 09:04 | 1189 / 2849 | training |
| Mon 05 Oct 20:20 | 291,000 (39.9%) | 3.20 | 117.1 / 0.37 | 105.2 / 0.68 | aquascan 49 catalunya 42 china_offshore 95 subpipe 25 uatd 36 | Wed 07 Oct 09:03 | 1233 / 2849 | training |
| Mon 05 Oct 21:20 | 304,000 (41.6%) | 3.37 | 114.6 / 0.35 | 97.2 / 0.67 | aquascan 50 catalunya 41 china_offshore 92 subpipe 22 uatd 36 | Wed 07 Oct 09:03 | 1277 / 2849 | training |
| Mon 05 Oct 22:20 | 316,000 (43.3%) | 3.05 | 109.9 / 0.36 | 99.1 / 0.68 | aquascan 44 catalunya 44 china_offshore 89 subpipe 24 uatd 35 | Wed 07 Oct 08:56 | 1321 / 2843 | training |
| Mon 05 Oct 23:20 | 328,000 (44.9%) | 3.39 | 105.1 / 0.36 | 99.7 / 0.68 | aquascan 46 catalunya 41 china_offshore 92 subpipe 23 uatd 37 | Wed 07 Oct 08:51 | 1365 / 2840 | training |
| Tue 06 Oct 00:20 | 340,000 (46.6%) | 3.39 | 103.1 / 0.37 | 97.7 / 0.67 | aquascan 41 catalunya 46 china_offshore 95 subpipe 21 uatd 33 | Wed 07 Oct 08:51 | 1409 / 2840 | training |
| Tue 06 Oct 01:20 | 352,000 (48.2%) | 3.37 | 107.2 / 0.35 | 102.1 / 0.69 | aquascan 45 catalunya 43 china_offshore 95 subpipe 24 uatd 36 | Wed 07 Oct 08:51 | 1453 / 2840 | training |
| Tue 06 Oct 02:20 | 364,000 (49.9%) | 3.39 | 106.8 / 0.36 | 102.1 / 0.69 | aquascan 47 catalunya 39 china_offshore 91 subpipe 23 uatd 35 | Wed 07 Oct 08:51 | 1497 / 2840 | training |
| Tue 06 Oct 03:20 | 376,000 (51.5%) | 3.21 | 109.2 / 0.36 | 93.8 / 0.70 | aquascan 47 catalunya 45 china_offshore 96 subpipe 25 uatd 35 | Wed 07 Oct 08:51 | 1541 / 2840 | training |
| Tue 06 Oct 04:20 | 388,000 (53.2%) | 3.39 | 111.6 / 0.36 | 95.8 / 0.66 | aquascan 46 catalunya 41 china_offshore 90 subpipe 23 uatd 36 | Wed 07 Oct 08:51 | 1585 / 2840 | training |
| Tue 06 Oct 05:20 | 400,000 (54.8%) | 3.39 | 107.7 / 0.35 | 97.9 / 0.68 | aquascan 45 catalunya 44 china_offshore 91 subpipe 23 uatd 35 | Wed 07 Oct 08:51 | 1629 / 2840 | training |
| Tue 06 Oct 06:20 | 412,000 (56.4%) | 3.39 | 111.3 / 0.36 | 103.9 / 0.68 | aquascan 49 catalunya 42 china_offshore 95 subpipe 25 uatd 33 | Wed 07 Oct 08:51 | 1673 / 2840 | training |
| Tue 06 Oct 07:20 | 424,000 (58.1%) | 3.38 | 107.3 / 0.36 | 103.7 / 0.67 | aquascan 47 catalunya 41 china_offshore 90 subpipe 22 uatd 36 | Wed 07 Oct 08:51 | 1717 / 2840 | training |
| Tue 06 Oct 08:20 | 436,000 (59.7%) | 3.21 | 112.6 / 0.36 | 103.1 / 0.66 | aquascan 48 catalunya 41 china_offshore 97 subpipe 21 uatd 37 | Wed 07 Oct 08:51 | 1761 / 2840 | training |
| Tue 06 Oct 09:20 | 448,000 (61.4%) | 3.39 | 112.1 / 0.36 | 100.8 / 0.67 | aquascan 49 catalunya 41 china_offshore 94 subpipe 23 uatd 34 | Wed 07 Oct 08:51 | 1805 / 2840 | training |
| Tue 06 Oct 10:20 | 460,000 (63.0%) | 3.37 | 110.7 / 0.36 | 106.7 / 0.68 | aquascan 45 catalunya 39 china_offshore 98 subpipe 23 uatd 33 | Wed 07 Oct 08:51 | 1849 / 2840 | training |
| Tue 06 Oct 18:09 | 554,000 (75.9%) | 3.38 | 110.8 / 0.36 | 101.1 / 0.67 | aquascan 51 catalunya 40 china_offshore 95 subpipe 26 uatd 34 | Wed 07 Oct 08:51 | 2191 / 2840 | training |
| Tue 06 Oct 18:20 | 556,000 (76.2%) | 3.21 | 109.2 / 0.35 | 102.1 / 0.67 | aquascan 47 catalunya 39 china_offshore 96 subpipe 24 uatd 34 | Wed 07 Oct 08:51 | 2198 / 2840 | training |
| Tue 06 Oct 19:20 | 568,000 (77.8%) | 3.38 | 111.1 / 0.34 | 103.2 / 0.67 | aquascan 46 catalunya 38 china_offshore 97 subpipe 24 uatd 32 | Wed 07 Oct 08:51 | 2242 / 2840 | training |
| Tue 06 Oct 20:20 | 580,000 (79.5%) | 3.38 | 112.2 / 0.36 | 103.0 / 0.65 | aquascan 46 catalunya 39 china_offshore 98 subpipe 24 uatd 36 | Wed 07 Oct 08:44 | 2286 / 2835 | training |
| Tue 06 Oct 21:20 | 592,000 (81.1%) | 3.38 | 114.7 / 0.37 | 102.7 / 0.66 | aquascan 47 catalunya 40 china_offshore 99 subpipe 25 uatd 34 | Wed 07 Oct 08:45 | 2330 / 2835 | training |
| Tue 06 Oct 22:20 | 605,000 (82.9%) | 3.38 | 109.0 / 0.36 | 101.2 / 0.67 | aquascan 46 catalunya 41 china_offshore 97 subpipe 25 uatd 30 | Wed 07 Oct 08:45 | 2374 / 2835 | training |
| Tue 06 Oct 23:20 | 617,000 (84.5%) | 3.39 | 107.6 / 0.36 | 99.9 / 0.67 | aquascan 43 catalunya 38 china_offshore 97 subpipe 23 uatd 33 | Wed 07 Oct 08:45 | 2418 / 2835 | training |
| Wed 07 Oct 08:46 | 730,000 (100%) | 3.38 | 106.8 / 0.36 | 101.1 / 0.66 | aquascan 46 catalunya 38 china_offshore 98 subpipe 24 uatd 35 | finished | about 2,836 / 2,836 | finished: pooled gate PASSED (best.pt step 451k: sonar 109.4, camera 111.0); per-source sonar not passed, see FINAL_REPORT_v11-p4-kerem-sonar5-a1.md |
