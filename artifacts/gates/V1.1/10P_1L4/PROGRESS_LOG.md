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
