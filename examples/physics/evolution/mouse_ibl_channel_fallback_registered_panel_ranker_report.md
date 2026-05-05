# Mouse IBL/OpenAlyx channel-fallback registered panel ranker

## target equation

$$
H_t=A H_{t-\ell}+B X_t+C R_t+\epsilon_t,
\qquad
y_t=g(X_t,R_t,\hat H_t,\epsilon_t).
$$

This ranker does not train the model. It selects sessions where this gate can be rerun with channel-level region fallback.

## scan

- OpenAlyx: `https://openalyx.internationalbrainlab.org`
- project: `ibl_neuropixel_brainwide_01`
- searched channel-fallback sessions: 483
- scanned offset: 0
- scanned limit: 24
- scanned sessions: 24
- ready sessions in chunk: 24
- selected sessions: 12
- panel ranker passed: `True`

## selected panel

| rank | eid | session | lab | subject | probes | strict | channel | metrics | score |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | `caa5dddc-9290-4e27-9f5e-575ba3598614` | `steinmetzlab/Subjects/NR_0031/2023-07-14/001` | `steinmetzlab` | `NR_0031` | 2 | 0 | 2 | 2 | 228 |
| 2 | `642c97ea-fe89-4ec9-8629-5e492ea4019d` | `steinmetzlab/Subjects/NR_0031/2023-07-12/001` | `steinmetzlab` | `NR_0031` | 2 | 0 | 2 | 2 | 228 |
| 3 | `1dbba733-24a4-4400-9436-53f1bd8428e8` | `hausserlab/Subjects/PL050/2023-06-12/001` | `hausserlab` | `PL050` | 2 | 0 | 2 | 2 | 228 |
| 4 | `3a3ea015-b5f4-4e8b-b189-9364d1fc7435` | `steinmetzlab/Subjects/NR_0029/2023-09-07/001` | `steinmetzlab` | `NR_0029` | 1 | 0 | 1 | 1 | 113 |
| 5 | `d85c454e-8737-4cba-b6ad-b2339429d99b` | `steinmetzlab/Subjects/NR_0029/2023-09-05/001` | `steinmetzlab` | `NR_0029` | 1 | 0 | 1 | 1 | 113 |
| 6 | `e6bdb1f4-b0bf-4451-8f23-4384f2102f91` | `hausserlab/Subjects/PL050/2023-06-15/001` | `hausserlab` | `PL050` | 1 | 0 | 1 | 1 | 113 |
| 7 | `ebce500b-c530-47de-8cb1-963c552703ea` | `churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001` | `churchlandlab_ucla` | `MFD_09` | 1 | 0 | 1 | 1 | 113 |
| 8 | `a7eba2cf-427f-4df9-879b-e53e962eae18` | `churchlandlab_ucla/Subjects/MFD_08/2023-09-08/001` | `churchlandlab_ucla` | `MFD_08` | 1 | 0 | 1 | 1 | 113 |
| 9 | `5ae68c54-2897-4d3a-8120-426150704385` | `churchlandlab_ucla/Subjects/MFD_08/2023-09-07/001` | `churchlandlab_ucla` | `MFD_08` | 1 | 0 | 1 | 1 | 113 |
| 10 | `11163613-a6c9-4975-9586-84dc00481547` | `churchlandlab_ucla/Subjects/MFD_07/2023-09-01/001` | `churchlandlab_ucla` | `MFD_07` | 1 | 0 | 1 | 1 | 113 |
| 11 | `de905562-31c6-4c31-9ece-3ee87b97eab4` | `steinmetzlab/Subjects/NR_0029/2023-08-31/001` | `steinmetzlab` | `NR_0029` | 1 | 0 | 1 | 1 | 113 |
| 12 | `e6594a5b-552c-421a-b376-1a1baa9dc4fd` | `steinmetzlab/Subjects/NR_0029/2023-08-30/001` | `steinmetzlab` | `NR_0029` | 1 | 0 | 1 | 1 | 113 |

## chunk top candidates

| rank | eid | session | lab | subject | probes | score |
|---:|---|---|---|---|---:|---:|
| 1 | `caa5dddc-9290-4e27-9f5e-575ba3598614` | `steinmetzlab/Subjects/NR_0031/2023-07-14/001` | `steinmetzlab` | `NR_0031` | 2 | 228 |
| 2 | `642c97ea-fe89-4ec9-8629-5e492ea4019d` | `steinmetzlab/Subjects/NR_0031/2023-07-12/001` | `steinmetzlab` | `NR_0031` | 2 | 228 |
| 3 | `1dbba733-24a4-4400-9436-53f1bd8428e8` | `hausserlab/Subjects/PL050/2023-06-12/001` | `hausserlab` | `PL050` | 2 | 228 |
| 4 | `3a3ea015-b5f4-4e8b-b189-9364d1fc7435` | `steinmetzlab/Subjects/NR_0029/2023-09-07/001` | `steinmetzlab` | `NR_0029` | 1 | 113 |
| 5 | `d85c454e-8737-4cba-b6ad-b2339429d99b` | `steinmetzlab/Subjects/NR_0029/2023-09-05/001` | `steinmetzlab` | `NR_0029` | 1 | 113 |
| 6 | `de905562-31c6-4c31-9ece-3ee87b97eab4` | `steinmetzlab/Subjects/NR_0029/2023-08-31/001` | `steinmetzlab` | `NR_0029` | 1 | 113 |
| 7 | `e6594a5b-552c-421a-b376-1a1baa9dc4fd` | `steinmetzlab/Subjects/NR_0029/2023-08-30/001` | `steinmetzlab` | `NR_0029` | 1 | 113 |
| 8 | `4e560423-5caf-4cda-8511-d1ab4cd2bf7d` | `steinmetzlab/Subjects/NR_0029/2023-08-29/001` | `steinmetzlab` | `NR_0029` | 1 | 113 |
| 9 | `e6bdb1f4-b0bf-4451-8f23-4384f2102f91` | `hausserlab/Subjects/PL050/2023-06-15/001` | `hausserlab` | `PL050` | 1 | 113 |
| 10 | `2584ce3c-db10-4076-89cb-5d313138dd38` | `hausserlab/Subjects/PL050/2023-06-13/001` | `hausserlab` | `PL050` | 1 | 113 |
| 11 | `ebce500b-c530-47de-8cb1-963c552703ea` | `churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001` | `churchlandlab_ucla` | `MFD_09` | 1 | 113 |
| 12 | `a7eba2cf-427f-4df9-879b-e53e962eae18` | `churchlandlab_ucla/Subjects/MFD_08/2023-09-08/001` | `churchlandlab_ucla` | `MFD_08` | 1 | 113 |
| 13 | `5ae68c54-2897-4d3a-8120-426150704385` | `churchlandlab_ucla/Subjects/MFD_08/2023-09-07/001` | `churchlandlab_ucla` | `MFD_08` | 1 | 113 |
| 14 | `11163613-a6c9-4975-9586-84dc00481547` | `churchlandlab_ucla/Subjects/MFD_07/2023-09-01/001` | `churchlandlab_ucla` | `MFD_07` | 1 | 113 |
| 15 | `15b69921-d471-4ded-8814-2adad954bcd8` | `churchlandlab_ucla/Subjects/MFD_07/2023-08-31/001` | `churchlandlab_ucla` | `MFD_07` | 1 | 113 |
| 16 | `c46b8def-620d-4d70-9397-be5694255f73` | `churchlandlab_ucla/Subjects/MFD_07/2023-08-29/001` | `churchlandlab_ucla` | `MFD_07` | 1 | 113 |
| 17 | `6899a67d-2e53-4215-a52a-c7021b5da5d4` | `churchlandlab_ucla/Subjects/MFD_06/2023-08-29/001` | `churchlandlab_ucla` | `MFD_06` | 1 | 113 |
| 18 | `3f71aa98-08c6-4e79-b4c8-00eae4f03eff` | `churchlandlab_ucla/Subjects/MFD_06/2023-08-25/001` | `churchlandlab_ucla` | `MFD_06` | 1 | 113 |
| 19 | `a1782f4f-86b0-480c-a7f2-3d8f1ab482ab` | `churchlandlab_ucla/Subjects/MFD_06/2023-08-24/001` | `churchlandlab_ucla` | `MFD_06` | 1 | 113 |
| 20 | `97c1d08d-57b4-4595-8052-91eb84ebfd74` | `churchlandlab_ucla/Subjects/MFD_06/2023-08-23/001` | `churchlandlab_ucla` | `MFD_06` | 1 | 113 |
| 21 | `35eeb752-8f4f-4040-9714-ba0f5b7ccdfe` | `churchlandlab_ucla/Subjects/MFD_06/2023-08-22/001` | `churchlandlab_ucla` | `MFD_06` | 1 | 113 |
| 22 | `004d8fd5-41e7-4f1b-a45b-0d4ad76fe446` | `churchlandlab_ucla/Subjects/MFD_05/2023-08-17/001` | `churchlandlab_ucla` | `MFD_05` | 1 | 113 |
| 23 | `ca4ecb4c-4b60-4723-9b9e-2c54a6290a53` | `churchlandlab_ucla/Subjects/MFD_05/2023-08-16/001` | `churchlandlab_ucla` | `MFD_05` | 1 | 113 |
| 24 | `7dd27c2f-9544-448d-918d-be0f9837b0e0` | `churchlandlab_ucla/Subjects/MFD_05/2023-08-15/001` | `churchlandlab_ucla` | `MFD_05` | 1 | 113 |

## spread

| item | count |
|---|---:|
| selected lab `steinmetzlab` | 6 |
| selected lab `churchlandlab_ucla` | 4 |
| selected lab `hausserlab` | 2 |
| selected subject `NR_0029` | 4 |
| selected subject `NR_0031` | 2 |
| selected subject `PL050` | 2 |
| selected subject `MFD_08` | 2 |
| selected subject `MFD_09` | 1 |
| selected subject `MFD_07` | 1 |

## next

- If the selected panel reaches the threshold, rerun the innovation-to-behavior gate on these sessions.
- If not, scan the next chunk with a larger `--offset` or increase `--max-sessions`.
