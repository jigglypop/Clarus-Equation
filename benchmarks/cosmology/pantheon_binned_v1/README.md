# Pantheon 40-bin compact input

저다운로드 초신성 거리 형상 게이트에 사용하는 공식 Pantheon DS17 40-bin
자료다. Pantheon+ 1701-object 자료가 아니며 두 원본 파일은 수정하지 않는다.

| 파일 | SHA-256 | 출처 |
|---|---|---|
| `lcparam_DS17f.txt` | `085daafcc4ae19ece72e69d69ac84fb0a4a1f52626ac4782e46571e6d679b000` | <https://raw.githubusercontent.com/dscolnic/Pantheon/master/Binned_data/lcparam_DS17f.txt> |
| `sys_DS17f.txt` | `642391b0a56ee4f0c3275e85376fbdb880c1c289503520fd32b3920c19f4d7d9` | <https://raw.githubusercontent.com/dscolnic/Pantheon/master/Binned_data/sys_DS17f.txt> |

`sys_DS17f.txt`는 첫 줄의 차원 40 뒤에 row-major 40×40 systematic
covariance가 온다. 게이트는 `lcparam_DS17f.txt`의 `dmb²`를 대각에 더하고
절대등급/Hubble intercept 하나를 해석적으로 profile한다.
