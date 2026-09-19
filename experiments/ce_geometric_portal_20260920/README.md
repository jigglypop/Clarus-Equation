# CE-GP1

양의 포털의 내부 곡률 후보를 검사한다. 기존 CE에서 SU(6)나 반사 경계조건이 유도됐다고 주장하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_geometric_portal.py --output results.json
```

32개 새 검사. 중요한 음성대조: 일부 부모 채널을 삭제하면 실제 질량 기울기를 잘못 읽을 수 있다. 명시적인 투영은 tree 포털을 주지만, 비영 CE 분할의 기원은 제공하지 않는다. 관측 피팅과 원격 Git 쓰기는 수행하지 않았다.
