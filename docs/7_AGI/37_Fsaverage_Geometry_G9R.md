# G9-R: 실제 fsaverage 국소 방향 기하

TemplateFlow의 왼쪽 fsaverage 10k pial surface, sulcal depth, curvature만 사용한다. 세 파일 합계는 373,423 bytes다. HCP S1200 2.3 GB 묶음은 `SKIPPED_COST`다.

성장장이나 연결 텐서가 없으므로 꼬임의 원인을 검증하지 않는다. 기준선은 중심화 좌표, 반경, 공식 scalar curvature를 사용한다. 후보는 pial mesh one-ring의 평균ㆍ표준 edge 길이, coordinate Laplacian 크기와 방사 성분, 이웃 covariance 고유값 비율을 추가한다. 목표는 공식 sulcal depth다.

무작위 vertex 분할은 공간 누출이 크므로 중심 기준 방위각을 8 sector로 나눈다. validation은 짝수 sector, locked test는 홀수 sector를 하나씩 holdout한다. 후보가 RMSE를 5% 이상 줄이고 sulcus/gyrus 부호 정확도를 2%p 이상 개선해야 한다. 통과해도 단일 집단평균 표면의 연관성일 뿐 발달ㆍ유전ㆍ기능ㆍ인과 증거가 아니다.

V1은 `GZipBase64Binary`를 gzip wrapper로만 해석해 zlib header 오류로 parser 단계에서 실패했다. V2는 zlib stream을 우선하고 gzip wrapper를 fallback으로 허용하며 압축 해제 후 원소 수를 검증한다. 모델과 판정 문턱은 유지한다.

V2는 parser와 무결성을 통과했지만 scalar curvature+위치 기준선 대비 local mesh 특징의 RMSE 개선이 1.74%에 그쳤고 부호 정확도는 0.48%p 하락했다. 따라서 방향 기하의 추가설명력 주장은 반증됐다. V3는 위치만 쓰는 atlas baseline 대비 curvature+local geometry 전체를 검사하는 약한 연관성 주장으로 축소하되, local-over-scalar ablation을 계속 출력한다.
