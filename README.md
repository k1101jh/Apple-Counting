# 사과나무 영상에서 YOLO와 Tracking을 사용한 사과 계수



## 데이터셋
- 농촌진흥청 사과나무 데이터셋
  - 대구 군위군 사과연구소에서 촬영
  - 2023년 7월, 8월, 10월에 촬영
  - 7월 데이터
    - 5개의 열에 대해 촬영. 각 열마다 5그루씩
  - 8월 데이터
    - 6개의 열에 대해 촬영. 각 열마다 5그루씩
  - 10월 데이터
    - 3개의 열에 대해 촬영. 각 열마다 17, 35, 17그루
- [이전 논문](https://doi.org/10.1016/j.compag.2022.107513)에서 제공한 [Sensitivity Anlaysis 데이터셋](https://zenodo.org/records/7383338)
  - 사과나무 영상 7개
  - 사람이 직접 촬영

## 사과 탐지 모델
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Faster-RCNN(mmdetection)](https://github.com/open-mmlab/mmdetection)
- [EfficientDet(mmdetection)](https://github.com/open-mmlab/mmdetection)

## 추적 알고리즘
- [ByteTrack](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py) 공식 깃허브: [link](https://github.com/ifzhang/ByteTrack)
- 제안 알고리즘
  - 기준 Track을 선별
  - 기준 Track의 속도를 사용하여 매칭에 실패한 track의 속도를 추정
  - 기준 Track의 속도를 새로 생성한 track의 속도로 사용

## 평가 지표
- [MOTA](https://doi.org/10.1007/s11263-020-01393-0)
  - MOT challenge에서 사용하는 추적 성능 지표

## 실행 코드
- counting.py
  - 영상을 입력받아 counting 수행
- visualize_counted_tracks.py
  - 영상에 tracking 결과 그리기
- sensitivity_analysis.py
  - Sensitivity Analysis 데이터로 counting 수행
- visualize_sensitivity_analysis_data.py
  - Sensitivity Analysis 영상에 tracking 결과 그리기
- eval_mot_metrics.py
  - counting 수행 이후 MOT 점수 계산

## 실험 결과

### Tracking 성능 평가
**Sensitivity Analysis**

|FPS|추적 알고리즘|탐지 확률 100%|탐지 확률 80%|탐지 확률 60%|탐지 확률 40%|탐지 확률 20%|
|---|:---|---:|---:|---:|---:|---:|
|30|ByteTrack|96.0%|76.0%|54.9%|32.5%|9.5%|
|30|제안 방법|**97.0%**|**77.2%**|**56.8%**|**35.5%**|**13.0%**|
|15|ByteTrack|85.0%|64.6%|42.2%|20.0%|4.7%|
|15|제안 방법|**95.2%**|**74.3%**|**50.2%**|**27.2%**|**6.6%**|
|10|ByteTrack|67.8%|47.6%|28.2%|12.4%|3.1%|
|10|제안 방법|**92.6%**|**65.2%**|**45.5%**|**22.6%**|**4.2%**|
|5|ByteTrack|24.8%|16.4%|9.9%|4.9%|1.3%|
|5|제안 방법|**71.4%**|**50.5%**|**28.1%**|**9.2%**|**2.4%**|


- 제안 방법은 [이전 논문](https://doi.org/10.1016/j.compag.2022.107513)이 제안한 Sensitivity Analysis 실험에서 모든 경우에서 ByteTrack보다 높은 MOTA 점수를 보인다.
  - 다만 이 실험에는 몇 가지 문제점이 존재한다.
  1. 추적 알고리즘의 입력으로 GT 값을 사용한다. 실제 환경에서는 탐지 모델과 추적 알고리즘을 같이 사용해야 하고, 탐지 모델의 FP를 고려할 수 있어야 한다.
  2. SORT 기반 방법의 FN 개수는 입력의 개수에 비례한다. 탐지 비율이 낮아졌을 때 FN이 급격히 증가하여 MOTA 점수가 낮아진다. HOTA 등의 다른 지표를 고려할 필요가 있다.
  3. 탐지 비율이 낮아질 때, 추적 알고리즘의 입력으로 경계 상자를 무작위로 선택한다. 이로 인한 성능 변화가 발생할 수 있다.
  4. 모든 경계 상자의 신뢰도 점수를 1로 취급한다. 1번 문제로 인해 발생한 문제이며, ByteTrack같은 방법을 적용할 수 없다.
- 위의 문제 1번 3번 4번을 해결하기 위해 세 가지 탐지 모델(YOLOv8, Faster-RCNN, EfficientDet)을 사용하여 사과를 탐지하고, 이를 추적 알고리즘의 입력으로 사용하여 실험을 진행하였다.

**YOLOv8**
|FPS|추적 알고리즘|Recall↑|Precision↑|FP↓|FN↓|ID switch 횟수↓|MOTA↑|
|---|:---|---:|---:|---:|---:|---:|---:|
|30|ByteTrack|47.6%|85.3%|41119|262892|480|39.3%|
|30|제안 방법|49.7%|83.7%|48405|252573|440|**39.9%**|
|15|ByteTrack|37.0%|88.8%|11704|158164|1244|31.9%|
|15|제안 방법|47.1%|86.1%|19147|132834|417|**39.3%**|
|10|ByteTrack|26.8%|90.3%|4792|122619|2749|22.3%|
|10|제안 방법|44.4%|87.7%|10419|93095|453|**37.9%**|
|5|ByteTrack|11.8%|90.6%|1025|73959|2763|7.3%|
|5|제안 방법|29.2%|90.9%|2442|59406|2276|**23.6%**|

**Faster R-CNN**
|FPS|추적 알고리즘|Recall↑|Precision↑|FP↓|FN↓|IDs↓|MOTA↑|
|---|:---|---:|---:|---:|---:|---:|---:|
|30|ByteTrack|44.3%|82.2%|48351|279312|757|**34.6%**|
|30|제안 방법|44.1%|80.4%|54033|280685|416|33.2%|
|15|ByteTrack|40.5%|81.8%|22647|149351|2915|30.4%|
|15|제안 방법|43.9%|82.8%|22822|140976|426|**34.6%**|
|10|ByteTrack|36.3%|81.7%|13566|106705|5453|24.9%|
|10|제안 방법|42.9%|84.1%|13537|95597|524|**34.5%**|
|5|ByteTrack|22.1%|81.1%|4326|65350|5739|10.1%|
|5|제안 방법|36.7%|85.0%|5434|53096|2566|**27.2%**|

**EfficientDet**
|FPS|추적 알고리즘|Recall↑|Precision↑|FP↓|FN↓|IDs↓|MOTA↑|
|---|:---|---:|---:|---:|---:|---:|---:|
|30|ByteTrack|47.4%|85.9%|39071|264016|582|39.5%|
|30|제안 방법|48.4%|85.1%|42426|258746|457|**39.9%**|
|15|ByteTrack|40.6%|87.3%|14792|149306|2130|33.8%|
|15|제안 방법|46.9%|87.3%|17221|133354|381|**39.9%**|
|10|ByteTrack|33.4%|87.9%|7668|111510|3914|26.5%|
|10|제안 방법|44.6%|88.7%|9528|92739|551|**38.6%**|
|5|ByteTrack|17.0%|89.0%|1752|69650|3381|10.9%|
|5|제안 방법|33.8%|90.3%|3064|55511|2699|**27.0%**|


↑: 높을수록 좋음
↓: 낮을수록 좋음

- Faster R-CNN을 사용한 30프레임 영상을 제외하고 제안 방법이 ByteTrack보다 높은 성능을 보임


## 참고사항
- ByteTrack 알고리즘에서 [제거한 track이 매칭될 수 있는 오류](https://github.com/ifzhang/ByteTrack/issues/259)를 발견하여 수정 후 사용하였음
- 본 프로젝트에서는 DeepSORT를 사용하지 않았지만, DeepSORT에서 매칭이 되지 않은 track을 결과에 포함하는 경우가 존재함을 발견:
[코드 위치](https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py)