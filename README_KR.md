# Copy-paste Augmentation for Nodule Detection

[English](https://github.com/seoulsky-field/copy-paste-nodule-detection/blob/main/README.md)  |  Korean


## 프로젝트 소개
해당 프로젝트는 NeurIPS ML4H 2022 참여에 사용된 프로젝트입니다.  

의료 데이터의 특징 중 하나는 data imbalanced가 심한 점을 고려하여 의료 영상 데이터의 폐에 존재하는 nodule을 복사 후 Soft Augmentation을 거쳐 정상 영상과 nodule이 존재하는 영상 모두에 최소 1개에서 최대 3개를 붙여 넣어 효과를 확인하고자 하였습니다.  

## 환경 설정
**Step 1.** Anaconda3를 다운 받고 가상 환경을 생성합니다.
```shell
conda create -n env_name python=3.8
```
**Step 2.** 필요한 라이브러리를 다운받습니다. (pytorch는 [공식 홈페이지](https://pytorch.org/get-started/locally/)를 참조하는 것을 권장합니다.)
```shell
conda install --yes --file requirements.txt
conda install pytorch=1.12.0 cudatoolkit=11.3 -c pytorch
```
**Step 3.** [MMDetection library](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)를 다운받습니다.
```shell
pip install openmim
mim install mmcv-full
```

## 결과

## 참여자
[김이삭](https://github.com/yisakk)(서울대학교병원 영상의학과, 서울대학교 박사과정)  
[전경민](https://github.com/seoulsky-field)(서울대학교병원 영상의학과, 단국대학교)  
[서다빈](https://github.com/sodabeans)(서울대학교병원 영상의학과, 고려대학교)  
[이주환](https://github.com/JHwan96)(서울대학교병원 영상의학과, 숭실대학교)

Detection 베이스라인은 [MMDetection Library](https://github.com/open-mmlab/mmdetection)를 이용하였습니다.

## 참고 문서
