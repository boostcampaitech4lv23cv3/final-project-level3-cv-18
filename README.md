# [CV-18] Light Observer
# 초보 운전자를 위한 안전 주행 보조 시스템

**Notion**: [Notion link](https://www.notion.so/CV-18-Light-Observer-6ac0befae87240198bee1e0ea5cb8b21),  **발표 영상**: [Youtube](https://youtu.be/Yp4-nnwkreA)  
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/> <img src="https://img.shields.io/badge/TensorRT-FF6F00?style=flat&logo=TensorFlow&logoColor=white"/> 

# 프로젝트 한줄 소개

🚙 초보 운전자를 위한 **안전 주행 보조 시스템**은 전방에서 갑자기 끼어드는 차량을 **Monocular 3D Object Detection**을 이용하여 상대 차량과의 **거리**와 **각도**를 인식하고 **위험도**를 예측하여 주행자가 안전하게 대처할 수 있도록 알려주는 시스템입니다.


# Light Observer 팀원 소개 ([Team Notion](https://www.notion.so/Level2-cv-18-shared-17da07e49fa7487792ba918be6007fd9))
<table align="center">
    <tr height="160px">
        <td align="center" width="200px">
            <a href="https://github.com/404Vector"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/39119364?v=4"/></a>
            <br />
            <a href="https://github.com/404Vector"><strong>김형석</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/teedihuni"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/66379349?v=4"/></a>
            <br/>
            <a href="https://github.com/teedihuni"><strong>이동훈</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/Jiyong-Jeon"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/68497156?v=4"/></a>
            <br/>
            <a href="https://github.com/Jiyong-Jeon"><strong>전지용</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/jungwonguk"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/98310175?v=4"/></a>
            <br />
            <a href="https://github.com/jungwonguk"><strong>정원국</strong></a>
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/jphan32"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/7111986?v=4"/></a>
            <br/>
            <a href="https://github.com/jphan32"><strong>한상준</strong></a>
            <br />
        </td>
    </tr>
    <tr height="40px">
        <td align="center" width="200px">
            <a href="https://www.linkedin.com/in/hyeongseok-kim-a280841b9/">Linkedin</a>
            <br/>
            <a href="https://tiryul.notion.site">Notion</a>
            <br/>
        </td>
        <td align="center" width="200px">
            <a></a> 
        </td>
        <td align="center" width="200px">
            <a href="https://jiyong-jeon.notion.site/Jeon-Jiyong-30ccaa36276d458ab0a8b1b06aab3c13">Notion</a>
            <br/>
        </td>
        <td align="center" width="200px">
            <a></a> 
        </td>
        <td align="center" width="200px">
            <a href="https://www.linkedin.com/in/jphan32/">Linkedin</a>
            <br/>
        </td>
    </tr>
</table>

- 김형석
  - Data Analysis, Coordinate Converting, Visualization(2D-3D projection), Inference Engine, Web Demo, Model Train & Inference, Environment(Server)
- 이동훈
  - Data Analysis & Converting, Coordinate Converting, Visualization(Bird Eyes View), Model Train & Inference, Model Research, Presentation
- 전지용
  - Data Analysis & Converting, Coordinate Converting, Visualization(Danger Object), Inference Engine, Model Train & Inference, Web Demo, Presentation
- 정원국
  - Data Analysis & Converting, Coordinate Converting, Visualization(Bird Eyes View), Model Train & Inference, Model Research, Presentation
- 한상준
  - Model conversion, Inference Engine, App Demo(tkinter), Model Train & Inference, Environment(Server, Edge Device)
---

# 프로젝트 데모

### **Web Demo**

![Left : only KITTI dataset  /  Right : Our Model(KITTI + Finetuning)](contents/ezgif-2-749e24f09f.gif)

### **Edge Device(Jetson Xavier)**

![xavier_AdobeExpress.gif](contents/xavier_AdobeExpress.gif)

---
# Document
- [프로젝트 소개](docs/introduce.md)
- Getting_started
  - [설치 방법](docs/Install.md)
  - [데이터 변환](KITTIVisualizer/Auto_transform.ipynb)
  - [PyTorch Model 변환](docs/PyTorch-Model-Convert.md)
- Demo
  - [How to Use ModelDeploy](docs/How-to-Use-ModelDeploy.md)
  - [How to build PyTorch for Jetson Xavier](docs/How-to-build-PyTorch-for-Jetson-Xavier.md)

---
# 폴더 구조
- KITTIVisualizer : 데이터 변환을 위한 폴더
- ModelDeploy : Model Serving을 위한 폴더 (Web Demo, App)
- assets : Web & App demo시 데이터 폴더
- mmdeploy : Model 변환을 위한 MMdeploy Custom 코드
- mmdetection3d : Train & Inference 위한 MMdet3d
- mmdetection3d custom : MMdetection3d에서 우리의 Custom 코드

---
# 관련 파일
- Xavier 환경 설치 패키지 모음
  - [Drive](https://drive.google.com/drive/folders/1KDf4QP_W_I_a-1GQIcSRUPVgD4dOtAt5?usp=share_link)
- 학습 모델 및 변환 모델 모음
  - [Drive](https://drive.google.com/drive/folders/1FEtoi_wX-5qFwhzOmZHGML4ygLYT2Y95?usp=share_link)
