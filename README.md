# [CV-18] Light Observer
# 초보 운전자를 위한 안전 주행 보조 시스템

Notion: [Notion link](https://www.notion.so/CV-18-Light-Observer-6ac0befae87240198bee1e0ea5cb8b21)    
발표 영상: [Youtube](https://youtu.be/Yp4-nnwkreA)  
테크 스택: 3D Object Detection, FFmpeg, FastAPI, ONNX, On-Device AI, PyTorch, Streamlit, TensorRT    

# 프로젝트 한줄 소개

🚙 초보 운전자를 위한 **안전 주행 보조 시스템**은 전방에서 갑자기 끼어드는 차량을 **Monocular 3D Object Detection**을 이용하여 상대 차량과의 **거리**와 **각도**를 인식하고 **위험도**를 예측하여 주행자가 안전하게 대처할 수 있도록 알려주는 시스템입니다.


# 팀원 소개 ([Team Notion](https://www.notion.so/Level2-cv-18-shared-17da07e49fa7487792ba918be6007fd9))

| 이름 | 역할 | 링크 |
| --- | --- | --- |
| 김형석 | Data Analysis, Coordinate Converting, Visualization(3D-2D Projection, BirdeyeView), Inference Engine, Web Demo(Streamlit&FastAPI), Model Train & Inference, Building a development environment(Server) | [github](https://github.com/404Vector), [notion](https://tiryul.notion.site), [linkdein](https://www.linkedin.com/in/hyeongseok-kim-a280841b9/) |
| 이동훈 | Data Analysis & Converting, Coordinate Converting, Visualization(Bird Eyes View), Model Train & Inference, Model Research, Presentation |  |
| 전지용 | Data Analysis & Converting, Coordinate Converting, Visualization(Danger Object), Inference Engine, Web Demo(Streamlit&FastAPI), Model Train & Inference, Presentation | [notion](https://www.notion.so/30ccaa36276d458ab0a8b1b06aab3c13), [github](https://github.com/Jiyong-Jeon) |
| 정원국 | Data Analysis & Converting, Coordinate Converting, Visualization(Bird Eyes View), Model Train & Inference, Model Research, Presentation |  |
| 한상준 | Model conversion, Inference Engine, App Demo(tkinter), Model Train & Inference, Building a development environment(Server, Edge Device) | [linkedin](https://www.linkedin.com/in/jphan32/) |

---

# 프로젝트 데모

### **Web Demo**

![Left : only KITTI dataset  /  Right : Our Model(KITTI + Finetuning)](contents/ezgif-2-749e24f09f.gif)

### **Edge Device(Jetson Xavier)**

![xavier_AdobeExpress.gif](contents/xavier_AdobeExpress.gif)

---
# Document
- [프로젝트 소개](docs/introduce.md)
- 설치 방법