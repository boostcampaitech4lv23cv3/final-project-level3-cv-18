# ModelDeploy

Deploy된 Model과 Model Weight를 갖고 Inference를 수행하고 결과를 시각화합니다.



## Run - ModelDeploy

주의 : 환경 구성이 모두 완료 되어야 합니다. [[Install](Install.md)]

### Run Frontend(streamlit)

```bash
./frontend.sh
```

or

```bash
python -m streamlit run ModelDeploy/frontend.py --server.address $address --server.port $port --server.fileWatcherType none
```

### Run Backend(FastAPI)

```bash
./backend.sh
```

or

```bash
uvicorn ModelDeploy.backend:app --port=$port --host=$address
```

## Make - Asset

Inference를 하기 위해서는 Asset 파일이 필요합니다.

Asset의 생성 방법은 다음과 같습니다.

[todo]
