run_server:
	uvicorn ModelDeploy.backend:app --port=30002 --host="0.0.0.0"

run_client:
	python -m streamlit run ModelDeploy/frontend.py --server.address 0.0.0.0 --server.port 30001

run_app: run_server run_client