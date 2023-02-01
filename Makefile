run_server:
	uvicorn ModelDeploy.backend:app --port=30002 --host="172.17.0.2"

run_client:
	python -m streamlit run ModelDeploy/frontend.py --server.address localhost --server.port 80

run_app: run_server run_client