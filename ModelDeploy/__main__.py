# entry point
# from . import main

# args = main.parse_args()
# main.main(args=args)
if __name__ == "__main__":
    import os
    address = "localhost"
    port = 30001
    os.system(f"streamlit run ModelDeploy/app.py --server.address {address} --server.port {port}")