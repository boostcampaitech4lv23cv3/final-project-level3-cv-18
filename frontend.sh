# run_frontend.sh
 
#! /bin/bash

if [ "" == "$1" ]; then
    address="0.0.0.0"

else
	address="$1"

fi


if [ "" == "$2" ]; then
    port="30001"

else
	port="$2"

fi

echo "COMMAND : python -m streamlit run ModelDeploy/frontend.py --server.address $address --server.port $port --server.fileWatcherType none"
python -m streamlit run ModelDeploy/frontend.py --server.address $address --server.port $port --server.fileWatcherType none