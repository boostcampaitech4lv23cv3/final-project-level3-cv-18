# run_backend.sh
 
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

	
echo "COMMAND : uvicorn ModelDeploy.backend:app --port=$port --host=$address"
uvicorn ModelDeploy.backend:app --port=$port --host=$address