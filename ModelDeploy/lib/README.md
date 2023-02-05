## Deploy MMDeploy Libraries

```bash
mkdir -p /opt && cd /opt
cp ~/detection3d/ModelDeploy/lib/cuda_`uname -a | awk '{print $13}'`/libmmdeploy.zip /opt
unzip libmmdeploy.zip && rm libmmdeploy.zip
echo "/opt/mmdeploy/lib" | tee -a /etc/ld.so.conf.d/mmdeploy.conf && ldconfig
cd ~/detection3d/ModelDeploy/lib
ln -s /opt/mmdeploy/lib libmmdeploy
```
