docker build -t gmps:latest .

docker run --rm -it gmps:latest /bin/bash -c "pushd /root/playground/GMPS; git pull origin master; ls; python3 launchers/remote_train.py; python3 launchers/remote_train.py"
