docker -v
if [ $? == 127 ]; then
	apt-get -y install docker
fi
docker build -t verapak:latest ..
