docker -v
if [ $? == 127 ]; then
	apt-get install docker
fi
docker build -t verapak:latest ..
