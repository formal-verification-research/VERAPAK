docker -v
if [ $? == 127 ]; then
	sudo apt install apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
	apt-cache policy docker-ce
	sudo apt -y install docker-ce
fi
mkdir verapak
mkdir verapak/in
mkdir verapak/out
docker stop verapak
docker rm verapak
docker run --name verapak -v "$PWD/verapak":/mnt -dit yodarocks1/verapak:latest /bin/sh