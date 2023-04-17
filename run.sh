docker stop verapak && docker rm verapak && docker build --tag "verapak:latest" .
docker run --privileged --name verapak -v $PWD:/src/in -v $PWD/out:/src/out -v /var/run/docker.sock:/var/run/docker.sock -dit verapak:latest /bin/sh
