
rsync -avh  \
	--exclude '.git' --exclude '__pycache__'  \
	--exclude '.Rproj.user'  /home/unimelb.edu.au/nbloomfield/Desktop/phd-data/funnybirds/FunnyBirds nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/data/


