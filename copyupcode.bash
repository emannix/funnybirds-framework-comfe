
rsync -avh  \
	--exclude '.git' --exclude '__pycache__'  \
	--exclude '.Rproj.user'  . nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/cloned_repos/funnybirds-framework-comfe


