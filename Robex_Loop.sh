cd /Working_Dir/

for f in /InputData/* ; do
	./ROBEX/runROBEX.sh /ROBEX_InputData/$f /ROBEX_OutputData/$f ;
	done
