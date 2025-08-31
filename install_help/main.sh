#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=ramboau

### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=optimization-out_ramboau.%J

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=4:00:00

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=2000

### Request number of CPUs
#SBATCH --ntasks=12


### Change to the work directory
cd $HOME/MA/RAMBOAU
source $HOME/.zshrc
micromamba activate ramboau

python ./run.py $@