#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=ramboau_install


### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=out_ramboau_install.%J

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=02:00:00

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4000
### Request number of CPUs
#SBATCH --cpus-per-task=3

### Change to the work directory
cd $HPCWORK/RAMBOAU
source $HOME/.zshrc
micromamba env create -f ./install_help/env_man.yml -y
micromamba activate ramboau
rm -rf pymoo
git clone --no-checkout https://github.com/anyoptimization/pymoo.git
cd pymoo
git checkout d8af6a4
python setup.py build_ext --inplace --cythonize
pip install . 
cd ..
pip install pygco
python ./main.py --problem bstvert --algo raqneirs

