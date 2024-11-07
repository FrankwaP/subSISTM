#!/bin/bash
#############################
# les directives Slurm vont ici:
# nom du job (apparait dans squeue)
#SBATCH -J M1
# temps r´eserv´e (hh:mm::ss)
#SBATCH -t 10:00:00
# nombre de coeurs n´ecessaires
#SBATCH --nodes=1
#SBATCH --ntasks=10
# se placer dans le r´epertoire de soumission
#SBATCH --chdir=.
# Nom à donner au fichier de sorti, reprenant les sorties python.
#SBATCH --output = slurm_script_RNN.py
# fin des directives
#############################
# informations qui seront ecrites dans slurm-xxx.out
echo "#############################"
echo "User:" $USER
echo "Date:" ‘date‘
echo "Host:" ‘hostname‘
echo "Directory:" ‘pwd‘
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################"
#############################
# on charge R
module load python/3.12.7
# Lancement du script : 

# Chargement de mon environnement python
source /gpfs/home/pierlotr/python/modele_homogene/python_env/bin/activate
# c’est fini
echo seff $SLURM_JOB_ID

python3.12.7 RNN_all_sim.py

deactivate

# écrire quand le script est fini
echo "Date:" ‘date‘
echo "Job finished"
