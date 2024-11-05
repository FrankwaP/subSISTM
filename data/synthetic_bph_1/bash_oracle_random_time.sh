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
module load R/4.2.0
# on lance le script Modele1.R
R CMD BATCH model_oracle_random_time.R
# c’est fini
echo seff $SLURM_JOB_ID
echo "Date:" ‘date‘
echo "Job finished"