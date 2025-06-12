#!/bin/bash
#SBATCH --job-name=mlp_timelag
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0024498
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu


cd /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger

#for lstm
L5 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/lstm/0611_132323/model_best.pth"
L10 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/lstm/0611_132554/model_best.pth"
L15= "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/lstm/0611_132826/model_best.pth"
L20 ="/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/lstm/0611_133111/model_best.pth"
L25= "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/lstm/0611_133420/model_best.pth"
L30 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/lstm/0611_133750/model_best.pth"

#for transformer
T5 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/transformer/0611_132323/model_best.pth"
T10 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/transformer/0611_132854/model_best.pth"
T15 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/transformer/0611_133736/model_best.pth"
T20 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/transformer/0611_134913/model_best.pth"
T25 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/transformer/0611_140357/model_best.pth"
T30 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/transformer/0611_142147/model_best.pth"

#for mlp
MP5 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/mlp/0611_132323/model_best.pth"
MP10 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/mlp/0611_132550/model_best.pth"
MP15 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/mlp/0611_132818/model_best.pth"
MP20 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/mlp/0611_133047/model_best.pth"
MP25 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/mlp/0611_133315/model_best.pth"
MP30 = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/models/mlp/0611_133547/model_best.pth"

for tlag in 5 10 15 20 25 30; do
    varname="MP${tlag}"
    model_path="${!varname}"
    ~/miniconda3/envs/mesogeos_py38/bin/python test.py --config configs/config_mlp/config_test.json --mp "$model_path"
done