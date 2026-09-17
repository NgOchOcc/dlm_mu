# EPOCHS=150 bash scripts/dlm/tofu/retain.sh llada1.5 0,1,2,3,4,5,6,7 retain95


conda activate unlearning
cd /prj/corp/airesearch/lasvegas/vol20-scratch/nluu/working/open-unlearning

EPOCHS=150 bash scripts/dlm/tofu/retain.sh llada1.5 0,1,2,3,4,5,6,7 retain99
