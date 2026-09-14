# export HF_HOME=//usr2/nluu/.cache/hf_cache/hub/models--GSAI-ML--LLaDA-1.5 
# export HF_HUB_OFFLINE=1

# EPOCHS=150 bash scripts/dlm/tofu/retain.sh llada1.5 0,1,2,3,4,5,6,7 retain100

EPOCHS=150 bash scripts/dlm/tofu/retain.sh llada1.5 0,1,2,3,4,5,6,7 retain95

EPOCHS=150 bash scripts/dlm/tofu/retain.sh llada1.5 0,1,2,3,4,5,6,7 retain99
