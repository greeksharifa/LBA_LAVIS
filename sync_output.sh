# Default
lba_remote_root=${LBA_REMOTE_ROOT:-/home/ywjang/LBA/LBA_Uv2/output}
scp -r -P 7777 "ywjang@147.46.15.69:${lba_remote_root}/." output/
scp -r -P 3491 "ywjang@147.46.215.24:${lba_remote_root}/." output/
scp -r "ywjang@147.46.78.101:${lba_remote_root}/." output/
# IGVLM
igvlm_remote_root=${IGVLM_REMOTE_ROOT:-/home/ywjang/LBA/IG-VLM_LBA}
scp -r -P 7777 "ywjang@147.46.15.69:${igvlm_remote_root}/result_NExTQA_13b_select1_sub_qas_val_xxl_fvu/" output/IGVLM/
