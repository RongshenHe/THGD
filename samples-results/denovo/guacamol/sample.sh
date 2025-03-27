base_path=/data2/chensm22/HRS/MHdiff/src/
coarsed_path=./coarsed_graphs.pkl
keep_chain=5
chain_step=5
save_final=5

python ${base_path}sample.py -nb 60 -b 512 -k $keep_chain -c $chain_step -s $save_final -r 1 \
--skip_t 1 \
--save_graph True \
--coarsed_path $coarsed_path \
--sample_type all \
--coarsed_cfg /data2/chensm22/HRS/MHdiff/configs/guacamol_exp/c_custom.yaml \
--coarsed_resume /data2/chensm22/HRS/HierDigress_ckpts/checkpoints/guacamol/c_custom_ring/NLL=59.8.ckpt \
--expanded_cfg /data2/chensm22/HRS/MHdiff/configs/guacamol_exp/e_split_charged.yaml \
--expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/checkpoints/guacamol/charged/NLL=94.6.ckpt \
--device 0 \