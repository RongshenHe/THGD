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
--coarsed_cfg /data2/chensm22/HRS/MHdiff/configs/zinc250k_exp/c_split.yaml \
--coarsed_resume /data2/chensm22/HRS/HierDigress_ckpts/checkpoints/zinc250k/c_charged/NLL44.9.ckpt \
--expanded_cfg /data2/chensm22/HRS/MHdiff/configs/zinc250k_exp/e_split.yaml \
--expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/checkpoints/zinc250k/e_charged/NLL82.9.ckpt \
--device 1 \