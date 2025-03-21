batch_size=1024
keep_chain=5
chain_step=5
save_final=5
sample_per_graph=10
coarsed_path=./coarsed_graphs.pkl
base_path=/data2/chensm22/HRS/DiGress/src/

# python sample.py -nb 1 -b 128 -k $keep_chain -c $chain_step -s 40 -r 3 \
# --sample_type cond \
# --save_graph True \
# --coarsed_path $coarsed_path \
# --coarsed_cfg /data2/chensm22/HRS/HierDigress_ckpts/moses/coarsed_mose_unconstrain/c_comm_unconstrained.yaml \
# --coarsed_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/coarsed_mose_unconstrain/epoch=71.ckpt \
# --sample_per_graph 4 \
# --skip_t 2 \
# --target_nodes 40 \
# --extra_nodes 5 \
# --expanded_cfg /data2/chensm22/HRS/HierDigress_ckpts/moses/expanded_mose_unconstrain/e_comm_unconstrained.yaml \
# --expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/expanded_mose_unconstrain/epoch=69.ckpt \
# --device 1 \

# python sample.py -nb 1 -b 128 -k $keep_chain -c $chain_step -s 40 -r 3 \
# --sample_per_graph 4 \
# --skip_t 2 \
# --sample_type from_coarse \
# --coarsed_path $coarsed_path \
# --expanded_cfg /data2/chensm22/HRS/HierDigress_ckpts/moses/expanded_mose_unconstrain/e_comm_unconstrained.yaml \
# --expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/expanded_mose_unconstrain/epoch=69.ckpt \
# --device 1 \

# --scaffold "C1C=CNC2=CC=CC=C21"
# --scaffold "CC1=CN(C2=CC=CC=C2C1=O)C"
# --scaffold "C1=CC(=CC(=C1)N)C(F)(F)F"
# --scaffold "C1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)))C"



# python sample.py -nb 1 -b 128 -k $keep_chain -c $chain_step -s 40 -r 3 \
# --sample_type all \
# --sample_per_graph 1 \
# --skip_t 2 \
# --scaffold "C1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)))C" \
# --save_graph True \
# --coarsed_path $coarsed_path \
# --coarsed_cfg /data2/chensm22/HRS/HierDigress_ckpts/guacamol/c_custom_ring/c_custom.yaml \
# --coarsed_resume /data2/chensm22/HRS/HierDigress_ckpts/guacamol/c_custom_ring/epoch=113.ckpt \
# --expanded_cfg /data2/chensm22/HRS/HierDigress_ckpts/guacamol/e_custom_ring/e_custom.yaml \
# --expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/guacamol/e_custom_ring/epoch=52.ckpt \
# --device 0 \

# python sample.py -nb 1 -b 256 -k $keep_chain -c $chain_step -s 40 -r 3 \
# --sample_type score \
# --skip_t 2 \
# --smiles "Cc1coc(C)c1C(=O)CC1(O)C(=O)N(C)c2ccccc21" \
# --expanded_cfg /data2/chensm22/HRS/DiGress/configs/mose_exp/e_custom.yaml \
# --expanded_resume /data2/chensm22/HRS/DiGress/outputs/2024-11-05/01-28-34-expanded_mose_custom/checkpoints/expanded_mose_custom_resume/epoch=29.ckpt \
# --device 0 \

python ${base_path}sample.py -nb 10 -b 512 -k $keep_chain -c $chain_step -s 5 -r 1 \
--skip_t 1 \
--save_graph True \
--coarsed_path $coarsed_path \
--sample_type all \
--coarsed_cfg /data2/chensm22/HRS/DiGress/configs/mose_exp/c_custom.yaml \
--coarsed_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/c_custom_ring/epoch=197.ckpt \
--expanded_cfg /data2/chensm22/HRS/DiGress/configs/mose_exp/e_custom.yaml \
--expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/e_custom_ring/epoch=189.ckpt \
--device 1 \


# python sample.py -nb 2 -b 512 -k $keep_chain -c $chain_step -s 40 -r 3 \
# --skip_t 2 \
# --sample_type expand \
# --expanded_cfg /data2/chensm22/HRS/DiGress/configs/mose_exp/e_custom.yaml \
# --expanded_resume /data2/chensm22/HRS/DiGress/outputs/2024-11-05/01-28-34-expanded_mose_custom/checkpoints/expanded_mose_custom/last.ckpt \
# --devices 1 \


# python sample.py -nb 4 -b 512 -k $keep_chain -c $chain_step -s 20 -r 3 \
# --skip_t 2 \
# --sample_type e_guidance \
# --coarsed_cfg /data2/chensm22/HRS/DiGress/configs/mose_exp/c_custom.yaml \
# --coarsed_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/c_custom/epoch=139.ckpt \
# --expanded_cfg /data2/chensm22/HRS/DiGress/configs/mose_exp/e_custom.yaml \
# --expanded_resume /data2/chensm22/HRS/HierDigress_ckpts/moses/e_custom/epoch=63.ckpt \
# --guidance_resume /data2/chensm22/HRS/HierDigress_ckpts/guidance/moses/epoch=19.ckpt \
# --device 0 \