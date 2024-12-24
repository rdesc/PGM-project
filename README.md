To install dependencies for the pgm project use the conda file ```env.yaml```
```
conda env create -f env.yaml
conda activate diffuser
pip install -e .
```
The new files for the project are located in the scripts folder:
```
bc_d4rl.py
dan_notes.md
eval_diffuser.sh
eval_hf_value_debug.py
eval_hf_value_diffuser.py
eval_hf_value_dmpc.py
eval_hf_value_dmpc_constraint.py
eval_hf_value_dmpc_height.py
eval_hf_value_dmpc_unet.py
eval_unguided_hf.py
plan_guided_df_notebook.ipynb
plan_guided_diff.py
plan_guided_hf.py
plotting
run_janner_eval.sh
train
train_diffuser.sh
train_hf.py
train_hf_action_transformer.py
train_hf_action_unet.py
train_hf_dynamics_transformer.py
train_hf_dynamics_unet.py
train_hf_transformer.py
train_hf_value.py
transformer_1d.py
transformer_block.py
value_guided_sampling.py
```
Particularly, [dan_notes](https://github.com/rdesc/PGM-project/blob/main/scripts/dan_notes.md) contains most of the commands we used to train and evaluate different models. 

### This was the state of our project:
<p align="center">
<!-- ![flop](https://github.com/user-attachments/assets/2bfaa11b-1a30-4935-af07-0b11bd46111f) -->
 <img src="https://github.com/user-attachments/assets/2bfaa11b-1a30-4935-af07-0b11bd46111f" width="60%" title="Diffuser model">

</p>

### Now the robots are happy:
<p align="center">
<img src="https://github.com/user-attachments/assets/44fb6648-b642-42a7-a25f-ef13844cd6a4"  width="60%" >
</p>



## Acknowledgements

Our implementation is based on the original [Diffusser codebase](https://github.com/jannerm/diffuser), and we reference Hugging Face Diffuser's [value-guided planning](https://huggingface.co/docs/diffusers/v0.16.0/en/using-diffusers/rl).
