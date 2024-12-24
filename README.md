To install dependencies for the pgm project use the conda file ```env.yaml```
```
conda env create -f env.yaml
conda activate diffuser
pip install -e .
```
The new files for the project 
```
srcipts/train_hf.py
scripts/train_hf_value.py
srcipts/plan_guided_hf.py
scripts/eval_hf_value.py
scripts/eval_unguided_hf.py
scripts/bc_d4rl.py
```

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

Our implementation is based on the original [Diffusser codebase](https://github.com/jannerm/diffuser).
