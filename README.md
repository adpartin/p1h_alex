### Pilot1 Hackathon (alex)
This repo was cloned from https://github.com/levinas/p1h  
A few modifications have been made:  
- data pre-processing for structured data (e.g., descriptors)  
- processing SMILES strings using neural networks

#### Command-line examples
By-cell drug response prediction using SMILES strings.
```bash
python by_cell_nn_smiles.py --file BR:MCF7_smiles.csv --epochs 100 --batch 128 --optimizer adam
```
By-cell drug response prediction using descriptors.
```bash
python by_cell.py --models xgboost --thres_frac_rows 0.95 --thres_frac_cols 0.97 --thres_var 0 --thres_corr 0.99 --thres_discrete 2 --create_iom yes --min_growth_bound -1 --max_growth_bound 1
```
