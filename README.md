# Introduction
This is a toy model for PEDS(Physics-enhanced deep surrogates for partial differential equations). The original paper is at https://doi.org/10.1038/s42256-023-00761-y.
This model reimplements the PEDS in Pytorch and solves a 1D diffusion equation to predict the flux. 

# Workflow
All the source codes are included in the `/src` folder. The codes in `/tools` folder provide an interface to train the model and evaluate models' performance.  
## Model Training
1. Modify the `/tools/config.py` file. Both the training and evaluation process will read the hyperparameters from this file.
2. Run `/tools/train_model.py` to train the model. Remember to set `.../Toy_PEDS` as your working directory before running the code. If you are running on a server, you can go to `/sh/train_model.sh` for reference.It's a simple examplary script used on HyperGator.
3. The models' weight will be stored in the directory assigned in `/tools/config.py`.  
## Evaluation  
After models' weights are saved, you could directly use `/tools/evaluation.ipynb` to test model's performance. 
