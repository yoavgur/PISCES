# 🪝PISCES - Precise In-parameter Suppression for Concept EraSure
Code for the paper "Precise In-Parameter Concept Erasure in Large Language Models" ([link](https://arxiv.org/pdf/2505.22586)).

<img width="600" alt="Screenshot 2025-05-29 at 14 16 35" src="https://github.com/user-attachments/assets/5f36b9a2-fe02-42f3-b514-7b17405de956" />


<img width="1460" alt="Screenshot 2025-05-29 at 14 16 35" src="https://github.com/user-attachments/assets/434fcccb-78ae-49b9-ac4b-0d9306a618ba" />

### Files
The codebase is still being finalized, but for now I uploaded the main files used in the paper. These are:
- `evals.py` - includes code for running all of our evaluations, as well as wrapper classes for making the API access to `transformers` and `transformer_lens` models identical.
- `feature_finder.py` - used to find the SAE features that will be used to unlearn a concept. Currently there's no demo of this (planning on adding soon), but all of the code is there.
- `editor.py` - this includes the code used to perform the unlearning. The most important function there is `unlearn_concept`, which receives a `Concept` object which includes the relevant features and hyperparameters to unlearn.

We also have a demo for erasing the Harry Potter concept (`erasing_harry_potter.ipynb`), which shows how to use the framework. Note that the editing code currently only works with the `transformer_lens` library.

In `data/cvs.json` you can find all of the data used in all of our evaluations.

### Citation
Please cite as:

```
@misc{gurarieh2025preciseinparameterconcepterasure,
      title={Precise In-Parameter Concept Erasure in Large Language Models}, 
      author={Yoav Gur-Arieh and Clara Suslik and Yihuai Hong and Fazl Barez and Mor Geva},
      year={2025},
      eprint={2505.22586},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22586}, 
}
```


Feel free to contact if you have any thoughts, questions or suggestions :)
