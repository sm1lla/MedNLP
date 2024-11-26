# MedNLP: Social Media Adverse Drug Event Detection

This project is a contribution to the **Social Media Adverse Drug Event Detection** track of the **NTCIR-17 MedNLP-SC** shared task which deals with multi-lingual adverse drug event detection. We worked with a dataset of social media texts in 4 different languages (English, German, French and Japanese). Some of these texts mention different side effects the authors had from drugs they took. We frame ADE (adverse drug event) detection problem as a multi-label classification task, where the input is a text snippet and the output consists of labels for 22 ADE classes.

For more information about the task and the concrete process please refer to our [paper](http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings17/pdf/ntcir/09-NTCIR17-MEDNLP-FoxS_slides.pdf).


## Approach 🧠
We use Huggingface to finetune a  [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) model. To improve the results we work did hyperparameter tuning and tried augmentation and ensemble learning.

## Setup 🛠️
To run the code you need to setup an environment with the packages in `environment.yaml`. When using conda this can be done with `conda env create -n <name> --file environment.yml`.

## Usage 👩‍💻
Fine a model using:
```
python -m src task=train
```
Run evaluation:
````
python -m src task=evaluate
````
Compute predictions on dataset:
````
python -m src task=predict
````
Analyze distibution of dataset:
````
python -m src task=dataset
````

Train an ensemble:
````
python -m src task=ensemble
````

Generate additional data with GPT model:
````
python -m src task=generate
````

Translate generated data:
````
python -m src task=translate
````

