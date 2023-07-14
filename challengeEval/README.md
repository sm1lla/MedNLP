# NTCIR'17 Social Media Subtask
## Adverse Drug Event detection for social media texts in Japanese, English, German, and French

The following measure are used, mostly based on the [scikit-learn library](https://scikit-learn.org/stable/index.html):


1) **Binary**: Calculates the performance of classifying a document into the classes "contains ADE" (positive) vs. "does not contain ADE" (negative). A document is considered to contain an ADE if a least one symptom (class) is positive (1). The most interesting scores in this case are precision, recall and F1 for the *positive* class.

2) **Per class**: Calculates precision, recall and F1 score for each class (symptom). This is useful to see if there are any differences in how systems detect different symptoms (individual scores per class).

3) **(Full) per label**: Calculates precision, recall and F1 score for each *label* (0 and 1) across samples and classes.

4) **Exact match accuracy**: Calculates the percentage of exact matches across all samples. The system has to predict a perfect labeling of a sample; as soon as one symptom is not correctly predicted, the sample will not be counted.



The predictions file is expected to have the exact same structure as the train data file.


| test_id   | text      | C0027497:nausea   | C0011991:diarrhea     | C0015672:fatigue  | ...   |
|---------  |---------- |-----------------  |-------------------    |------------------ |-----  |
| 1058      | Tweet 1   | 1                 | 1                     | 1                 | ...   |
| 1120      | Tweet 2   | 0                 | 1                     | 0                 | ...   |
| 2770      | Tweet 3   | 0                 | 0                     | 0                 | ...   |
| 2250      | Tweet 4   | 0                 | 1                     | 0                 | ...   |
| 9217      | Tweet 5   | 1                 | 1                     | 1                 | ...   |
| 1444      | Tweet 6   | 1                 | 1                     | 1                 | ...   |
| 6771      | Tweet 7   | 0                 | 1                     | 1                 | ...   |
| 8845      | Tweet 8   | 0                 | 0                     | 0                 | ...   |
| 8212      | Tweet 9   | 1                 | 0                     | 0                 | ...   |
| 9116      | Tweet 10  | 0                 | 1                     | 1                 | ...   |
| 9271      | Tweet 11  | 0                 | 0                     | 0                 | ...   |
| 534       | Tweet 12  | 0                 | 0                     | 0                 | ...   |


Install the necessary libraries and run the script like so:

```shell
pip install -r requirements.txt

python evaluate_ntcir_2023.py -gold_file samples/sample_gold.csv -prediction_file samples/sample_predicted_1.csv
```
Please make sure to provide the correct format for your predictions. You can find mock examples in `samples/`.


The script run on samples/sample_predicted_1.csv should return the following:

```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Binary Scores (ADE vs. no ADE):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              precision    recall  f1-score   support

      no ADE       0.75      0.75      0.75         4
         ADE       0.88      0.88      0.88         8

    accuracy                           0.83        12
   macro avg       0.81      0.81      0.81        12
weighted avg       0.83      0.83      0.83        12



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Individual) Per Class Scores:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   precision    recall  f1-score   support

  C0027497:nausea       0.75      0.50      0.60         6
C0011991:diarrhea       0.71      0.83      0.77         6
 C0015672:fatigue       0.40      0.67      0.50         3

        micro avg       0.62      0.67      0.65        15
        macro avg       0.62      0.67      0.62        15
     weighted avg       0.67      0.67      0.65        15
      samples avg       0.42      0.46      0.43        15



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(Full) Per Label Scores:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              precision    recall  f1-score   support

           0       0.75      0.71      0.73        21
           1       0.62      0.67      0.65        15

    accuracy                           0.69        36
   macro avg       0.69      0.69      0.69        36
weighted avg       0.70      0.69      0.70        36


Exact accuracy: 0.5833333333333334
```

If you add `--csv_output` to the evaluation command, the script will create two CSVs containing the counts for true positives (TPs), false positives (FPs), true negatives (TNs), and false negatives (FNs) for the binary and per-class case.
