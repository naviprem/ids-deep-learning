Metrics: 

#### True Positives (TP)  
TP is the number of correctly classified attacks.

#### True Negatives (TN) 
TN is the number of correctly classified normal records.

#### False Positives (FP)
FP is the number of misclassified attacks. 

#### False Negatives
FN denotes the number of misclassified normal records. 

#### Accuracy
The accuracy is the percentage of the correctly classified records over all the rows of data set.

```bash
Accuracy = (TP + TN) / Total number of Records
```

#### Precision

Precision tells us what proportion of records we classified as attack, actually were attacks.

```bash
TP / (TP + FP)
```

#### Recall (sensitivity)

Recall tells us what proportion of records that actually were attacks were classified by 
the model as attacks

```bash
TP / (TP + FN)
```

#### False Positive Rate (FPR)

A false positive error, is a result that indicates a given condition exists, when it does not.

```bash
False Positive Rate (FPR) = FP/ (FP + TN)
```

#### False Negative Rate (FPR)
A false negative error, is a prediction that indicates that a condition does not hold, 
while in fact it does. 

```bash
False Negative Rate (FNR) = FN/ (FN + TP)
```

#### The False Alarm Rate (FAR) 
FAR reflects the rate of misclassified records to correctly classified records. 

```bash
FAR = (FPR + FNR) / 2 
```
The highest trusted detection is accomplished, when the accuracy value closes to 100% and FAR closes to 0%. 

## References

https://en.wikipedia.org/wiki/False_positive_rate

https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal