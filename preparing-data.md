## Preparing the Data
Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, 
and restructured. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, 
however, there are some qualities about certain features that must be adjusted. 
This preprocessing can help tremendously with the outcome and predictive power of nearly 
all learning algorithms.

### Transforming Skewed Continuous Features
A dataset may sometimes contain at least one feature whose values tend to lie near a single number, 
but will also have a non-trivial number of vastly larger or smaller values than that single number.  
Algorithms can be sensitive to such distributions of values and can underperform if the range is not 
properly normalized.  
