## Data Sets

### Adult
[Adult](https://archive.ics.uci.edu/ml/datasets/Adult.) contains personal information from the 1994 US census. Each record in the dataset corresponds to a person with mixed data types, i.e., 8 categorical and 6 numerical attribute values.
The attribute income is used as the label to predict whether the person has income larger than 50K per year (positive) or not (negative).

### EyeState
[EyeState](http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State.) contains records obtained from one continuous EEG (electroencephalogram) measurement with the Emotiv EEG Neuroheadset. Specifically, each record has multiple attributes
corresponding to various metrics measured by the Neuroheadset, and all the attributes in the dataset are numerical. Moreover, a label corresponding to the eye state is associated with each record, which is one of two possible values, 1 (eye-closed) and 0 (eye-open).

### Ipums
[Ipums](https://www.openml.org/d/381) is the Public Use Microdata Sample (PUMS) census data from the Los Angeles and Long Beach areas. Following the common practice in ML, we partition the dataset into source and target based on the timestamp and train a classifier for predicting attribute MovedIn. In particular, we use the tuples in 1998 as the source and tuples in 1999 as the target. The source has 17 attributes with missing rates ranging from 1.8% to 71.0%, and the target has 18 attributes with missing rates ranging from 1.9% to 69.8%.

### Okcupid
[Okcupid](https://www.openml.org/d/41440) contains user profile data for San Francisco OkCupid users. It includes people within a 25 mile radius of San Francisco, who were online in 2011. We train a classifier to predict the job of a user from three categories, STEM, non-STEM and Student, where STEM stands for jobs in computer/hardware/software/science/tech/engineering. Like the Ipums dataset, we use the data with last online time before 2012-06-27 18:04 as the source
data, and use the rest as the target data. The source has 13 attributes with missing rates ranging from 0.1% to 74.8%, and the target has 13 attributes with missing rates ranging from 0.04% to 80.1%.
