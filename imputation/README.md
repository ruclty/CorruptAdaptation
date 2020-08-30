# Imputation Methods


## Categorical Attributes(May not work for Mixed-type data)
* [Holoclean](https://github.com/HoloClean/holoclean): A statistical inference engine to impute, clean, and enrich data.

## Mixed-type Attributes
* [GAIN](https://github.com/ruclty/CorruptAdaptation/tree/master/imputation/GAIN) : Impute missing data using Generative Adversarial Networks.
* [MissForest](https://github.com/stekhoven/missForest) : A nonparametric, mixed-type imputation method for basically any type of data.
* MICE : Implemented by mice package in R environment.
    * Install R on Windows: <https://pan.baidu.com/s/1kW-vutRESSgFOkyQ0WJ3Bg>, code: ue5r
        * Firstly run R-4.0.2-win.exe, then install RStudio.
    * Open RStudio
Install mice package:
```sh
> install.packages("mice")
```
Impute missing data with MICE:

```sh
> library(mice)
> miss_data <- read.csv(miss_data_path, na.strings=c(""), stringsAsFactors = TRUE)
> imp <- mice(miss_data, m = 5, method = "pmm")
> imp_data <- complete(imp)
> write.csv(imp_data, file=imp_data_path, sep = ",")
```
* Fill with MEAN or ZERO : Statistical methods that replace the missing values with zero value (i.e., Zero), mean value (i.e., Mean)
