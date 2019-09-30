# enriched_LSTMs
Getting more out of LSTMs for classifying multimodal health data

### Background

### Data
We used emergency department (ED) visit record data to develop our models. The records had one free-text field, chief complaint, along with a number of other discrete variables, like age group, sex, mode of arrival, and hospital code. This is the third project we've done with the data, so if you're interested in learning more about them, check out our papers about 

### Code
Example preprocessing run:
```python preprocessing.py --data_dir=C:/data/syndromic/ ^
--input_file=sample.csv ^
--file_type=csv ^
--text_column=cc ^
--clean_text=True ^
--convert_numerals=True ^
--target_column=ccs
```

And an example training and test run:
```python train_and_test.py --data_dir=C:/data/syndromic/ ^
--text_file=word_sents.hdf5 ^
--records_npz=sparse_records.npz ^
--records_csv=sample.csv ^
--target_column=ccs ^
--patience=1
```
