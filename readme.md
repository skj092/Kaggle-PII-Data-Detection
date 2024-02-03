# PII Data Detection

**DataSet Provided**: 1. `train.json` 2. `test.json` 3. `sample_submission.csv`

**About `train.csv` file**
1. It is a list of 6807 objects.
2. Each Object contain 5 keys:
    1. `document` : id of the essay (int)
    2. `full_text`: Easay text (string)
    3. `tokens`: Full_text converted in tokens
    4. `trailing_whitespace`: Whether there is a whitespace in the full_text after the token
    5. `labels`: Label for each token

**PII Types**

1. There are 7 types of label that need to be assigned to each token.
    1. `NAME_STUDENT`:
    2. `EMAIL`
    3. `USERNAME`
    4. `ID_NUM`
    5. `PHONE_NUM`
    6. `URL_PERSONAL`
    7. `STREET_ADDRESS`

