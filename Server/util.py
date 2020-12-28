import pickle
import json
import numpy as np
import os

__gender = None
__status = None
__education = None
__employment = None
__dependent = None
__property_area = None
__data_columns = None
__model = None


def get_answer(applicantincome, coapplicantincome, loanamount, loanterm, credithistory, gender, status, dependent,
               education, employement, propertyarea):
    loc_index1 = __data_columns.index(gender)
    loc_index2 = __data_columns.index(status)
    loc_index3 = __data_columns.index(dependent)
    loc_index4 = __data_columns.index(education)
    loc_index5 = __data_columns.index(employement)
    loc_index6 = __data_columns.index(propertyarea)
    x = np.zeros(len(__data_columns))
    x[0] = applicantincome
    x[1] = coapplicantincome
    x[2] = loanamount
    x[3] = loanterm
    x[4] = credithistory
    if loc_index1 >= 0:
        x[loc_index1] = 1
    if loc_index2 >= 0:
        x[loc_index2] = 2
    if loc_index3 >= 0:
        x[loc_index3] = 3
    if loc_index4 >= 0:
        x[loc_index4] = 4
    if loc_index5 >= 0:
        x[loc_index5] = 5
    if loc_index6 >= 0:
        x[loc_index6] = 6
    # x.to_csv('x.csv')
    # y = pd.read_csv(x)

    return __model.predict([x])[0]


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __gender
    global __status
    global __dependent
    global __education
    global __employment
    global __property_area
    path = os.path.dirname(__file__)
    artifacts = os.path.join(path, "artifacts"),

    with open(artifacts[0] + "/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __gender = __data_columns[5:7]
        __status = __data_columns[7:9]
        __dependent = __data_columns[9:13]
        __education = __data_columns[13:15]
        __employment = __data_columns[15:17]
        __property_area = __data_columns[17:20]
    global __model
    if __model is None:
        with open(artifacts[0] + "/Loan_model.pickle", 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_gender():
    return __gender


def get_status():
    return __status


def get_dependent():
    return __dependent


def get_education():
    return __education


def get_employment():
    return __employment


def get_property_area():
    return __property_area


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_answer(5000, 500, 200, 1, 1, 'gender_male', 'married_no', 'dependents_0', 'education_graduate',
                     'self_employed_yes', 'property_area_semiurban'))
    print(get_answer(0, 0, 2000000000, 1, 1, 'gender_female', 'married_no', 'dependents_0', 'education_graduate',
                     'self_employed_yes', 'property_area_semiurban'))
