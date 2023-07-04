import pandas as pd
import numpy as np
from view import plot_distributions, dict_bins
import pandas as pd
import numpy as np
from view import plot_distributions, translate_column_names, correlation_matrix

'''
Reading and reprocessing start data
'''
df_all_values_patients = pd.read_excel(r'data/test.xlsx')
df_all_values_patients = df_all_values_patients.drop(index=119)
df_all_patients_grade = df_all_values_patients.loc[:, [
                                                    'Возраст',
                                                    'ИМТ',
                                                    'Давность заболевания',
                                                    'Количество обострений за последний год',
                                                    'IgE',
                                                    'СРБ ',
                                                    'липиды',
                                                    'Глюкоза на момент поступления',
                                                    'Эозинофиллия',
                                                    'Бронхиальная астма'
                                                ]
                                            ]


with pd.ExcelWriter('result/result.xlsx') as writer:
    df_all_patients_grade.to_excel(writer, sheet_name='data_grade', index=False)