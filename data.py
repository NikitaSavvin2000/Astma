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
df_all_patients = df_all_values_patients.loc[:, [
                                                    'Возраст',
                                                    'ИМТ',
                                                    'Давность заболевания',
                                                    'Количество обострений за последний год',
                                                    'IgE',
                                                    'СРБ ',
                                                    'липиды',
                                                    'Глюкоза на момент поступления',
                                                    'Эозинофиллия'
                                                ]
                                            ]
df_all_patients.astype(float)
df_all_patients = translate_column_names(df_all_patients)

df_mean_ill_all = df_all_values_patients.loc[df_all_values_patients['Бронхиальная астма'] == 'средней тяжести']
df_mean_ill_all = df_mean_ill_all.loc[:, [
                                                    'Возраст',
                                                    'ИМТ',
                                                    'Давность заболевания',
                                                    'Количество обострений за последний год',
                                                    'IgE',
                                                    'СРБ ',
                                                    'липиды',
                                                    'Глюкоза на момент поступления',
                                                    'Эозинофиллия'
                                                ]
                                            ]
df_mean_ill_all.astype(float)
df_mean_ill_all = translate_column_names(df_mean_ill_all)

df_seriously_ill = df_all_values_patients.loc[df_all_values_patients['Бронхиальная астма'] == 'тяжелое течение']
df_seriously_ill = df_seriously_ill.loc[:, [
                                                    'Возраст',
                                                    'ИМТ',
                                                    'Давность заболевания',
                                                    'Количество обострений за последний год',
                                                    'IgE',
                                                    'СРБ ',
                                                    'липиды',
                                                    'Глюкоза на момент поступления',
                                                    'Эозинофиллия'
                                                ]
                                            ]
df_seriously_ill.astype(float)
df_seriously_ill = translate_column_names(df_seriously_ill)

'''
    Seriously ill grope separate in 2 gropes:
    first group ('a' - part) 16 patientss (control group),
    second group ('b' - part) 25 patientss (experimental group)
 '''
df_a_seriously_ill = df_seriously_ill[:16]
df_b_seriously_ill = df_seriously_ill[16:]


'''
Creating average values from values df_all_patients columns, as (summ all values)/(count values)
'''
mean_values_all_patients = df_all_patients.mean()
mean_values_mean_ill_all = df_mean_ill_all.mean()
mean_values_seriously_ill = df_seriously_ill.mean()
mean_values_a_seriously_ill = df_a_seriously_ill.mean()
mean_values_b_seriously_ill = df_b_seriously_ill.mean()

'''
Creating dispersion values from values df_all_patients columns, as 
mean_values_all_patients = np.mean(values)
deviations = values - mean_values_all_patients
sum_of_squares = np.sum(deviations ** 2)
dispersion = sum_of_squares / len(ages)
'''
values_dispersion_all_patients = df_all_patients.var()
values_dispersion_mean_ill_all = df_mean_ill_all.var()
values_dispersion_seriously_ill = df_seriously_ill.var()
values_dispersion_a_seriously_ill = df_a_seriously_ill.var()
values_dispersion_b_seriously_ill = df_b_seriously_ill.var()

'''standard_deviation'''
values_standard_deviation_all_patients = df_all_patients.std()
values_standard_deviation_mean_ill_all = df_mean_ill_all.std()
values_standard_deviation_seriously_ill = df_seriously_ill.std()
values_standard_deviation_a_seriously_ill = df_a_seriously_ill.std()
values_standard_deviation_b_seriously_ill = df_b_seriously_ill.std()


df_avg_patients = pd.DataFrame(
    {
        'Column name': mean_values_all_patients.index,
        'Average values all patients': mean_values_all_patients.values,
        'Standard deviation values 1 group patients': mean_values_mean_ill_all.values,
        'Average values all 2 group patients': mean_values_seriously_ill.values,
        'Average values 2 group part "a" patients': mean_values_a_seriously_ill.values,
        'Average values 2 group part "b" patients': mean_values_b_seriously_ill.values
    }
)
df_dispersion_patients = pd.DataFrame(
    {
        'Column name': values_dispersion_all_patients.index,
        'Dispersion values': values_dispersion_all_patients.values,
        'Dispersion values 1 group patients': values_dispersion_mean_ill_all.values,
        'Dispersion values all 2 group patients': values_dispersion_seriously_ill.values,
        'Dispersion values 2 group part "a" patients': values_dispersion_a_seriously_ill.values,
        'Dispersion values 2 group part "b" patients': values_dispersion_b_seriously_ill.values
    }
)
df_standard_deviation_patients = pd.DataFrame(
    {
        'Column name': values_standard_deviation_all_patients.index,
        'Standard deviation values all patients': values_standard_deviation_all_patients.values,
        'Standard deviation values 1 group patients': values_standard_deviation_mean_ill_all.values,
        'Standard deviation values all 2 group patients': values_standard_deviation_seriously_ill.values,
        'Standard deviation values 2 group part "a" patients': values_standard_deviation_a_seriously_ill.values,
        'Standard deviation values 2 group part "b" patients': values_standard_deviation_b_seriously_ill.values

     }
)

bins = dict_bins(df_all_patients)

plot_distributions(df_all_patients, bins, 'all_patients')
plot_distributions(df_mean_ill_all, bins, 'mean_ill_all')
plot_distributions(df_seriously_ill, bins, 'seriously_all_ill')
plot_distributions(df_a_seriously_ill, bins, 'a_seriously_ill')
plot_distributions(df_b_seriously_ill, bins, 'b_seriously_ill')


correlation_matrix(df_all_patients, 'all_patients')
correlation_matrix(df_mean_ill_all, 'mean_ill_all')
correlation_matrix(df_seriously_ill, 'seriously_all_ill')
correlation_matrix(df_a_seriously_ill, 'a_seriously_ill')
correlation_matrix(df_b_seriously_ill, 'b_seriously_ill')


with pd.ExcelWriter('result/result.xlsx') as writer:
    df_all_patients.to_excel(writer, sheet_name='data', index=False)
    df_all_patients_grade.to_excel(writer, sheet_name='data_grade', index=False)
    df_avg_patients.to_excel(writer, sheet_name='average_measurements', index=False)
    df_dispersion_patients.to_excel(writer, sheet_name='dispersion_measurements', index=False)
    df_standard_deviation_patients.to_excel(writer, sheet_name='standard_deviation', index=False)
