import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce
from openai import OpenAI
from collections import defaultdict


def merge_similar_files():

    pass


class Logging:

    def __init__(self, func):

        self.log = []
        self.func = func

    def __call__(self, *args, **kwargs):
        func_name = self.func.__name__
        self.log.append(func_name)
        print(self.log)
        return self.func(*args, **kwargs)

    def __get__(self, instance, owner):

        return lambda *args, **kwargs: self.__call__(instance, *args, **kwargs)


class DataManager:

    def __init__(self, path=None):

        self.path = path
        self.categories = defaultdict(list)
        self.errors = defaultdict(list)

    def log_err(self, err_type, err):

        self.errors[err_type].append(err)

    @Logging
    def group_similar_dataframes(self):

        if self.path is None:
            raise TypeError('Expected path got None. Add path to arguments.')

        categories = self.agent.group_files(self.path)
        categories = categories.split(',')

        categories = categories.split(',')
        categories = [x.strip() for x in categories]

        for filename in os.listdir(self.path):

            file_pth = os.path.join(self.path, filename)
            for cat in categories:
                if cat in filename:
                    self.categories[cat].append(file_pth)

    @Logging
    def stack_similar_dataframes(self, save_type):

        if save_type not in ['excel', 'csv']:
            raise TypeError(f'Expected save type ["excel", "csv"] got {save_type}')

        pth_to_output = 'DataManager'
        if not os.path.exists(pth_to_output):
            os.mkdir(pth_to_output)

        for key, values in self.categories.items():

            dfs = []
            cols = []
            for df in values:

                if not df.endswith('.csv') or not df.endswith('.xlsx'):
                    self.log_err(err_type='Wrong Format', err=df)

                if df.endswith('.csv'):
                    df = pd.read_csv(df, low_memory=False)
                    dfs.append(df)

                elif df.endswith('.xlsx'):
                    df = pd.read_excel(df)
                    dfs.append(df)

                cols.append(df.columns)

                common_cols = list(reduce(lambda x, y: set(x).intersection(y), cols))

                dfs = [df[common_cols] for df in dfs]

            df = pd.concat(dfs, axis=0)
            if save_type == 'excel':
                df.to_excel(f'DataManager/{key}_collected.xlsx', index=False)
            elif save_type == 'csv':
                df.to_csv(f'DataManager/{key}_collected.csv', index=False)


class Bot:

    def __init__(self):

        with open('key.txt', 'r') as f:
            key = f.readline()

        self.client = OpenAI(api_key=key)

    def cleaning_overview(self):

        prompt = f'''You are given the filename and filetypes alongside the list of functions contained within the cleaning class
                Your job is to decide based on this data the steps needed to ensure that this dataset is well clearned. This includes dropping duplicates if deemed important
                or removing nan values if the dataset is long enough for example. Output the list of steps needed to clean the dataset and seperate each function with a comma'''

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": {"url": self.url}}
                    ]
                }
            ]
        )

        choice = response.choices[0]

    def rename_cols(self, filename, cols):

        prompt = f'''Infer from the file name {filename}, what the meaning of each columns is and rename
        the columns so that they can be more easily understood. The column names are {cols}. When returning
        the columns make sure that each one is separated by a comma.'''

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": {"url": self.url}}
                    ]
                }
            ]
        )

        choice = response.choices[0]
        return choice.message.content

    def group_files(self, directory_list):

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a categorisation assistant."},
                {"role": "user", "content": f"Categorise these items into main groups: {directory_list}. "
                                            f"Only list categories with names that appear in a list of file. E.g"
                                            f'jan_features, feb_features would both go in a category called features.'
                                            f' Return only the name of the category seperated by a comma if there is '
                                            f'more than one'}
            ]
        )

        return response.choices[0].message.content


class Clean:

    def __init__(self, dataframe=None) -> None:

        self.agent = Bot()
        self.df = dataframe
        self.categories_: dict = defaultdict(list)
        self.save_location: str = 'Cleaned_dataframes'
        self.errors = defaultdict(list)

    def drop_(self):

        self.df.dropna(how='all', inplace=True)
        self.df.dropna(how='all', axis=1, inplace=True)

    def agent_rename_cols(self):
        pass

    @staticmethod
    def remove_delimiters(x):

        delimiters = [',', '.', '/', '-', ' ']
        for dlm in delimiters:
            if dlm in x:
                x = x.replace(dlm, '')
        return x

    def format_date(self):

        for col in self.df.columns:

            day_month_format = ''
            date_format = ['_'] * 8
            remaining = ''

            if 'date' in col:

                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.replace(r'\.0$', '', regex=True)
                    .str.replace(r'[-/ .]', '', regex=True)
                    .str.zfill(8)
                )

                new_row = self.df[col].apply(lambda val: len(val))
                unique = new_row.unique()
                if len(unique) != 1:
                    most_common_value = self.df[col].value_counts().index[0]
                    self.df = self.df[self.df[col] == most_common_value]

                    # Add an error logger here

                for x in self.df[col]:

                    opt1_year, opt2_year = x[:4], x[4:]
                    opt1_year_int, opt2_year_int = int(opt1_year), int(opt2_year)

                    if 1900 <= opt1_year_int <= 2099 and 1900 <= opt2_year_int <= 2099:
                        continue

                    elif 1900 <= opt1_year_int <= 2099:
                        remaining = x.replace(opt1_year, '')
                        date_format[:4] = 'Y' * 4

                    elif 1900 <= opt2_year_int <= 2099:
                        remaining = x.replace(opt2_year, '')
                        date_format[4:] = 'Y' * 4

                    split = len(remaining) // 2
                    one, two = remaining[:split], remaining[split:]

                    if int(one) == int(two) or (int(one) < 12 and int(two) < 12):
                        continue

                    elif int(one) > 12:

                        day_month_format = f'ddmm'

                    elif int(two) > 12:

                        day_month_format = f'mmdd'

                    find_filler = [pos for pos, x in enumerate(date_format) if x == '_']
                    first_filler, last_filler = find_filler[0], find_filler[-1] + 1

                    date_format[first_filler:last_filler] = day_month_format

                    seen = set()
                    unique = [x for x in date_format if not (x in seen or seen.add(x))]
                    ft = '%'.join(unique)
                    ft = f'%{ft}'

                    self.df[col] = pd.to_datetime(self.df[col], format=ft, errors='coerce')
                    self.df.dropna(subset=[col], inplace=True)
                    break

    def return_df(self):

        return self.df

def main():

    pth = ''
    dm = DataManager(pth)
    dm.group_similar_dataframes()
    dm.stack_similar_dataframes(save_type='csv')


if __name__ == "__main__":

    pth = 'scrambled_output.xlsx'
    df = pd.read_excel(pth)
    clean = Clean(df)
    start = time.time()
    clean.format_date()
    end = time.time()
    df = clean.return_df()
    print(df)
    print(end - start)



