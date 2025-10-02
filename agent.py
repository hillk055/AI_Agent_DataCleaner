import pandas as pd
from datetime import datetime


class Bot:

    def __init__(self):

        pass


class Clean:

    def __init__(self, df):

        self.df = df
        self.columns = df.columns

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
                self.df[col] = self.df[col].astype(str).str.zfill(8)

                values = self.df[col]
                values = [self.remove_delimiters(x) for x in values]

                num_date_formats = set([len(x) for x in values])
                if len(num_date_formats) != 1:
                    raise ValueError('Mismatching date formats')

                for x in values:

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

                    self.df[col] = list(map((lambda x: datetime.strptime(x, ft).strftime("%Y-%m-%d")), values))
                    break

    def return_df(self):

        return self.df['tourney_date']


if __name__ == "__main__":

    pth = ''
    dataframe = pd.read_csv(pth, nrows=100)
    clean = Clean(dataframe)
    clean.format_date()
    df = clean.return_df()
    print(df)
