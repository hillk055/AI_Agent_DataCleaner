import pandas as pd
from datetime import datetime


class Bot:

    def __init__(self):

        pass


class Clean:

    def __init__(self, df):

        self.df = df
        self.columns = df.columns

    def get_dtypes(self):

        print(self.df.dtypes)

    def format_date(self):

        date_cols = [x for x in self.df.columns if 'date' in x]
        for column in date_cols:
            self.df[column] = self.df[column].apply(lambda x: str(x))

        date_format = ['_'] * 8
        remaining = ''

        for j in date_cols:
            for x in self.df[j]:
                opt1_year, opt2_year = x[:4], x[4:]
                opt1_year_int, opt2_year_int = int(opt1_year), int(opt2_year)
                if 1900 <= opt1_year_int <= 2099:

                    remaining = x.replace(opt1_year, '')
                    date_format[:4] = 'Y' * 4

                if 1900 <= opt2_year_int <= 2099:

                    remaining = x.replace(opt2_year, '')

                split = len(remaining)//2
                one, two = remaining[:split], remaining[split:]

                if int(one) > 12:
                    date_format[4:6] = ['d'] * 2
                    date_format[6:] = ['m'] * 2

                    break
                elif int(two) > 12:
                    date_format[4:6] = ['m'] * 2
                    date_format[6:] = ['d'] * 2
                    break

                else:
                    continue

            # If x is not in seen x in seen will return False
            seen = set()
            unique = [x for x in date_format if not (x in seen or seen.add(x))]
            ft = '%'.join(unique)
            ft = f'%{ft}'

            for column in date_cols:

                self.df[column] = self.df[column].apply(lambda x: datetime.strptime(x, ft).date())

            print(self.df[date_cols])













    def drop_cols(self):

        pass


if __name__ == "__main__":

    pth = '/Users/keeganhill/Desktop/Projects/Trees/tennis_atp/atp_matches_qual_chall_1993.csv'
    dataframe = pd.read_csv(pth, nrows=100)
    clean = Clean(dataframe)
    clean.format_date()
