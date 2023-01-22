import pandas as pd

def get_attr(obj, string):
    return getattr(obj,string)


class ILottoCSV(object):
    
    def __init__(self, input_csv="Lotto.csv", output_csv="input/lottoIL_filt.csv") -> None:
        self.file_name = input_csv
        self.out_file =  output_csv
        self.header = ["Date", "Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6", "Ball_Bonus"]
        self.ball_numbers = 37
        self.strong_numbers = 7 
        self.filter_csv() 
        
        
    def filter_csv(self):
        names = self.header
        lotto = pd.read_csv(self.file_name, encoding='latin-1')
        curr_names = lotto.columns
        lotto.drop(curr_names[0], axis=1, inplace=True)

        cnt_idx = 0
        for column_headers in lotto.columns: 
            if cnt_idx > len(names)-1:
                lotto.drop(column_headers, axis=1, inplace=True)
            else:
                lotto.rename(columns = {column_headers:names[cnt_idx]}, inplace = True)
                cnt_idx += 1

        lotto = pd.DataFrame(lotto).set_index(names[0])

        lotto.drop(lotto[ (get_attr(lotto, names[1]) > self.ball_numbers) 
                        | (get_attr(lotto, names[2]) > self.ball_numbers) 
                        | (get_attr(lotto, names[3]) > self.ball_numbers) 
                        | (get_attr(lotto, names[4]) > self.ball_numbers) 
                        | (get_attr(lotto, names[5]) > self.ball_numbers) 
                        | (get_attr(lotto, names[6]) > self.ball_numbers) 
                        | (get_attr(lotto, names[7]) > self.strong_numbers) 
                        ].index, inplace=True)

        lotto.to_csv(self.out_file)

if __name__ == "__main__":
    test = ILottoCSV()