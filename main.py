import pandas as pd
import matplotlib.pyplot as plt
from apriori_python import apriori
import time
from apyori import apriori as apriory2
from efficient_apriori import apriori as apriory3
from fpgrowth_py import fpgrowth

def loadData(path):
    return pd.read_csv(path)

def apriori_apply(transactions):
    start = time.perf_counter()
    t, rules = apriori(transactions, minSup= 0.03, minConf=0.25)
    finish = (time.perf_counter() - start)
    print(rules)
    return {'time': finish, 'result': rules}

def prepare_dataset(data):
    transactions = []
    for k in range(data.shape[0]):
        row = data.iloc[k].dropna().tolist()
        transactions.append(row)
    return transactions

def fp_apply(transactions):
    start = time.perf_counter()
    itemsets, rules = fpgrowth(transactions, minSupRatio= 0.03, minConf= 0.25)
    finish = time.perf_counter() - start
    return {'time': finish, 'result': rules}

def cmp_methods(result_apriori, result_fp):
    print('apriori: ', result_apriori['time'])
    print('fp', result_fp['time'])

if __name__ == '__main__':
    data = loadData("data.csv")
    data.stack().value_counts(normalize=True).head(20).plot(kind='bar')
    # plt.title("Относительные частоты")
    # plt.show()

    data.stack().value_counts(normalize=True).head(20).apply(lambda item:item / data.shape[0]).plot(kind='bar')
    # plt.title('Фактические частоты')
    # plt.show()

    data = prepare_dataset(data)

    apr = apriori_apply(data)
    fp = fp_apply(data)
    print(apr, fp)
    plt.title('time comparation')
    cmp_data = [apr['time'], fp['time']]

    print('this', cmp_data)
    plt.clf()
    plt.bar(['apriori', 'fp'], cmp_data)
    plt.show()


