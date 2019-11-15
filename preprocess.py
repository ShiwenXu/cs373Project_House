# CS373 Project

import csv
import io
import random

def read_file():
        with open('HousePrices_HalfMil.csv', 'r') as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                data_lsts = list(readCSV)
        return data_lsts

def processData(data_lsts):
        selectlsts = []
        price = []
        attributes_name = []
        # select wanted features
        for i in range(len(data_lsts)):
                if i > 0:
                        # store price
                        price.append(int(data_lsts[i].pop(15)))
                else:
                        data_lsts[i].pop(15)
                # remove fiber
                data_lsts[i].pop(11)
                # remove indian marbel
                data_lsts[i].pop(6)
                # remove black marbel
                data_lsts[i].pop(5)
                # remove FirePlace
                data_lsts[i].pop(2)
                
                if i == 0:
                        attributes_name = data_lsts[i]
                else:
                        selectlsts.append(data_lsts[i])

        return selectlsts, attributes_name, price

def price_range(price):
        
        price_min = min(price)
        price_max = max(price)
        
        # price range of the dataset
        price_range = int(price_max) - int(price_min)
        increment = price_range / 10
        dic_price = {}
        
        # assign price scope to each price
        for i in price:
                scope = (i - price_min) / increment
                dic_price[i] = scope + 1
                
        return dic_price

def select_training(selectlsts, price):
        # random choose index for trainging and testing
        rand_lsts = random.sample([i for i in range(500000)],1100)
        testing = random.sample(rand_lsts, 100)
        rand_1000 = list(set(rand_lsts) - set(testing))
        rand_600 = random.sample(rand_1000, 600)
        
        # init lists for training and testing
        test_data = []
        test_price = []
        
        data_1000 = []
        price_1000 = []
        
        data_600 = []
        price_600 = []
        
        # obtain data for training and testing
        for i in testing:
                test_data.append(selectlsts[i])
                test_price.append(price[i])
        
        for i in rand_1000:
                data_1000.append(selectlsts[i])
                price_1000.append(selectlsts[i])
        
        for i in rand_600:
                data_600.append(selectlsts[i])
                price_600.append(selectlsts[i])

        return test_data, test_price, data_1000, price_1000, data_600, price_600

def main():
        data_lsts = read_file()
        selectlsts, attributes_name, price = processData(data_lsts)
        test_data, test_price, data_1000, price_1000, data_600, price_600 = select_training(selectlsts, price)
        price_range_dic = price_range(price)

main()