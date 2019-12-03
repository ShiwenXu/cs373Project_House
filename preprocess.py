# CS373 Project

import csv
import io
import random
import numpy as np
from itertools import islice

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

def price_range(price, price_1000, price_600, test_price, price_500_tune):
        
        price_min = min(price)
        price_max = max(price)
        
        # price range of the dataset
        price_range = int(price_max) - int(price_min)
        increment = price_range / 10
        dic_price = {}
        
        counter = 0
        # assign price scope to each price
        for i in price:
                scope = (i - price_min) / increment
                dic_price[i] = scope + 1
        label_1000 = []
        label_600 = []
        label_test = []
        label_500_tune = []
        
        for i in price_1000:
                label_1000.append(dic_price[i])
        for j in price_600:
                label_600.append(dic_price[j])
        for k in test_price:
                label_test.append(dic_price[k])
        for l  in price_500_tune:
                label_500_tune.append(dic_price[l])
                
        return label_1000, label_600, label_test, label_500_tune

def select_training(selectlsts, price):
        # random choose index for trainging and testing
        rand_lsts = random.sample([i for i in range(500000)],1900)
        length_to_split = [1100, 500]
        # testing = random.sample(rand_lsts, 100)
        Inputt = iter(rand_lsts) 
        Output = [list(islice(Inputt, elem)) 
                for elem in length_to_split] 
        testing = random.sample(Output[0], 100)
        rand_1000 = list(set(Output[0]) - set(testing))
        rand_600 = random.sample(rand_1000, 600)
        rand_500_tune = Output[1]
        
        # init lists for training and testing
        test_data = []
        test_price = []
        
        data_1000 = []
        price_1000 = []
        
        data_600 = []
        price_600 = []
        
        data_500_tune = []
        price_500_tune = []
        
        # obtain data for training and testing
        for i in testing:
                test_data.append(selectlsts[i])
                test_price.append(price[i])
        
        for i in rand_1000:
                data_1000.append(selectlsts[i])
                price_1000.append(price[i])
        
        for i in rand_600:
                data_600.append(selectlsts[i])
                price_600.append(price[i])

        for i in rand_500_tune:
                data_500_tune.append(selectlsts[i])                
                price_500_tune.append(price[i])

        return test_data, test_price, data_1000, price_1000, data_600, price_600, data_500_tune, price_500_tune

def obtain_result():
        data_lsts = read_file()
        selectlsts, attributes_name, price = processData(data_lsts)
        test_data, test_price, data_1000, price_1000, data_600, price_600, data_500_tune, price_500_tune = select_training(selectlsts, price)
        label_1000, label_600, label_test, label_500_tune = price_range(price, price_1000, price_600, test_price, price_500_tune)
        
        test_data = np.asmatrix(test_data)
        data_1000 = np.asmatrix(data_1000)
        data_600 = np.asmatrix(data_600)
        data_500_tune = np.asmatrix(data_500_tune)
        
        test_label = (np.asmatrix(label_test)).T.astype(np.int)
        label_1000 = (np.asmatrix(label_1000)).T.astype(np.int)
        label_600 = (np.asmatrix(label_600)).T.astype(np.int)
        label_500_tune = (np.asmatrix(label_500_tune)).T.astype(np.int)
        
        
        return data_500_tune, label_500_tune, data_600, label_600, data_1000, label_1000, test_data, test_label


obtain_result()
