import csv
from datetime import datetime

def get_name():
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    name = "outputs/correlation-" + date_time + ".csv"
    print("Writing to: ", name)

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_num','test_acc','test_loss','train_acc','train_loss','quality_L1',
         'quality_L2','quality_prod','KG_L1', 'MC_L1','MC_L3','ER_L1','mquality_L1','mquality_prod','mquality-wL1',
         'mquality-wp','qlayer0mode3','qlayer0mode4','qlayer-1mode3','qlayer-1mode4','KG0','KG-1','MC0','MC-1','qnL1','qnL3','qnL4','qnL5'])
        #file.write('\n')

    return name

def write(name, line):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)
        #file.write('\n')

'''
if __name__ == "__main__":
    name = get_name()
    write(name, ["test", "hello"])
    write(name, ["test2", "hell2"])
    #write(name, ["test3", "hello3"])
'''