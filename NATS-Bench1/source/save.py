import csv
from datetime import datetime

def get_name():
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    name = "correlation-" + date_time + ".csv"
    print("Writing to: ", name)

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_num','test_acc','test_loss','train_acc','train_loss','quality_L1', 'quality_L2','KG_L1', 'MC_L1'])
        #file.write('\n')

    return name

def write(writer, line):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)
        #file.write('\n')


if __name__ == "__main__":
    name = get_name()
    write(name, ["test", "hello"])
    write(name, ["test2", "hell2"])
    #write(name, ["test3", "hello3"])