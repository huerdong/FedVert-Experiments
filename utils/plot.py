from matplotlib import pyplot as plt

class MetricTracker:
    def __init__(self, start_time):
        self.cur_flops = 0
        self.cur_transfer = 0
        #self.start_time = start_time
        #self.timestamps = []
        self.flops = [] # Cumulative flops at each round
        self.transferred = [] # Cumulative bytes transferred per round
        self.accuracy = []

    def add_flop(self, flops):
        self.cur_flops += flops
        #self.timestamps.append(time - self.start_time)
        self.flops.append(self.cur_flops)

    def add_transfer(self, bytes_transferred):
        self.cur_transfer += bytes_transferred
        self.transferred.append(self.cur_transfer)
        
    def add_accuracy(self, accuracy):
        self.accuracy.append(accuracy)

    def write(self, name_prefix):
        with open(f"out/{name_prefix}_flop_readout.csv", "w") as ofile:
            for (flop, acc) in zip(self.flops, self.accuracy):
                ofile.write(f"{flop},{acc}\n")
        
        with open(f"out/{name_prefix}_transfer_readout.csv", "w") as ofile:
            for (trans, acc) in zip(self.transferred, self.accuracy):
                ofile.write(f"{trans},{acc}\n")
        
        flop_fig = plt.figure()
        plt.plot(self.flops, self.accuracy)
        plt.title("Accuracy over Floating Point Operations")
        plt.xlabel("FLOPs")
        plt.ylabel("Accuracy")
        plt.savefig(f"out/{name_prefix}_flop_plot.png")

        plt.close()

        trans_fig = plt.figure()
        plt.plot(self.transferred, self.accuracy)
        plt.title("Accuracy over Bytes Transferred")
        plt.xlabel("Bytes Transferred")
        plt.ylabel("Accuracy")
        plt.savefig(f"out/{name_prefix}_transfer_plot.png")

        plt.close()

    def write_plots(self, name_prefix):        
        flop_fig = plt.figure()
        plt.plot(self.flops, self.accuracy)
        plt.title("Accuracy over Floating Point Operations")
        plt.xlabel("FLOPs")
        plt.ylabel("Accuracy")
        plt.savefig(f"out/{name_prefix}_flop_plot.png")

        trans_fig = plt.figure()
        plt.plot(self.transferred, self.accuracy)
        plt.title("Accuracy over Bytes Transferred")
        plt.xlabel("Bytes Transferred")
        plt.ylabel("Accuracy")
        plt.savefig(f"out/{name_prefix}_transfer_plot.png")

    def load_from_csv(self, flops_file, trans_file):
        with open(flops_file) as infile:
            data = infile.readlines()
            for l in data:
                res = l.split(',')
                flops = float(res[0])
                acc = float(res[1])
                self.flops.append(flops)
                self.accuracy.append(acc)

        with open(trans_file) as infile:
            data = infile.readlines()
            for l in data:
                res = l.split(',')
                transfers = float(res[0])
                self.transferred.append(transfers)
    
            