# LeNet5 object
import numpy as np
from utils.ConvolveLayer import *
from utils.ActivationLayer import *
from utils.helper import *
from utils.FullyConnectedLayer import *
from utils.MaxpoolLayer import *
from utils.LossLayer import *

class LeNet5(object):

    def __init__(self):
        self.C1 = ConvolveLayer((6,1,5,5), (1,32,32), stride = 1, pad = 0)
        self.S2 = MaxpoolLayer(filtersize = 2)
        self.A2 = ActivationLayer(ReLU, d_ReLU)
        self.C3 = ConvolveLayer((16,6,5,5), (6,14,14), stride = 1, pad = 0)
        self.S4 = MaxpoolLayer(filtersize = 2)
        self.A4 = ActivationLayer(ReLU, d_ReLU)
        self.C5 = ConvolveLayer((120,16,5,5), (16,5,5), stride = 1, pad = 0)
        self.A5 = ActivationLayer(ReLU, d_ReLU)
        self.F6 = FullyConnectedLayer((120,84))
        self.A6 = ActivationLayer(ReLU, d_ReLU)
        self.F7 = FullyConnectedLayer((84,10))


    def Forward_Propagation(self, input_image, input_label, mode):

        self.inputshape = input_image.shape  # for testing

        self.label = input_label
        self.C1_res = self.C1.forward_prop(input_image)
        self.S2_res = self.S2.forward_prop(self.C1_res)
        self.A2_res = self.A2.forward_prop(self.S2_res)
        self.C3_res = self.C3.forward_prop(self.A2_res)
        self.S4_res = self.S4.forward_prop(self.C3_res)
        self.A4_res = self.A4.forward_prop(self.S4_res)
        self.C5_res = self.C5.forward_prop(self.A4_res)
        self.A5_res = self.A5.forward_prop(self.C5_res)

        self.flatten = self.A5_res[:,:,0,0]

        self.F6_res = self.F6.forward_prop(self.flatten)
        self.A6_res = self.A6.forward_prop(self.F6_res)
        self.F7_res = self.F7.forward_prop(self.A6_res)

        # print(f"input_image = \n{input_image}")
        # print(f"self.C1_res = \n{self.C1_res}")
        # print(f"self.S2_res = \n{self.S2_res}")
        # print(f"self.A2_res =\n{self.A2_res}")
        # print(f"self.C3_res =\n{self.C3_res}")
        # print(f"self.S4_res =\n{self.S4_res}")
        # print(f"self.A4_res =\n{self.A4_res}")
        # print(f"self.C5_res =\n{self.C5_res}")
        # print(f"self.A5_res =\n{self.A5_res}")
        # print(f"self.F6_res =\n{self.F6_res}")
        # print(f"self.A6_res =\n{self.A6_res}")
        # print(f"self.F7_res =\n{self.F7_res}")

        if mode == 'train':
            self.L8 = LossLayer(input_label)
            self.loss = self.L8.forward_prop(self.F7_res)
            print(f"loss = {self.loss}")
            return self.loss
        elif mode == 'test':
            pred_res = np.argmax(self.F7_res, axis = 1)
            # accuracy = np.sum(pred_res == input_label) / (input_label.shape[0])
            # print(f"accuracy = {accuracy}")
            error = np.sum(pred_res != input_label)
            return error, pred_res


    def Back_Propagation(self, lr_global, momentum = 0.9):

        d_global = self.L8.backward_prop()
        d_F7 = self.F7.backward_prop(d_global, lr_global)
        d_A6 = self.A6.backward_prop(d_F7)
        d_F6 = self.F6.backward_prop(d_A6, lr_global)
        d_F6 = d_F6[:,:,np.newaxis,np.newaxis]  # unflatten
        d_A5 = self.A5.backward_prop(d_F6)
        d_C5 = self.C5.backward_prop(d_A5, lr_global)
        d_A4 = self.A4.backward_prop(d_C5)
        d_S4 = self.S4.backward_prop(d_A4)
        d_C3 = self.C3.backward_prop(d_S4, lr_global)
        d_A2 = self.A2.backward_prop(d_C3)
        d_S2 = self.S2.backward_prop(d_A2)
        d_C1 = self.C1.backward_prop(d_S2, lr_global)

        # print(f"d_global = \n{d_global}")
        # print(f"d_F7 = \n{d_F7}")
        # print(f"d_A6 = \n{d_A6}")
        # print(f"d_F6 = \n{d_F6}")
        # print(f"d_A5 = \n{d_A5}")
        # print(f"d_C5 = \n{d_C5}")
        # print(f"d_A4 = \n{d_A4}")
        # print(f"d_S4 = \n{d_S4}")
        # print(f"d_C3 = \n{d_C3}")
        # print(f"d_A2 = \n{d_A2}")
        # print(f"d_S2 = \n{d_S2}")
        # print(f"d_C1 = \n{d_C1}")

        assert(d_C1.shape == (self.inputshape))
        # print(d_C1.shape)


