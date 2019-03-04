import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split


fileNameandLoc= r'''C:\Users\Asus\Desktop\CH\colormix\seedr\course dl\practicex\zero_to_deep_learning_video\data\HR_comma_sep.csv''';
#tab=torch.utils.data.DataLoader(dataset=fileNameandLoc)
tab = pd.read_csv(fileNameandLoc)

tab.info()
#print(tab.resultx)
#print(tab.left)
#print(tab.satisfaction_level.head(7),tab.last_evaluation.head(7))
# Python code to remove duplicate elements
''' #the following part is check the redudancies in sales list
def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

# Driver Code
duplicate = [2, 4, 10, 20, 5, 2, 20, 4]
print(Remove(tab.sales))
''' #polt data p1/2
''' #Plot the features histograms
plt.subplot(5, 2, 1)
plt.hist(tab.satisfaction_level, bins=50, color='b', alpha=0.5)
plt.ylabel('sastisfaction')
plt.subplot(5, 2, 2)
plt.hist(tab.last_evaluation, bins=50, color='r', alpha=0.9)
plt.ylabel('last evaluation')
plt.subplot(5, 2, 3)
plt.hist(tab.number_project, bins=50, color='g', alpha=0.7)
plt.ylabel('nbr project')
plt.subplot(5, 2, 4)
plt.hist(tab.average_montly_hours, bins=50, color='pink', alpha=0.9)
plt.ylabel('Monthly hours')
plt.subplot(5, 2, 5)
plt.hist(tab.time_spend_company, bins=50, color='b', alpha=0.5)
plt.ylabel('time_spend_company')
plt.subplot(5, 2, 6)
plt.hist(tab.Work_accident, bins=50, color='r', alpha=0.9)
plt.ylabel('Work accident')
plt.subplot(5, 2, 7)
plt.hist(tab.left, bins=50, color='black', alpha=0.5)
plt.ylabel('left')
plt.subplot(5, 2, 8)
plt.hist(tab.promotion_last_5years, bins=50, color='yellow', alpha=0.9)
plt.ylabel('promotion last 5years')

sal = {'low': 0, 'medium' : 0.5, 'high': 1}
plt.subplot(5, 2, 9)
plt.hist(tab.salary.map(sal), bins=50, color='orange', alpha=0.9)
plt.ylabel('salary')
sal = {'sales': 1, 'accounting' : 0.1, 'hr': 0.2,'technical':0.3,'support':0.4, 'management':0.5, 'IT':0.6, 'product_mng':0.7, 'marketing':0.8, 'RandD':0.9}
plt.subplot(5, 2, 10)
plt.hist(tab.sales.map(sal), bins=50, color='purple', alpha=0.9)
plt.ylabel('sales')
plt.show()
''' #plot data p2/2


#### Preparing Data ####
sal = {'low': 0, 'medium' : 0.5, 'high': 1}
tab['salaryMod']=tab.salary.map(sal)
sal = {'sales': 1, 'accounting' : 0.1, 'hr': 0.2,'technical':0.3,'support':0.4, 'management':0.5, 'IT':0.6, 'product_mng':0.7, 'marketing':0.8, 'RandD':0.9}
tab['salesMod']=tab.sales.map(sal)
tab['leftx']=tab.left
tab=tab.drop(['left','sales','salary'],axis=1)
#normalization x-max / max-min
tab.average_montly_hours = (tab.average_montly_hours- tab.average_montly_hours.min())/(tab.average_montly_hours.max()-tab.average_montly_hours.min())
tab = tab.rename(columns={'leftx': 'left', 'salesMod': 'sales', 'salaryMod':'salary'})
tab.info()
#splitting data
nc =(len(tab.columns))-1

traindf, testdf = train_test_split(tab, test_size=0.2)
x_data = Variable(torch.Tensor(traindf.iloc[:,0:nc].values)) #Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = (Variable(torch.Tensor(traindf.iloc[:,nc:].values))) #Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
xt_data = Variable(torch.Tensor(testdf.iloc[:,0:nc].values)) #test input data
yt_data = (Variable(torch.Tensor(testdf.iloc[:,nc:].values))) #test output data

'''
for cnt in range(len(y__data)):
    y_data[cnt]=str(y__data)
    yt_data[cnt]=str(yt__data)
'''
#print(yt_data.min())
#################

class Model(torch.nn.Module):

    def __init__(self,input_size, num_classes): #initializing
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)  # hidden layer
        #self.linear = F.sigmoid()  # n input/feature , one output
        # here where other NN layers are added

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

# save our model
model = Model(nc, 1)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(size_average=True)#.nn.CrossEntropyLoss()#nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print('epoch {}, loss {}',epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#### Test the Model  ####
predicted = model(xt_data)
#predicted = torch.max(predicted.data - 0.27,0)
#predicted = torch.min(predicted.data+1,1)
meanz=predicted.mean()
meanz=1.25*meanz
for cnt in range (len(predicted)):
    if predicted[cnt]>meanz : predicted[cnt]=1
    else: predicted[cnt]=0

total = yt_data.size(0)
print(predicted[1:11])
correct = (predicted == yt_data).sum()
TPcorrect=0
for cnt in range (len(predicted)):
    if ((predicted[cnt] == yt_data[cnt])and (predicted[cnt] == 0)) : TPcorrect=TPcorrect+ 1

#TPcorrect= ((predicted == yt_data)and (predicted == 0)).sum()


print('Accuracy of the model  %d %%' % (100 * correct / total))
print('Precision of the model  %d %%' % (100 * TPcorrect / (len(predicted)-sum(predicted))))
print('Accuracy of the model  %d %%' % (100 * TPcorrect / (len(predicted)-sum(yt_data))))
plt.hist(predicted.detach().numpy(), bins=50, color='yellow', alpha=0.9)
plt.xlabel('0 stayed - 1 left')
plt.show()
