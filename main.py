import sys
sys.path.append('R:\classes 2020-22\Fall 2021\mnist')
import my_dataset as db
import models
import torch
#%%
tr,ts,vl = db.dataset(True)
x = tr[0][:,:,:,0].reshape(tr[0].shape[0],28*28)
y = tr[1]
xv = ts[0][:,:,:,0].reshape(ts[0].shape[0],28*28)
yv = ts[1]
xt = vl[0].reshape(vl[0].shape[0],28*28)
yt = vl[1]
#%%
#LogisticRegression
model = models.Logistic_Regression()
model.fit(x, tr[1])
print('LogisticRegression')
print('Test Accuracy:',models.accuracy(y,model.predict(x)))
print('Evaluation Accuracy:',models.accuracy(yv,model.predict(xv)))
print('Test(custom dataset) Accuracy:',models.accuracy(yt,model.predict(xt)))
'''
Observations:
  The model fails to converge.
  Reported accuracy statistics:
    LogisticRegression
    Test Accuracy: tensor(0.9339)
    Evaluation Accuracy: tensor(0.9255)
    Test(custom dataset) Accuracy: tensor(0.3600) 
'''
#%%
#SVM
model = models.SVM()
model.fit(x, tr[1])
print('SVM')
print('Test Accuracy:',models.accuracy(y,model.predict(x)))
print('Evaluation Accuracy:',models.accuracy(yv,model.predict(xv)))
print('Test(custom dataset) Accuracy:',models.accuracy(yt,model.predict(xt)))
'''
Observations:
  
  Reported accuracy statistics:
    SVM
    Test Accuracy: tensor(0.9899)
    Evaluation Accuracy: tensor(0.9792)
    Test(custom dataset) Accuracy: tensor(0.6000)
'''
#%%
x=torch.from_numpy(x)
xv=torch.from_numpy(xv)
xt=torch.from_numpy(xt)
y=torch.from_numpy(y)
yv=torch.from_numpy(yv)
yt=torch.from_numpy(yt)
#%%
#1 Layvr NN with reg
model = models.NN1layer(device='cuda')
losses = model.fit(x, y, xv, yv,regularize=True,print_losses=50,eps=100,lr=1E-4,batch_size=32)#100,1E-5,16
models.plot_loss(losses,title='3 Layvr NN, w/ Reg&dropout')
print('Dense NN 1 layers(perceptron)')
print('Test Accuracy:',models.accuracy(y,model.predict(x)))
print('Evaluation/Validation Accuracy:',models.accuracy(yv,model.predict(xv)))
print('Test(custom dataset) Accuracy:',models.accuracy(yt,model.predict(xt)))
'''
Epoch 1 :
      Training loss: 1.9810410239537557 , Evaluation loss: 1.7682199802918313
Epoch 50 :
      Training loss: 1.5525813439687093 , Evaluation loss: 1.540137346738424
Epoch 100 :
      Training loss: 1.545045236269633 , Evaluation loss: 1.533633223328835
Observations:
  
  Reported accuracy statistics:
    Dense NN 1 layers(perceptron)
    Test Accuracy: tensor(0.9285)
    Evaluation/Validation Accuracy: tensor(0.9272)
    Test(custom dataset) Accuracy: tensor(0.4600)
'''
#%%
#10 Layvr NN with reg & dropout
model = models.NN3layer(dropout=0.5,device='cuda')
losses = model.fit(x, y, xv, yv,regularize=True,print_losses=50,eps=100,lr=1E-4,batch_size=32)#100,1E-5,16
models.plot_loss(losses,title='3 Layvr NN, w/ Reg&dropout')
print('Dense NN 10 layers')
print('Test Accuracy:',models.accuracy(y,model.predict(x)))
print('Evaluation/Validation Accuracy:',models.accuracy(yv,model.predict(xv)))
print('Test(custom dataset) Accuracy:',models.accuracy(yt,model.predict(xt)))
'''
Epoch 1 :
      Training loss: 1.7473403388341269 , Evaluation loss: 1.559548319914402
Epoch 50 :
      Training loss: 1.51746822903951 , Evaluation loss: 1.4876713236937156
Epoch 100 :
      Training loss: 1.509052510579427 , Evaluation loss: 1.4844706054681387
Observations:
  
  Reported accuracy statistics:
    Dense NN 10 layers
    Test Accuracy: tensor(0.9821)
    Evaluation/Validation Accuracy: tensor(0.9768)
    Test(custom dataset) Accuracy: tensor(0.6600)
'''
#%%
#10 layer FCNN with reg & dropout
model = models.CNN3NN3layer(dropout=0.5,device='cuda')
losses = model.fit(x, y, xv, yv,regularize=True,print_losses=50,eps=100,lr=1E-4,batch_size=32)#100,1E-5,16
models.plot_loss(losses,title='3 Layvr CNN + 3 layer NN, w/ Reg&dropout')
print('FCNN 10 layers')
print('Test Accuracy:',models.accuracy(y,model.predict(x)))
print('Evaluation/Validation Accuracy:',models.accuracy(yv,model.predict(xv)))
print('Test(custom dataset) Accuracy:',models.accuracy(yt,model.predict(xt)))
'''
Epoch 1 :
      Training loss: 1.8404939883550009 , Evaluation loss: 1.5636011086977446
Epoch 50 :
      Training loss: 1.5114175075531007 , Evaluation loss: 1.4820362539627614
Epoch 100 :
      Training loss: 1.5021543897628784 , Evaluation loss: 1.477834382882485
Observations:
  
  Reported accuracy statistics:
    FCNN 10 layers
    Test Accuracy: tensor(0.9855)
    Evaluation/Validation Accuracy: tensor(0.9833)
    Test(custom dataset) Accuracy: tensor(0.8600)
'''
#%%