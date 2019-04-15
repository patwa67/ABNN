import mxnet as mx
import numpy as np
from numpy import genfromtxt
#Load data (phenotype in first column and there after SNPs columnwise with 0,1,2 coding)
my_data = genfromtxt('QTLMAS2010ny012.csv', delimiter=',')
#Mean standardize phenotype
label = np.array(my_data[:,0]-np.mean(my_data[:,0]))
#Number of SNPs
nsnp = 9723
#Size of training and test data
ntrain = 2326
ntest = 900
#Create training and test data
traindata = my_data[:ntrain, 1:(nsnp+1)]
testdata = my_data[ntrain:, 1:(nsnp+1)]
#Convert SNP data to one-hot encoding
hottrain = mx.nd.one_hot(mx.nd.array(traindata, dtype=np.int32), 3)
hottest = mx.nd.one_hot(mx.nd.array(testdata, dtype=np.int32), 3)
nhotvar = nsnp*3
del my_data
#Set SGD batch size
batch_size = 150
#Create internal data iterators
train_iter = mx.io.NDArrayIter(hottrain, label[:ntrain], batch_size, shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(hottest, label[ntrain:], batch_size)
#Set up one layer, one node neural network with 0.5 drop-out
X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
dropout = mx.symbol.Dropout(data=X, name='do1', p = 0.5)
fully_connected_layer  = mx.sym.FullyConnected(data=dropout, name='fc1', num_hidden = 1)
#For two layers, add the following two rows
#dropout2 = mx.symbol.Dropout(data=fullyc1, name='do2', p = 0.5)
#fullyc2  = mx.sym.FullyConnected(data=dropout2, name='fc2', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")
#Create the model, add context=mx.gpu() for GPU computation
model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)
#Initialize output to checkpoint
model_prefix = 'mx_reg'
checkpoint = mx.callback.do_checkpoint(model_prefix)
#Fit the model using the Adam optimizer with weight decay 1.4,
#learnng rate 0.0001 and 6000 iterations (epochs)
niter = 6000
model.fit(train_iter, eval_iter, optimizer='Adam',
            optimizer_params={'wd':1.4,'learning_rate':0.0001},
            num_epoch=niter,
            epoch_end_callback=checkpoint)
#Use MSE as loss error
metric = mx.metric.MSE()
#Create matrices for the weights, predicted values and MSE,
#and load results back into Python
wres = np.random.rand(niter,nhotvar)
preds = np.random.rand(niter,ntest)
mses= np.random.rand(niter,1)
for x in xrange(1,niter):
  sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, x)
  assert sym.tojson() == lro.tojson()
  model.set_params(arg_params, aux_params)
  wres[x-1,:] = model.get_params()[0]['fc1_weight'].asnumpy()
  preds[x-1,:] = np.transpose(model.predict(eval_iter).asnumpy())
  mses[x-1] = model.score(eval_iter, metric)[0][1]
#Calculate model averages without burn-in
ulim = niter-1
bi = 999
wmean = np.mean(wres[bi:ulim,:],axis=0) #One-hot encoded weight mean
predsmean = np.mean(preds[bi:ulim,:],axis=0) #Yhat mean
msesmean = np.mean(mses[bi:ulim,:],axis=0) #MSE mean
msessd = np.std(mses[bi:ulim,:],axis=0) #MSE SD
