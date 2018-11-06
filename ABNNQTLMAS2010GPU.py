# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mxnet as mx
import numpy as np

from numpy import genfromtxt
my_data = genfromtxt('QTLMAS2010ny012.csv', delimiter=',')
label = np.array(my_data[:,0]-np.mean(my_data[:,0]))
ntrain = 2326
traindata = my_data[:ntrain, 1:9724]
testdata = my_data[ntrain:, 1:9724]

hottrain = mx.nd.one_hot(mx.nd.array(traindata, dtype=np.int32), 3)
hottest = mx.nd.one_hot(mx.nd.array(testdata, dtype=np.int32), 3)

del my_data

batch_size = 150

train_iter = mx.io.NDArrayIter(hottrain, label[:ntrain], batch_size, shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(hottest, label[ntrain:], batch_size)

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
dropout = mx.symbol.Dropout(data=X, name='do1', p = 0.5)
fullyc1  = mx.sym.FullyConnected(data=dropout, name='fc1', num_hidden = 1)
#dropout2 = mx.symbol.Dropout(data=fullyc1, name='do2', p = 0.5)
#fullyc2  = mx.sym.FullyConnected(data=dropout2, name='fc2', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fullyc1, label=Y, name="lro")

model = mx.mod.Module(context = mx.gpu(),
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)

model_prefix = 'mx_reg'
checkpoint = mx.callback.do_checkpoint(model_prefix)

model.fit(train_iter, eval_iter, optimizer='Adam',
            optimizer_params={'wd':1.4,'learning_rate':0.0001},
            num_epoch=6000,
            epoch_end_callback=checkpoint)

metric = mx.metric.MSE()

wres = np.random.rand(6000,29169)
#wres2 = np.random.rand(6000,2)
preds = np.random.rand(6000,900)
mses= np.random.rand(6000,1)
bias= np.random.rand(6000,1)
for x in xrange(1,6000):
  sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, x)
  assert sym.tojson() == lro.tojson()
  model.set_params(arg_params, aux_params)
  wres[x-1,:] = model.get_params()[0]['fc1_weight'].asnumpy()
  #wres2[x-1,:] = model.get_params()[0]['fc2_weight'].asnumpy()
  bias[x-1,:] = model.get_params()[0]['fc1_bias'].asnumpy()
  preds[x-1,:] = np.transpose(model.predict(eval_iter).asnumpy())
  mses[x-1] = model.score(eval_iter, metric)[0][1]

wmean = np.mean(wres[1000:5998,:],axis=0)
#wmean2 = np.mean(wres[1000:5998,1,:],axis=0)
#preds2 = np.mean(preds[1000:5998,:],axis=0)
predsmean = np.mean(preds[1000:5998,:],axis=0)
predvar2 = np.var(preds[1000:5998,:],axis=0)
#predsvar = np.mean(predvar2,axis=0)

#wmean = genfromtxt('wmeanQTLMAS201014.out')
nphot=hottest.asnumpy()
nphot.shape=(900,29169)
ypred=np.matmul(nphot,wmean)
np.sum((label[ntrain:]-ypred)**2)/900

nphottp=nphot.transpose(2,2,2).reshape(-1,nphot.shape[0]).transpose()
ypred=np.matmul(nphottp,wmean)
np.sum((label[ntrain:]-ypred)**2)/900

meanmses = np.mean(mses[1000:5998])
stdmses = np.std(mses[1000:5998])

np.savetxt('wmeanQTLMAS201014.out', wmean, delimiter=',')
#np.savetxt('wmean22cv5250.out', wmean2, delimiter=',')
np.savetxt('msesQTLMAS201014.out', mses, delimiter=',')
