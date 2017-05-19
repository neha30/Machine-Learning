import pandas as pd
from sklearn.model_selection import train_test_split
import operator

priors=dict()	#prior probability
prob=dict()		#independent probability
condProb=dict() #conditional probability

def train_data(train,smooth=0.5):

	priors.clear()	 	#P(yes),P(no)
	prob.clear()		#P(overcast),P(normal)

	Y_train=train.iloc[:,-1]

	#calculate the prior probability
	for y in Y_train:
		if y not in priors:
			priors[y]=0
		priors[y]+=1

	for key,val in priors.items():
		priors[key]=round(float(val)/len(train),3)

	#print "priors: ",priors

	#calculate the independent probability
	X_train=train
	X_train=X_train.drop(labels='Play',axis=1)
	for a in list(X_train.columns.values):
		unique_val=list(train[a].unique())
		val_counts=X_train[a].value_counts()
		for val in unique_val:
			prob[val]=round(float(val_counts[val])/len(X_train),3)

	#print prob

	#get the attributes names
	attribute=list(train.columns.values)
	attribute.remove('Play')

	#finding conditional probability
	for atr in attribute:
		unique_inp=list(train[atr].unique())
		unique_out=list(train['Play'].unique())
		ct=pd.crosstab(train['Play'],train[atr],margins=True)
		ct=pd.crosstab(train['Play'], train[atr]).apply(lambda r: (r+smooth)/(r.sum()+(smooth*len(unique_inp))), axis=1)	#apply smoothing
		condProb[atr]=ct

	#for key,val in condProb.items():
		#print key
		#print val


def test_data(test):
	prob_class={}
	predicted=[]
	expected=list(test.iloc[:,-1])
	test=test.drop(labels='Play',axis=1)
	attribute=list(test.columns.values)

	for i in range(len(test)):
		x=list(test.iloc[i].values)  #x=[rainy,mild,normal,false]
		print x,"--->",

		#den=1
		#for val in x:
		#	den*=prob[val]				 #den=prob(rainy)*prob(mild)*prob(normal)*prob(false)

									
		#calulate the probability for evry class
		for out in ['yes','no']:
			num=1						#num=condProb[rainy][yes]*condProb[mild][yes]*condProb[false][yes]*prior(yes)
			for i in range(len(x)):
				ct=condProb[attribute[i]]
				#adding laplace smoothing
				num*=round(ct[x[i]][out],3)
			num*=priors[out]
			
			prob_class[out]=round(float(num),5)

		pred=max(prob_class.iteritems(),key=operator.itemgetter(1))[0]
		print pred
		predicted.append(pred)

	return predicted,expected

def Accuracy(pred,exp):
	correct = 0
	total = len(exp)
	for i in range(len(pred)):
		if pred[i] == exp[i]:
			correct += 1

	acc = (float(correct)/total)*100
	return acc


if __name__ == "__main__":
	data=pd.read_csv('tennis.csv')
	train,test=train_test_split( data,test_size=0.3)
	train_data(train)
	predicted,expected = test_data(test)
	accuracy=Accuracy(predicted,expected)
	print "Accuracy is: ",accuracy


