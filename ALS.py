#You can get the dataset from : https://grouplens.org/datasets/movielens/20m/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
		

def initialMatrix():
	'''Get data from the files and return Q,R user item matrices'''

	# pass in column names for each CSV and read them using pandas. 
	# Column names available in the readme file

	#Reading ratings file:
	#r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
	ratings=pd.read_csv('/home/neha/Desktop/challenges/movielens_dataset/ratings.csv',encoding='latin-1')

	#Reading items file:
	#i_cols = ['movieId','tiitle','genres']
	items = pd.read_csv('/home/neha/Desktop/challenges/movielens_dataset/movies.csv',encoding='utf-16')
	movie_titles=items['title']
    #print users.shape
	#users.head()

	#print ratings.shape
	#ratings.head()

	#print items.shape
	#items.head()

	#Joined the tables into one dataframe 
	df=ratings
	print "Getting 'rp' matrix with rows:user and column:movies"
	rp = df.pivot_table(cols=['movieId'],rows=['userId'],values='rating')
	print "'rp' matrix shape is:",rp.shape

	rp = rp.fillna(0); # Replace NaN
	rp.head()

	Q = rp.values
	Q.shape

	print "Getting 'R' binary matrix of rating or no rating"
	R=Q>0.5
	R[R==True]=1
	R[R==False]=0
	# To be consistent with our Q matrix
	R=R.astype(np.float64,copy=False)

	return Q,R,movie_titles

def runALS(Q,R,n_factors,n_iterations,lambda_):
	'''
	Run Alternating Least Square Algorithm
	Q: USer Iterm rating matrix
	R: User-Item  matrix with 1 if there is rating otherwise 0
	n_factors: number of latent features
	n_iterations: number of times to run algorithm
	lambda_: Regularization parameter
	'''
	print 'Initializing '
	lambda_=0.1
	n_iterations=20
	n_factors=3
	n,m=Q.shape

	Users=5*np.random.rand(n,n_factors)
	Items=5*np.random.rand(m,n_factors)

	MSE_List=[]

	def get_error(Q, Users, Items, R):
    # This calculates the MSE of nonzero elements
		return np.sum((R * (Q - np.dot(Users, Items.T))) ** 2) / np.sum(R)

	print "Starting iterations:"

	for ite in range(n_iterations):
		for i,Ri in enumerate(R):
			#print Items.shape,np.diag(Ri).shape,Items.T.shape,Q[i].T.shape
			Users[i]=np.linalg.solve(np.dot(Items.T,np.dot(np.diag(Ri),Items))+lambda_*np.eye(n_factors),
			                    np.dot(Items.T,np.dot(np.diag(Ri),Q[i].T))).T


		print "Error after solving for User matrix",get_error(Q,Users,Items,R)


		for j,Rj in enumerate(R.T):
			Items[j]=np.linalg.solve(np.dot(Users.T,np.dot(np.diag(Rj),Users))+lambda_*np.eye(n_factors),
			                np.dot(Users.T,np.dot(np.diag(Rj),Q[:,j])))

		print "Error after solving for Item Matrix:",get_error(Q,Users,Items,R)

		MSE_List.append(get_error(Q,Users,Items,R))
		print "%sth iteration is complete" %ite

	print MSE_List
	#predicted ratings
	Q_hat=np.dot(Users,Items.T)

	plt.plot(range(1,len(MSE_List)+1),MSE_List)
	plt.ylabel('Error')
	plt.xlabel('Iterations')
	plt.title('MSE v/s Iterations Graph')
	plt.show()

	return Q_hat

def recommendation(Q,Q_hat,user_id,movie_titles):
	movie=[]
	for i in range(Q_hat.shape[1]):
		if Q_hat[int(user_id)-1][i]>=3.5:
			movie.append(movie_titles[i])
	

if __name__=='__main__':
	Q,R,movie_titles=initialMatrix()
	Q_hat=runALS(Q,R,n_factors=3,n_iterations=20,lambda_=0.1)
	#recommendation(Q,Q_hat,user_id,movie_titles)
	
	print Q_hat
	print Q_hat.shape
	print Q_hat[1][1]
	uid=raw_input("enter target user id :")
	recommendation(Q,Q_hat,uid,movie_titles)