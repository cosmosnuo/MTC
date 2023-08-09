
MTC: Multi-view Trace Clustering in Event Logs

Python packages required
	numpy (http://www.numpy.org/)
	networkx (https://networkx.org/)
	scipy (https://scipy.org/)
	torch (https://pytorch.org/)
	sklearn(https://scikit-learn.org/)
	pm4py(https://pm4py.fit.fraunhofer.de/)

How to use **************************************************

Command line usage:

	First:
	      getdata.py [-f value]
	      options:
		    -f filename -- Log file (default Hospital_log.xes)

	Second:
          Cluster.py [-df value] [-n value]
          options:
                -df datafile -- file of feature dataset (default hospital_overview.npz)
                -n n_clusters -- the final number of target classes, integer (default 4)

Examples:

	First:
	      getdata.py -f Hospital_log.xes
	Second:
	      Cluster.py -df hospital_overview.npz -n 4

Explain:

	After  the First step running, some files would be generated, such as:
	        hospital_overview.npz -- Feature matrix

	Then, the Second step will start training the model for trace clustering.
 	
Note: 
	
	here are other parameters that can be adjusted in Cluster.py, as explained below:
          -k -- number of k in k-nn (default 3)
          -pe ptrain_epochs -- number of pre-train_epochs (default 2000)
          -te train_epochs -- number of train_epochs (default 400)
          -hv hidden_dimsV -- list of V1 hidden dimensions (default [64, 64])
          -hd hidden_dims -- list of feature hidden dimensions (default [64, 64])
          -la1 lambda1 -- Rate for gtr (default 0.001)
          -la2 lambda2 -- Rate for clu (default 1)

A reference for users **************************************************

	If you have any quetions for this code, you can email:XXXX.
	We would also be happy to discuss the topic of tace clustering with you.
