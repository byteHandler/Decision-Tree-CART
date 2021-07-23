import numpy as np
import math
import pydot
from IPython.display import Image
class TreeNode:
    '''
    Stores information about each node of the decision tree including links to its child and parent
    Links to childs are stored in the form of a linked list
    '''
    def __init__(self,spl_info,title,condition):
        '''
        Constructor details : Initializes a node of the decision tree

        spl_info : Split information about a node including selected attribute , gain ratio and threshold
        title : in case of internal nodes , it stores the name of selected attribute , for leaf not it stores the name of target predicted class
        condition : it stores condition of split
        '''
        self.title = title
        self.spl_info = spl_info
        self.parent = None
        self.children = []
        self.condition = condition
    def add_child(self, node):
        '''
        Adds the specified node to the child list of current node and assigns its parent to current node

        node : a TreeNode object
        '''
        self.children.append(node)
        node.parent = self
class decisionTree:
    def __init__(self, X_data , Y_data,aType,colNames,atrnames):
        '''
        Constructor details : Initializes the decision tree classifier

        X_data : Dataset in numpy array format excluding the target attribute
        Y_data : Target attribute column of the dataset in numpy array format
        aType : a list containing 0,1 , 0 if X_data[i] is categorical , 1 if numeric
        colNames : a list of strings representing column names for each column in X_data parameter
        atrnames : target attribute column is expected to have encoded values such as 0,1,2 in place of lets say setosa,versicolor,virginica, the original names are given in this attribute in form of list of strings 
        '''
        self.root = None
        self.X_data = X_data
        self.Y_data = Y_data
        self.aType = aType
        self.colNames = colNames
        self.atrnames=atrnames
    def entropy(self,target):
        '''
        Calculates the entropy given the data in form of a  1D numpy array vector 'target'

        target : 1 Dimensional Numpy Array containing a single row of the dataset

        returns value of the entropy of the input column
        '''
        classes = []
        for i in np.unique(target):
            classes.append((target==i).sum())
        res = 0
        #print(classes)
        s = sum(classes)
        #print(s)
        for i in classes:
            #print("{} {} {}".format(i,i/s,math.log2(i/s)))
            temp = -1 * (i/s) * math.log2(i/s)
            res+= temp
        return res
    def select_attribute(self,atrType,atrs,target,atrnames): # atrType[i] will be 0 when attribue 'i' is catrogrical , 1 otherwise
        '''
        Selects an attribtue from the dataset with maximum 'Gain Ratio' 

        INPUT :

        atrType : a list containing 0 and 1 , 0 if attribute[i] is categorical , 1 if it is numeric
        atrs : a 2D np array matrix containing dataset except target variable
        target : a np array consisting of target attribute data
        atrnames : a list of strings , containing names of attributes
==========================================================================================================================
        OUTPUT :

        a dictionary containing following information

        name : Name of selected attribute for split
        index : column wise index of the selected attribute
        threshold : threshold in case of numeric attribute , set to None if attribute selected is categorical
        gain : Information Gain of selected attribtue (this function currently selects attribute according to gain ratio)

        '''
        #print("#SAMPLES : {} N_attrs : {}".format(len(target),atrs.shape[1]))
        #if len(target) <20:
           # print("SAMPLES : {} {}".format(target,atrs))
        target = np.array(target)
        nrows = len(target)
        atrs = np.array(atrs)
        information_gains=[] # will store the values of gain ratio for each attribute
        ed = self.entropy(target)
        thresholds=[None for i in range(atrs.shape[1])]
        for i in range(atrs.shape[1]):
            curr = np.array(list(zip(atrs[:,i],target)))
            # (5.6 , 0)
            if atrType[i]==0: # Categorical Attribute
                atr_entropies = [] # will store entripy 
                spl_info_nums = []
                unique_vals = np.unique(atrs[:,i])
                for j in unique_vals:
                    selection = curr[list(np.where(curr[0:,]==j)[0]),1] # select rows with unique value equal to j 
                    # selection = [0,1,0,1,1,2,0,1,1,1,1]
                    atr_entropies.append((self.entropy(selection),len(selection)))
                    #spl_info_nums.append(len(selection)) # WAS SELECTIONS EARLIER 
                ig = 0
                for j in atr_entropies:
                    ig+=((j[1]/nrows)*j[0])
                ig = ed - ig
                #split_info = self.entropy(spl_info_nums)
                information_gains.append(ig)
                #print("Do something")
            else: # Numeric Attribute
                temp_gain_ratios=[] # will store gain ratio for each candidate attribute , we will choose the best one
                sorted_curr = curr[curr[:,0].argsort()] # sort the array with respect to the column value
                candidate_thresholds = [] # candidate threshold values we will calculate entropies for
                for j in range(1,len(sorted_curr)):
                    if sorted_curr[j][1] != sorted_curr[j-1][1]:
                        thres = (sorted_curr[j][0]+sorted_curr[j-1][0])/2
                        candidate_thresholds.append(thres)
                # now as we have got our candidate threshold values , we will calculate entropy for each of these values
                for j in candidate_thresholds:
                    selection_1 = curr[list(np.where(curr[0:,0]<=j)[0]),1]
                    selection_2 = curr[list(np.where(curr[0:,0]>j)[0]),1] # THIS LINE OF CODE CAN BE OPTIMIZED
                    entropy_1 = self.entropy(selection_1)
                    entropy_2 = self.entropy(selection_2)
                    #temp_split_info = self.split_infoo([len(selection_1),len(selection_2)])
                    ig = ed - (((len(selection_1)/nrows)*entropy_1) + ((len(selection_2)/nrows)*entropy_2))
                    #temp_gain_ratio = ig /temp_split_info
                    temp_gain_ratios.append(ig)
                best_index = temp_gain_ratios.index(max(temp_gain_ratios))
                information_gains.append(temp_gain_ratios[best_index])
                thresholds[i]=candidate_thresholds[best_index]
                #print("Do something")
        #print(len(information_gains))
        max_index = information_gains.index(max(information_gains))
        spl_info={'name':atrnames[max_index],'index':max_index , 'threshold':thresholds[max_index],'gain':information_gains[max_index]}
        return spl_info
    def split_infoo(self,args):
        '''
        Calculates split info of the column , with count if each unique value specified

        args : a list of integers containing count of each unique variables 

        returns split info for the specified counts
        '''
        s = sum(args)
        temp =0
        for i in args:
            try:
                temp += (-1 * (i/s) * math.log2(i/s))
            except ValueError:
                print("VALUE ERROR : i-{} s-{}".format(i,s))
                continue
        return temp
    def split_data(self,atrs,target,split_info):
        '''
        Splits the given dataset according to  selected attribute
        atrs : Dataset in numpy array format (all columns except target) which needs to be split
        target : Target attribute column of dataset in numpy array format
        split_info : 
                A dictionary containing following information

                name : Name of selected attribute for split
                index : column wise index of the selected attribute
                threshold : threshold in case of numeric attribute , set to None if attribute selected is categorical
                gain : Gain Ratio of selected attribtue (this function currently selects attribute according to gain ratio)
        
        returns two datasets after performing split accordingly
        '''
        if split_info['threshold'] != None:
            split_1_indexes = list(np.where(atrs[0:,split_info['index']]<=split_info['threshold']))
            split_1_X = atrs[split_1_indexes,:]
            split_1_Y = target[tuple(split_1_indexes)]
            split_2_indexes = list(np.where(atrs[0:,split_info['index']]>split_info['threshold']))
            split_2_X = atrs[split_2_indexes,:]
            split_2_Y = target[tuple(split_2_indexes)]
            split_1_X = split_1_X[0]
            split_2_X = split_2_X[0]
            #split_1_Y = split_1_Y[0]
            #split_2_Y = split_2_Y[0]
            #split_1_X = np.delete(split_1_X , split_info['index'],axis=1) # Dropping the column belonging to selected attribute
            #split_2_X = np.delete(split_2_X , split_info['index'],axis=1)
            return [(split_1_X,split_1_Y),(split_2_X,split_2_Y)]
        else:
            splits = []
            unique_vals = np.unique(atrs[:,split_info['index']])
            for i in unique_vals:
                split_indexes = list(np.where(atrs[0:,split_info['index']]==i))
                split_X = atrs[split_indexes,:]
                split_Y = target[tuple(split_indexes)]
                split_X = split_X[0]
                splits.append((split_X,split_Y,i))
            return splits
    def build_tree(self,X,Y,root,atrType,atrnames,condition):
        '''
        Recursively uses select_attribute() and split_data() functions to generate decision tree
        '''
        if len(np.unique(Y))==1:
            #print("CHILD ADDED")
            root.add_child(TreeNode(None , Y[0],condition))
            return
        split= self.select_attribute(atrType,X,Y,atrnames)
        #print(split)
        if split['gain']==0: # Base condition of recursion
            return
        splits = self.split_data(X,Y,split)
        temp = atrnames.copy()
        #temp.remove(split['name'])
        if root == None:
            self.root = TreeNode(split,split['name'],condition)
            #self.build_tree(X1,Y1,self.root,atrType,temp)
            #self.build_tree(X2,Y2,self.root,atrType,temp)
            if split['threshold'] != None:
                self.build_tree(splits[0][0],splits[0][1],self.root,atrType,temp,"{}<={}".format(split['name'],split['threshold']))
                self.build_tree(splits[1][0],splits[1][1],self.root,atrType,temp,"{}>{}".format(split['name'],split['threshold']))
            else:
                for i in splits:
                    self.build_tree(i[0],i[1],self.root,atrType,temp,"{}=={}".format(split['name'],i[2]))
            #for i in splits:
            #    self.build_tree(i[0],i[1],self.root,atrType,temp)
        else:
            root.add_child(TreeNode (split,split['name'],condition))
            if split['threshold'] != None:
                self.build_tree(splits[0][0],splits[0][1],root.children[-1],atrType,temp,"{}<={}".format(split['name'],split['threshold']))
                self.build_tree(splits[1][0],splits[1][1],root.children[-1],atrType,temp,"{}>{}".format(split['name'],split['threshold']))
            else:
                for i in splits:
                    self.build_tree(i[0],i[1],root.children[-1],atrType,temp,"{}=={}".format(split['name'],i[2]))
            #child1 = self.select_attribute(list(np.ones(len(atrType)-1)),X_1,Y_1,atrnames.copy().remove(split['Name']))

    def fit(self):
        '''
        Builds the decision tree model
        '''
        self.build_tree(self.X_data,self.Y_data,self.root,self.aType,self.colNames,"ROOT")
        self.get_graph()
    def print_tree(self,root):
        '''
        Prints the tree in Pre-Order Traversal format , in case of internal node it prints selected attribute , split information and its children
        In case of Leaf node it prints the predicted target variable
        '''
        if root.children != []:
            print("TITLE : {}".format(root.title))
            print("SPLIT INFO : {}".format(root.spl_info))
            print("CHILDREN : {}".format(root.children))
        else:
            print("LEAF NODE : {}".format(root.title))
        for i in root.children:
            self.print_tree(i)
    def display(self):
        '''
        Prints the decision tree in Pre-Order Traversal format , it simply calls the print_tree() function , this function was created so that 
        decision tree can simply be printed using display() without having to pass any parameters
        '''
        self.print_tree(self.root)
    def predict(self,values): # a list containing a values of dependent variables (IN SAME ORDER AS THAT OF DATASET)
        '''
        Predicts the target attribute using the sample data mentioned in values, by traversing the tree until a leaf node is reached

        values : a numpy array or list of sample values in the same order as that of dataset 
        '''
        values_dict = {}
        for i in range(len(values)):
            values_dict[self.colNames[i]]=values[i]
        predict_root = self.root
        while predict_root.spl_info != None:
            temp = values_dict[predict_root.spl_info['name']]
            if predict_root.spl_info['threshold'] != None:
                 if temp <= predict_root.spl_info['threshold']:
                    predict_root = predict_root.children[0]
                 else:
                    predict_root = predict_root.children[1]
            else:
                for child in predict_root.children:
                    if temp == child.spl_info['condition'].split("==")[1]:
                        predict_root = child
                        break
        return predict_root.title
    def get_accuracy(self , test_x,test_y):
        '''
        Gives accuracy of the dataset by testing the model on the input dataset

        test_x : Testing dataset containing only dependent attributes
        test_y : Testing dataset containing only target attribute
        '''
        mis_classifications = 0 
        for i in range(len(test_x)):
            model_prediction = self.predict(test_x[i])
            if model_prediction != test_y[i]:
                mis_classifications += 1
        return (100-((mis_classifications/len(test_x))*100))
    def plot(self,root,graph):
        '''
        Generates a pydot graph of the decision tree 
        Traversing the tree in Pre-Order Way and appending nodes and edges to pydot graph object accordingly
        '''
        nodeId = repr(root).split(" ")[-1][:-1]
        if root.children != []:
            nodelabel = root.spl_info['name']+'\n'+"Gain :"+str(root.spl_info['gain'])+'\n'+"Threshold :"+str(root.spl_info['threshold'])
            node = pydot.Node(name=nodeId , label=nodelabel ,color = 'red',shape='rectangle')
            graph.add_node(node)
        else:
            node = pydot.Node(name=nodeId , label=self.atrnames[root.title] ,color='green')
            graph.add_node(node)
        if root.parent != None:
            parent_id = repr(root.parent).split(" ")[-1][:-1]
            edge = pydot.Edge(parent_id,nodeId,label=root.condition)
            graph.add_edge(edge)
        for i in root.children:
            self.plot(i,graph)
    def get_graph(self):
        graph = pydot.Dot('DecisionTree',graph_type='graph',bgcolor='white')
        self.plot(self.root, graph)
        graph.write_png('decisiontree.png')
    def display_tree(self):
        Image('decisiontree.png')