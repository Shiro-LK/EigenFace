# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:11:04 2017

@author: Shiro
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
class Face_Recognition():
    def __init__(self):
        self.width = 0
        self.height = 0
        self.mod = 0
        self.information = []
        
    def load_data_from_file(self, file, noise=False):
        '''
            load data from file
            return : X list of images (n_features, n_samples)
                     Y list of labels (labels, )
        '''
        f = open(file, 'r')
        data = [line.split() for line in f]
        size = (cv2.imread(data[0][0], 0)).shape
        self.width = size[1]
        self.height = size[0]
                        
        Y = np.zeros((len(data), ), dtype=int)
        X = np.empty(shape=(size[0]*size[1], len(data)), dtype='float64') 
        for i, ldata in enumerate(data):
            Y[i] = int(data[i][1])
            img = cv2.imread(data[i][0], 0)
            if noise == True:
                mean = 0
                sigma = 100
                gauss = np.random.normal(mean,sigma,size)
                img = img + gauss
            X[:,i] = img.flatten()[:]
        return X, Y
    
    
        
    def PCA(self, X, Y):
        '''
            Compute the PCA of the training sets
            X : dim N² x M , N² dim image, M number of images
            Y : list of labels, dim M
            mean : dim N² x 1 (features, 1)
            phi : dim : N² x M, (features, n_sample)
            u : dim N² x M, ui dim N² x 1 , (features, n_sample)
            W : dim K x M (features_weights, n_sample)
        '''
        self.y = Y

        #print('X dim :', X.shape)
        # -- Average vector -- #
        self.average = self.mean_train(X)
        #print('mean dim :', self.average.shape)
        
        # -- Substract mean to image -- #
        phi = X - self.average
        
        # -- Compute EigenValue and EigenVectors for A.T A-- #
        D = np.dot(phi.T, phi)
        #print('phi dim :', phi.shape)
  
        eigenValues, eigenVectors = np.linalg.eig(D)
        #print('EigenValues : ', eigenValues.shape)
        #print('EigenVectors : ', eigenVectors.shape)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx].reshape(-1,1)
        eigenVectors = eigenVectors[:,idx]
        # information
        self.information = eigenValues/eigenValues.sum()
        #print('EigenValues :', eigenValues)
        
        # -- Compute EigenValue and EigenVectors -- #
        u = np.dot(phi, eigenVectors)
        u /= np.linalg.norm(u, axis=0)
        #print('eigenvectors finals : ', u.shape)
        self.eigenvectors = u
        
        # -- Compute weight vectors -- #
        self.W = np.dot(u.T, phi)
        #print(' w shape : ',self.W.shape) # ligne = wi, colonne numero image
        
        # -- Save variables -- #
        
        self.U = u
        #print('W :', self.W)
        
    def mean_train(self, X):
        '''
            compute the mean of the training data
            X : (n_features, n_samples)
            return mean (mean_features, 1)
        '''
        return X.mean(axis=1).reshape(-1, 1)
        
    # -- prediction and evaluation -- #    
    
    def predict(self, X, K):
        '''
            X : data image N² x M (n_features, n_samples)
            K : number of dimension to keep
            return label (n_sample, )
        '''
        # -- Weight of the test sample -- #
        wtest = self.compute_weights_test(X, K)
        #print('wtest :', wtest.shape)
        
        # -- Compute the euclideane distance between the weight of our training sample and the testing sample -- #
        
        dist = self.distance_euclideane(wtest, self.W, K)
        
        # -- Compute the closest face recognition -- #
        preds = self.min_distance(dist)
        
        return preds
        
    def compute_weights_test(self, x, K): ## for all image
        '''
            X = (n_features, n_sample)
            K : number of dimension to keep
            w _tes dim : K x M 
        '''    
        #print('x test : ', x.shape)
        
        w_test = np.dot(self.eigenvectors.T, x-self.average)
        return w_test
    
    def distance_euclideane(self, wtest, W, K):
        '''
            wtest = (weights, n_sample_test)
            W = (weights, n_sample_train)
            K : K best weights (number of dimension to keep)
            return : the distance euclideane between each test_image and train_image
        '''
        #print('shape', wtest.shape, W.shape)
        dist = []
        for i in range(0, wtest.shape[1]):
            dist.append(np.sum(np.square(W[:K, :] - wtest[:K,i].reshape(-1,1)), axis=0))
        #return np.sum(np.square(np.transpose(np.repeat(a[:, np.newaxis], b.shape[1], axis=1))-b[0:K,:]), axis=0)
        return  np.asarray(dist)
    def min_distance(self, dist):
        '''
            Compute the closest face with the euclidean distance
            
        '''
        return self.y[np.argmin(dist, axis=1)]
    def evaluate(self, x, y, K=10):
        '''
            evaluate the accuracy of recognition
            X : sample test
            Y : label test
            return : accuracy
        '''
        pred = self.predict(x, K)
        good = (pred==y).sum()
      
        return 1.0*good/len(y)
        
    
    # -- Display -- # 
    def display_all_eigenfaces(self, number_images, K):
        '''
            number_images : number in the image training we want to obtain
            K : number of weight 
            print the different reconstruction of the image from the weights of the eigenface
        '''
        n = round(K/5)
        
        for i in range(1, K+1):
            #images.append(self.display_eigenface(number_images, i)) cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            img = self.display_eigenface(number_images, i) #cv2.cvtColor(self.display_eigenface(number_images, i).astype(np.float32),cv2.COLOR_GRAY2RGB)#
            fig= plt.subplot(n, 5, i)
            plt.subplots_adjust(hspace = .1)

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.imshow(img,'gray')
            plt.title('K = '+str(i))
        plt.show()
        plt.savefig('test.jpg')
        
    def display_eigenface(self, number_images, K):
            '''
                number_images : number in the image training we want to obtain
                K : number of weight 
                return : the reconstruction of the image from the weights of the eigenface
            '''
            w = self.W[0:K,number_images].reshape(-1,1).T
            u = self.U[:,0:K]
            #print((u*w).sum(axis=1).shape)
            img = (u*w).sum(axis=1).reshape(-1,1) + self.average
            return cv2.convertScaleAbs(img.reshape(112,92))
    def save_eigenface(self, number_images, K, name):
            '''
                number_images : index of the image in the training set we want to obtain
                K : number of dimension to keep (weights) rapport
                do : Save image in the folder of the script
            '''
            img = self.display_eigenface(number_images, K)
            cv2.imwrite(name+'.jpg', img)

    # -- Apprentissage -- #
    
    def fit(self, K):
            '''
                train an svm model on the eigenface from the training set
                K : number of dimension to keep in the dataset train
            '''
            self.mod = SVC(C=1.0, kernel='linear')
            self.mod.fit(np.transpose(self.W[:K, :]), self.y)
           
    def predict_svm(self, X, K):
        '''
            return : the prediction of the SVM model
            X : images to train N x M with M the number of images, N the number of pixel per image
            K : number of dimension to keep for the prediction and evaluation
        '''
        w_test = self.compute_weights_test(X, K)
        preds = self.mod.predict(np.transpose(w_test[:K, :]))
        return preds
    
    def evaluate_svm(self, X, Y, K):
        '''
            Evaluate the SVM model
            X : images to train N x M with M the number of images, N the number of pixel per image
            Y : label of the image
            K : number of dimension to keep for the prediction and evaluation
        '''
        preds = self.predict_svm(X, K)
        res = accuracy_score(Y, preds)
        #res_f1 = f1_score(Y, preds, average='samples')
        return res, preds
def __main__():
    mod = Face_Recognition()
    k = 39
    X_train, Y_train = mod.load_data_from_file('train40.txt')
    X_test, Y_test = mod.load_data_from_file('test40.txt', noise=False)
    X_val, Y_val = mod.load_data_from_file('validation40.txt', noise=False)
    mod.PCA(X_train, Y_train)
    print('accuracy test dist euclidienne : ', mod.evaluate(X_test, Y_test, K = k ))
    #mod.display_all_eigenfaces(0, 20)
    print('accuracy validation dist euclidienne : ', mod.evaluate(X_val, Y_val, K = k ))
    ## Apprentissage
    mod.fit(K = k)
    res, preds = mod.evaluate_svm(X_train, Y_train, k)
    print('accuracy SVM train: ', res)
    res, preds = mod.evaluate_svm(X_test, Y_test, k)
    print('accuracy SVM test: ', res)
    #print(preds)
    #print(mod.information[0:39].sum())
    #mod.save_eigenface(0,40,'40')
__main__()
