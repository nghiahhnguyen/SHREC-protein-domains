import numpy as np                                                                                     
                                                                                                            
def EuclideanDistance(vec1,vec2):                                                                      
   return np.linalg.norm(vec1-vec2)                                                                   
                                                                                                            
def calculate_distance(data):                                                                        
    distance_matrix = [[]]                                                                             
    for i in data:                                                                                     
      for j in i:                                                                                    
        distance_matrix[i][j] = EuclideanDistance(data[i,:],data[j,:])                             
    return distance_matrix 
