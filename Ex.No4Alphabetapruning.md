# Ex.No: 4   Implementation of Alpha Beta Pruning 
### DATE:                                                                            
### REGISTER NUMBER : 212222040106 
### AIM: 
Write a Alpha beta pruning algorithm to find the optimal value of MAX Player from the given graph.
### Steps:
1. Start the program
2. Initially  assign MAX and MIN value as 1000 and -1000.
3.  Define the minimax function  using alpha beta pruning
4.  If maximum depth is reached then return the score value of leaf node. [depth taken as 3]
5.  In Max player turn, assign the alpha value by finding the maximum value by calling the minmax function recursively.
6.  In Min player turn, assign beta value by finding the minimum value by calling the minmax function recursively.
7.  Specify the score value of leaf nodes and Call the minimax function.
8.  Print the best value of Max player.
9.  Stop the program. 

### Program:
```
MAX, MIN = 1000, -1000 

def minimax(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex] 
 
    if maximizingPlayer: 
        best = MIN
        
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta) 
            best = max(best, val) 
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta) 
            best = min(best, val) 
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best 
 
# Driver Code 
if __name__ == "__main__":
    values = [2, 3, 4, 5, -1, 4, 2, 6]
    print("The optimal value is :", minimax(0,0,True,values,MIN , MAX))
```








### Output:

![image](https://github.com/user-attachments/assets/93277ad3-c5bb-4ef9-a037-49f578b6714c)


### Result:
Thus the best score of max player was found using Alpha Beta Pruning.
