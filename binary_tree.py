from collections import deque

class TreeNode:
    def __init__(self, val=0, left = None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder, inorder):
        if not inorder or not preorder:
            return

        #Get the root node from preorder list
        root = TreeNode(preorder[0])

        #Finding the index inorder to recognize left subtree and right subtree
        mid = inorder.index(preorder[0])

         # Recursively construct the left subtree and right subtree
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid+1:])

        # Return the root node of the constructed binary tree
        return root
    
    def levelOrderWithNulls(self, root):
        """
        BFS 
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)

        # Removing trailing None values to match the expected output format
        while result and result[-1] is None:
            result.pop()

        return result
    
    def minDepth(self, root):
        """
        Given a binary tree, find its minimum depth.
        The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
        Note: A leaf is a node with no children.       
        """
        def dfs(root):
            #If the current root is None, that means we have reached beyond leaf nodes
            #So we can return a very large number to ignore this path in the min calculations
            if root is None:
                return float("inf")
    
            #If the current node has no children, then return depth 1
            if root.left is None and root.right is None:
                return 1
            
            #recursively find the min depth of the left subtree
            left_depth = dfs(root.left)
            right_depth = dfs(root.right)

            #Return minimum of the depths from the left and right subtree
            #add one for the current node's depth
            return 1 + min(left_depth, right_depth)
                    
        if not root:
            return 0

        return dfs(root)
        
    def levelOrderWithoutNulls(self, root):
        """
        Given the root of a binary tree, return the level order traversal of its nodes' values.
        (i.e., from left to right, level by level).
        Input: root = [3,9,20,null,null,15,7]
        Output: [[3],[9,20],[15,7]]

        BFS
        q = deque
        """

        res = []
        q = deque()

        q.append(root)
        while q:
            q_length = len(q)
            level = []
            for i in range(q_length):
                node = q.popleft()
                if node:
                    #For every level, appending the node
                    level.append(node.val)
                    #Adding the child to the queue
                    q.append(node.left)
                    q.append(node.right)

            if level:
                res.append(level)
        return res
    
    def bstFromPreorder(self, preorder):
        """
        Constructs a binary search tree (BST) from a given preorder traversal list.
        left < root < right

        """
        n = len(preorder)

        root = TreeNode(preorder[0])
        stack = [root]

        for i in range(1, n):
            node, child = stack[-1], TreeNode(preorder[i])

            #If the child val > parent, then pop the stack until we find the right parent
            while stack and stack[-1].val < child.val:
                node = stack.pop()

            #BST logic  for left child and right child
            if node.val < child.val:
                node.right = child # The child should be the right child of the parent
            else:
                node.left = child # The child should be the left child of the parent

            stack.append(child)       
        return root 







if __name__ == "__main__":
    inorder = [9,3,15,20,7]
    preorder = [3,9,20,15,7]
    sol = Solution()
    tree = sol.buildTree(preorder, inorder)
    output = sol.levelOrderWithNulls(tree)
    
    print("Level order : ", sol.levelOrderWithoutNulls(tree))
    print("Minimum depth tree: ", sol.minDepth(tree))
    print(" Binary Tree from Preorder and Inorder Traversal:")
    print("Level Order Traversal with Nulls:", output)
    t =  sol.bstFromPreorder([8,5,1,7,10,12])
    print("bstFromPreorder --- ", sol.levelOrderWithNulls(t))
        