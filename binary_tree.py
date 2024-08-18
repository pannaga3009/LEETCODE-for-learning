from collections import deque


class TreeNode:
    def __init__(self,val=0, left=None, right=None):
        self.left = left
        self.right = right
        self.val = val

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
        



if __name__ == "__main__":
    inorder = [9,3,15,20,7]
    preorder = [3,9,20,15,7]
    sol = Solution()
    tree = sol.buildTree(preorder, inorder)
    output = sol.levelOrderWithNulls(tree)
    
    print(" Binary Tree from Preorder and Inorder Traversal:")
    print("Level Order Traversal with Nulls:", output)
        