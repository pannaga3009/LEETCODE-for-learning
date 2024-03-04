class SongTracker:
    def __init__(self):
        """Intialzing a dict with songs and count the recently played one
        """       
        self.play_count = {}
        
    
    def play_song(self, song):
        self.play_count[song] = self.play_count.get(song, 0) + 1
        print("---- adding song---", self.play_count)

    def most_played_song(self):
        if not self.play_count:
            return 
        
        most_played = max(self.play_count, key = self.play_count.get)
        return most_played
    
    def isValid(self, s: str) -> bool:
        valid_dict = {'{':'}', '(':')', '[':']'}
        stack = []
        stack.append(s[0])

        for i in range(1, len(s)):
            if stack and valid_dict.get(stack[-1]) == s[i]:
                stack.pop()
            else:
                stack.append(s[i])
        
        return len(stack) == 0

song_t = SongTracker()
song_t.play_song('Song 1')
song_t.play_song("Song 2")
song_t.play_song("Tylor swift song")
song_t.play_song("Tylor swift song")
song_t.play_song("Tylor swift song")
song_t.play_song("Song 2")
print("Most played is" , song_t.most_played_song())


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root) -> bool:

        def dfs(root):
            if root is None:
                return 0
            left_depth = dfs(root.left)
            right_depth = dfs(root.right)
            return 1 + max(left_depth, right_depth)


        if root is None:
            return True
        
        left_depth = dfs(root.left)
        right_depth = dfs(root.right)

        if abs(left_depth - right_depth) <= 1:
            return self.isBalanced(root.left) and self.isBalanced(root.right)
        return False
    
# Create the tree from the given list
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

# Instantiate the Solution class
sl = Solution()

# Call the isBalanced method with the root of the tree
result = sl.isBalanced(root)
print(result)


