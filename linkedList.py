class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def getIndex(self, index):
        if index < 0 or index > self.size:
            return -1
        
        current = self.head
        for _ in range(index):
            current = current.next
        
        return current.val
    
    def addToIndex(self, val, index):
        if index < 0 or index > self.size:
            return

        newNode = ListNode(val)
        if index == 0:
            newNode.next = self.head
            self.head = newNode
        else:
            current = self.head
            for _ in range(index-1):
                current = current.next
            newNode.next = current.next
            current.next = newNode
        self.size += 1
    
    def deleteIndex(self, index):
        if index < 0 or index > self.size:
            return
        
        if index == 0:
            self.head = self.head.next
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            current.next = current.next.next
        self.size -= 1


