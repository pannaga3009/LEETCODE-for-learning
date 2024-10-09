class WordProcessor:
    def __init__(self, text=""):
        self.text = text
    
    def add(self, newtext):
        self.text += newtext
        return self.text

    def delete(self, start, end=None):
        if end == None:
            self.text = self.text[:start] + self.text[start+1:]
        else:
            self.text = self.text[:start] + self.text[end+1:]

    def display(self):
        return self.text
    
    def edit(self, start, newtext):
        # Edits text starting from the specified index
        end = start + len(newtext)
        self.text = self.text[:start] + newtext + self.text[end:]
        return self.text


if __name__ == "__main__":
    word = "Hello World"
    wp = WordProcessor(word)
    print("", wp.display())

    wp.add(" Pannaga is going to have a great day")
    print("After Add:", wp.display())

    wp.edit(5, "-----")
    print("After Edit:", wp.display())

    wp.delete(5,10)
    print("After Delete:", wp.display())



    