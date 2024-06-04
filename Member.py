class Member:
    def __init__(self, member_id, member_name):
        self.member_id = member_id
        self.member_name = member_name
        self.borrowed_books = []

    def borrow_book(self, book):
        self.borrowed_books.append(book)
        print(f"{self.member_name} has borrowed {book.title}.")

    def return_book(self,book):
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            print(f"Book is returned {book.title}")
        else:
            print(f"{book.title} is not found")
    
    def display_borrowed_books(self):
        if self.borrowed_books:
            print(f"{self.member_name} has borrowed following books")
            for book in self.borrowed_books:
                book.display_info()
        else:
            print(f"{self.member_name} has not borrowed any books.")



