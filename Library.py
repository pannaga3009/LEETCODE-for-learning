class Library:
    def __init__(self):
        self.books = []
        self.members = []

    def add_book(self, book):
        self.books.append(book)
        print(f"Book titled '{book.title}' added to the library.")
    
    def register_member(self, member):
        self.members.append(member)
        print(f"Member '{member.member_name}' added to the library: '{member.member_name}' registered with ID '{member.member_id}")

    def lend_books(self, book, member):
        if book in self.books:
            self.books.remove(book)
            member.borrow_book(book)
            print(f"Book titled '{book.title}' lent to member '{member.member_name}'.")
        else:
            print(f"Book titled '{book.title}' is not available in the library.")

    def accept_returned_book(self, book, member):
        if book not in self.books:
            self.books.append(book)
            member.return_book(book)
            print(f"Book titled '{book.title}' returned by member '{member.member_name}'.")
        else:
            print(f"Book titled '{book.title}' was not borrowed from the library.")


