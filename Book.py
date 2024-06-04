from Library import Library
from Member import Member
"""
You are designing a library management system. The system should have the following features:

Book: This class should have attributes for the title, author, and ISBN. It should also have a method to display book details.
Member: This class should have attributes for the member name, member ID, and a list of borrowed books. It should have methods to borrow and return books.
Library: This class should manage the collection of books and the members. It should have methods to add books, register members, lend books to members, and accept returned books.
Please write the class definitions for Book, Member, and Library in Python.
"""

class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn

    def display_info(self):
        print(f"The title of the book  is {self.title}, author is {self.author} and isbn is {self.isbn}")



if __name__ == "__main__":
    library = Library()

    book1 = Book("Best Techniques", "Pannaga", 3009)
    book2 = Book("TTD", "Taylor swift", 1989)

    book1.display_info()
    library.add_book(book1)
    library.add_book(book2)

    member1 = Member(1997, "Mehaa" )
    member2 = Member(2005, "Poorni")

    library.register_member(member1)
    library.register_member(member2)

    library.lend_books(book1, member1)

    # Display borrowed books
    member1.display_borrowed_books()

    # Return the book to the library
    library.accept_returned_book(book1, member1)

# Display borrowed books again
    member1.display_borrowed_books()


