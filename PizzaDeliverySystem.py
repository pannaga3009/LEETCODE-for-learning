import uuid
from typing import List


"""
Let's go through the process of designing the pizza delivery system in Python

Order: Represents an order placed by a customer
Pizza: Represents a pizza in the order
Customer: Represents a customer placing the order
Payment: Represents the payment details for the order
DeliveryPerson: Represents the person responsible for delivering the order
OrderManager: Handles the creation and status management of orders
KitchenManager: Manages the preparation of pizza
DeliveryManager:  Manages the assignment and tracking of deliveries
PaymentProcessor: Handles payment processing

"""
#Define the customer class
class Customer:
    def __init__(self, name, contact_info):
        self.customer_id = uuid.uuid4()
        self.name = name
        self.contact_info = contact_info
        self.order_history = []

    def add_order_to_history(self, order):
        self.order_history.append(order)


class Pizza:
    def __init__(self, size, crust_type, toppings):
        self.pizza_id = uuid.uuid4()
        self.size = size
        self.crust_type = crust_type
        self.toppings = toppings

class Payment:
    def __init__(self, amount, payment_method):
        self.payment_id = uuid.uuid4()
        self.amount = amount
        self.payment_method = payment_method
        self.status = "Pending"

    def process_payment(self):
        self.status = "Completed"
        print(f"Payment of {self.amount} processed using {self.payment_method}")


class DeliveryPerson:
    def __init__(self, name):
        self.delivery_person_id = uuid.uuid4()
        self.name = name
        self.current_order = None
    
    def assign_order(self, order):
        self.current_order = order
        print(f"Delivery person {self.name} assigned to order {order.order_id}")

class Order:
    def __init__(self, customer: Customer, pizzas: List[Pizza], payment:Payment, delivery_address: str):
        self.order_id = uuid.uuid4()
        self.customer = customer
        self.pizzas = pizzas
        self.status = "Placed" #Default status
        self.payment = payment
        self.delivery_address = delivery_address
        self.delivery_person = None

    def update_status(self, status: str):
        self.status = status
        print(f"Order {self.order_id} status updated to {self.status}")

class OrderManager:
    def create_order(self, customer: Customer, pizzas: List[Pizza], payment: Payment, delivery_address: str):
        order = Order(customer, pizzas, payment, delivery_address)
        customer.add_order_to_history(order)
        print(f"Oreder {order.order_id} created for customer {customer.name}")
        return order
    
    def update_order_status(self, order: Order, status: str):
        order.update_status(status)

    def get_order_status(self, order: Order):
        return order.status

class KitchenManager:
    def prepare_pizza(self, pizza: Pizza):
        print(f"Preparinf pizza {pizza.pizza_id} with size {pizza.size}, crust {pizza.crust_type}, and toppings {pizza.toppings}")

    def notify_order_ready(self, order: Order):
        order.update_status("ReadyForDelivery")
        print(f"Order {order.order_id} is ready for delivery")

class DeliveryManager:
    def assign_delivery(self, order: Order, delivery_person: DeliveryPerson):
        delivery_person.assign_order(order)
        order.delivery_person = delivery_person
        order.update_status("Out for Delivery")

    def track_delivery(self, order: Order):
        print(f"Tracking delivery for order {order.order_id}")

    def update_delivery_status(self, order: Order, status: str):
        order.update_status(status)

class PaymentProcessor:
    def process_payment(self, payment: Payment):
        payment.process_payment()

    def refund_payment(self, payment:Payment):
        payment.status  = "Refunded"
        print(f"Payment {payment.payment_id} has been refunded")

if __name__ == "__main__":
    customer = Customer(name="Jane", contact_info="jan@example.com")

    #Create a pizza
    pizza1 = Pizza(size="Large", crust_type="Thin", toppings=["Cheese", "Pepporoni"])
    pizza2 = Pizza(size="Medium", crust_type="Thick", toppings=["Mushrooms", "Onions"])

    #process_payment
    payment = Payment(amount=25.05, payment_method="Credit Card")
    payment_processor = PaymentProcessor()
    payment_processor.process_payment(payment)

    order_manager = OrderManager()
    order = order_manager.create_order(customer, [pizza1, pizza2], payment, "123 Main St")

    kitchen_manager = KitchenManager()
    kitchen_manager.prepare_pizza(pizza1)
    kitchen_manager.prepare_pizza(pizza2)

    #Assign delivery
    delivery_person = DeliveryPerson(name="Rose smith")
    delivery_manager = DeliveryManager()
    delivery_manager.assign_delivery(order, delivery_person)

    delivery_manager.track_delivery(order)
    delivery_manager.update_delivery_status(order, "Delivered")